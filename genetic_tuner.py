# =============================================================================
# genetic_tuner.py
# Genetic Algorithm Hyperparameter Tuner for FinPulse CNN-GRU-LSTM Model
# =============================================================================
#
# HOW TO RUN:
#   python genetic_tuner.py --ticker RELIANCE.NS --generations 10 --population 12
#
# OUTPUT:
#   - Prints best hyperparameters found
#   - Saves results to ga_results_<TICKER>.json
#   - Prints ready-to-paste hyperparameters.py entry
# =============================================================================

import numpy as np
import random
import json
import copy
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import torch

# ── Project imports ────────────────────────────────────────────────────────────
from data_loader         import fetch_all_stocks_sequential
from feature_engineering import build_features, create_sequences
from ml_model            import train_model, DEVICE

# =============================================================================
# SEARCH SPACE  — every gene and its allowed values
# =============================================================================
SEARCH_SPACE = {
    "seq_len":        [20, 25, 30, 35, 40, 45, 50, 60],
    "lstm_units":     [32, 64, 96, 128],
    "gru_units":      [32, 64, 96, 128],
    "conv_filters":   [32, 64, 128],
    "dropout":        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "learning_rate":  [0.0005, 0.0007, 0.0008, 0.001, 0.0015, 0.002],
    "batch_size":     [16, 32, 64],
    "epochs":         [30, 50],          # kept low during search; real training uses more
}

# GA settings (can be overridden via CLI args)
DEFAULT_POPULATION  = 12     # number of individuals per generation
DEFAULT_GENERATIONS = 8      # how many generations to evolve
ELITE_FRACTION      = 0.2    # top X% survive unchanged
MUTATION_RATE       = 0.25   # probability any single gene mutates
TOURNAMENT_K        = 3      # tournament selection size


# =============================================================================
# CHROMOSOME HELPERS
# =============================================================================
def random_individual() -> dict:
    """Create one random set of hyperparameters (one 'chromosome')."""
    return {gene: random.choice(values) for gene, values in SEARCH_SPACE.items()}


def crossover(parent_a: dict, parent_b: dict) -> dict:
    """
    Uniform crossover: each gene is independently taken from either parent
    with 50/50 probability. Produces one child.
    """
    child = {}
    for gene in SEARCH_SPACE:
        child[gene] = parent_a[gene] if random.random() < 0.5 else parent_b[gene]
    return child


def mutate(individual: dict, mutation_rate: float = MUTATION_RATE) -> dict:
    """
    For each gene, with probability=mutation_rate, replace it with a
    random value from the allowed options for that gene.
    """
    mutant = copy.deepcopy(individual)
    for gene, options in SEARCH_SPACE.items():
        if random.random() < mutation_rate:
            mutant[gene] = random.choice(options)
    return mutant


def tournament_select(population: list, fitnesses: list, k: int = TOURNAMENT_K) -> dict:
    """
    Pick k random individuals, return the one with the best (lowest) fitness.
    Lower val_loss = better fitness.
    """
    contestants = random.sample(list(zip(population, fitnesses)), k)
    winner = min(contestants, key=lambda x: x[1])
    return copy.deepcopy(winner[0])


# =============================================================================
# FITNESS FUNCTION  — train model briefly, return validation loss
# =============================================================================
def evaluate_fitness(individual: dict, X_train, y_train, X_val, y_val,
                     n_features: int) -> float:
    """
    Train the FinPulse model with this individual's hyperparameters and
    return the final validation loss. Lower = better.
    Returns a large penalty value if training crashes.
    """
    try:
        _, history = train_model(
            X_train, y_train, X_val, y_val,
            seq_len       = individual["seq_len"],
            n_features    = n_features,
            lstm_units    = individual["lstm_units"],
            gru_units     = individual["gru_units"],
            conv_filters  = individual["conv_filters"],
            dropout       = individual["dropout"],
            learning_rate = individual["learning_rate"],
            epochs        = individual["epochs"],
            batch_size    = individual["batch_size"],
        )
        val_loss = min(history["val_loss"])

        # Free GPU memory between evaluations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return val_loss

    except Exception as e:
        print(f"      [!] Evaluation crashed: {e}")
        return 9999.0   # heavy penalty — this individual is effectively dead


# =============================================================================
# DATA PREPARATION
# =============================================================================
def prepare_data(ticker: str):
    """Fetch and preprocess data for one ticker. Returns split arrays."""
    print(f"\n[GA] Fetching data for {ticker}...")
    raw     = fetch_all_stocks_sequential([ticker], years=2)
    price_df = raw[ticker]["price_data"]
    fund     = raw[ticker]["fundamentals"]

    if price_df.empty:
        raise ValueError(f"No price data returned for {ticker}")

    raf     = fund.get("RAF", 1.0)
    feat_df = build_features(price_df.copy(), raf=raf)

    # Use maximum seq_len from search space so all individuals can be evaluated
    # on the same data split (shorter seq_lens simply use a subset of sequences)
    max_seq = max(SEARCH_SPACE["seq_len"])

    if len(feat_df) < max_seq + 20:
        raise ValueError(f"Not enough data rows ({len(feat_df)}) for seq_len={max_seq}")

    X, y, scaler_X, scaler_y, feat_cols, raw_ohlc = create_sequences(feat_df, seq_len=max_seq)

    split  = int(len(X) * 0.80)
    return (
        X[:split], y[:split],   # train
        X[split:], y[split:],   # val
        X.shape[2],              # n_features
        scaler_X, scaler_y,
    )


# =============================================================================
# GENETIC ALGORITHM  — main loop
# =============================================================================
def run_ga(ticker: str, population_size: int, generations: int) -> dict:
    """
    Runs the full Genetic Algorithm and returns the best hyperparameter dict.
    """
    print("=" * 65)
    print(f"  FinPulse Genetic Hyperparameter Tuner")
    print(f"  Ticker      : {ticker}")
    print(f"  Population  : {population_size}")
    print(f"  Generations : {generations}")
    print(f"  Device      : {DEVICE}")
    print("=" * 65)

    # ── Prepare data once (reused every generation) ────────────────────────
    X_train, y_train, X_val, y_val, n_features, scaler_X, scaler_y = prepare_data(ticker)
    print(f"[GA] Data ready — train: {len(X_train)}, val: {len(X_val)}, features: {n_features}")

    # ── Generation 0: random initial population ────────────────────────────
    population = [random_individual() for _ in range(population_size)]
    best_individual = None
    best_fitness    = float("inf")
    history_log     = []

    n_elite = max(1, int(population_size * ELITE_FRACTION))

    for gen in range(1, generations + 1):
        gen_start = time.time()
        print(f"\n{'─'*65}")
        print(f"  Generation {gen}/{generations}")
        print(f"{'─'*65}")

        # ── Evaluate every individual ──────────────────────────────────────
        fitnesses = []
        for idx, individual in enumerate(population):
            print(f"  [{idx+1:2d}/{population_size}] "
                  f"seq={individual['seq_len']} "
                  f"lstm={individual['lstm_units']} "
                  f"gru={individual['gru_units']} "
                  f"conv={individual['conv_filters']} "
                  f"drop={individual['dropout']:.2f} "
                  f"lr={individual['learning_rate']} "
                  f"bs={individual['batch_size']} "
                  f"ep={individual['epochs']}",
                  end="  →  ", flush=True)

            # Slice X/y to match this individual's seq_len if shorter than max
            sl = individual["seq_len"]
            max_sl = max(SEARCH_SPACE["seq_len"])
            if sl < max_sl:
                # trim sequences from the front (most recent sl steps)
                Xtr = X_train[:, max_sl-sl:, :]
                Xvl = X_val[:,   max_sl-sl:, :]
            else:
                Xtr, Xvl = X_train, X_val

            fit = evaluate_fitness(individual, Xtr, y_train, Xvl, y_val, n_features)
            fitnesses.append(fit)
            print(f"val_loss = {fit:.6f}")

            if fit < best_fitness:
                best_fitness    = fit
                best_individual = copy.deepcopy(individual)
                best_individual["_val_loss"] = round(fit, 6)
                print(f"  ★  New best! val_loss = {fit:.6f}")

        # ── Sort by fitness ────────────────────────────────────────────────
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1])
        gen_best_fit = ranked[0][1]
        gen_avg_fit  = np.mean(fitnesses)

        print(f"\n  Gen {gen} summary — best: {gen_best_fit:.6f} | avg: {gen_avg_fit:.6f} | time: {time.time()-gen_start:.0f}s")

        history_log.append({
            "generation":  gen,
            "best_loss":   round(gen_best_fit, 6),
            "avg_loss":    round(gen_avg_fit, 6),
            "best_params": copy.deepcopy(ranked[0][0]),
        })

        if gen == generations:
            break   # no need to breed a new population after last generation

        # ── Elitism: carry forward the top individuals unchanged ───────────
        elites = [copy.deepcopy(ind) for ind, _ in ranked[:n_elite]]

        # ── Breed next generation ──────────────────────────────────────────
        new_population = elites[:]
        while len(new_population) < population_size:
            parent_a = tournament_select(population, fitnesses)
            parent_b = tournament_select(population, fitnesses)
            child    = crossover(parent_a, parent_b)
            child    = mutate(child)
            new_population.append(child)

        population = new_population

    # ── Final result ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  GENETIC ALGORITHM COMPLETE")
    print("=" * 65)
    print(f"  Best val_loss : {best_fitness:.6f}")
    print(f"  Best params   :")
    for k, v in best_individual.items():
        if not k.startswith("_"):
            print(f"    {k:20s}: {v}")

    return best_individual, history_log


# =============================================================================
# OUTPUT HELPERS
# =============================================================================
def save_results(ticker: str, best: dict, history: list):
    filename = f"ga_results_{ticker.replace('.','_')}.json"
    out = {
        "ticker":      ticker,
        "best_params": best,
        "history":     history,
    }
    with open(filename, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[GA] Results saved → {filename}")
    return filename


def print_hyperparams_entry(ticker: str, best: dict):
    """Print a ready-to-paste block for hyperparameters.py"""
    lines = [
        f'\n{"─"*65}',
        f'  PASTE THIS INTO hyperparameters.py → STOCK_HYPERPARAMS:',
        f'{"─"*65}',
        f'    "{ticker}": {{',
        f'        "seq_len":        {best["seq_len"]},',
        f'        "epochs":         120,   # increase from GA search value for final training',
        f'        "batch_size":     {best["batch_size"]},',
        f'        "learning_rate":  {best["learning_rate"]},',
        f'        "dropout":        {best["dropout"]},',
        f'        "lstm_units":     {best["lstm_units"]},',
        f'        "gru_units":      {best["gru_units"]},',
        f'        "conv_filters":   {best["conv_filters"]},',
        f'    }},',
        f'{"─"*65}',
    ]
    print("\n".join(lines))


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm Hyperparameter Tuner for FinPulse")
    parser.add_argument("--ticker",      type=str, default="RELIANCE.NS", help="NSE ticker e.g. TCS.NS")
    parser.add_argument("--population",  type=int, default=DEFAULT_POPULATION,  help="Population size per generation")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS, help="Number of generations")
    parser.add_argument("--seed",        type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    best_params, history_log = run_ga(
        ticker          = args.ticker,
        population_size = args.population,
        generations     = args.generations,
    )

    save_results(args.ticker, best_params, history_log)
    print_hyperparams_entry(args.ticker, best_params)
