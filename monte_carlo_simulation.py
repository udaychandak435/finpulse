# import numpy as np
# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")

# np.random.seed(42)


# def simulate_portfolio_paths(
#     current_prices:   dict,    # {ticker: price}
#     quantities:       dict,    # {ticker: qty}
#     garch_vols:       dict,    # {ticker: annualised_vol}
#     log_returns_dict: dict,    # {ticker: pd.Series}
#     n_simulations:   int = 2000,
#     n_days:          int = 30,
# ) -> dict:
#     """
#     Monte Carlo simulation of portfolio value using Geometric Brownian Motion.

#     S_{t+1} = S_t × exp[(μ - 0.5σ²)Δt + σ√Δt × Z],  Z ~ N(0,1)

#     Returns:
#         - all_paths:          np.ndarray (n_simulations × n_days+1)  - portfolio value paths
#         - final_values:       np.ndarray (n_simulations,) - portfolio value at end
#         - percentile_5:       5th percentile (worst-case scenario boundary)
#         - percentile_95:      95th percentile
#         - expected_value:     mean final portfolio value
#         - current_portfolio:  starting portfolio value
#     """
#     dt      = 1 / 252
#     tickers = list(current_prices.keys())

#     # Compute μ per stock from historical log returns
#     mu_dict = {}
#     for t in tickers:
#         ret = log_returns_dict[t].dropna()
#         mu_dict[t] = float(ret.mean()) * 252   # annualise

#     current_portfolio = sum(
#         current_prices[t] * quantities.get(t, 0) for t in tickers
#     )

#     all_paths = np.zeros((n_simulations, n_days + 1))
#     all_paths[:, 0] = current_portfolio

#     for step in range(1, n_days + 1):
#         portfolio_step = np.zeros(n_simulations)

#         for ticker in tickers:
#             qty     = quantities.get(ticker, 0)
#             if qty == 0:
#                 continue
#             mu      = mu_dict[ticker]
#             sigma   = garch_vols.get(ticker, 0.20)   # default 20% vol

#             Z = np.random.standard_normal(n_simulations)
#             drift   = (mu - 0.5 * sigma ** 2) * dt
#             shock   = sigma * np.sqrt(dt) * Z
#             price_t = current_prices[ticker] * np.exp(drift * step + shock * np.sqrt(step))
#             portfolio_step += price_t * qty

#         all_paths[:, step] = portfolio_step

#     final_values = all_paths[:, -1]

#     return {
#         "all_paths":         all_paths,
#         "final_values":      final_values,
#         "percentile_5":      float(np.percentile(final_values, 5)),
#         "percentile_95":     float(np.percentile(final_values, 95)),
#         "expected_value":    float(np.mean(final_values)),
#         "current_portfolio": current_portfolio,
#         "n_simulations":     n_simulations,
#         "n_days":            n_days,
#     }






import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


def simulate_portfolio_paths(
    current_prices:   dict,
    quantities:       dict,
    garch_vols:       dict,
    log_returns_dict: dict,
    n_simulations:    int = 2000,
    n_days:           int = 30,
) -> dict:
    """
    Monte Carlo simulation of portfolio value using Geometric Brownian Motion.
    S_{t+1} = S_t × exp[(μ - 0.5σ²)Δt + σ√Δt × Z],  Z ~ N(0,1)
    """
    dt      = 1 / 252
    tickers = list(current_prices.keys())

    mu_dict = {}
    for t in tickers:
        ret = log_returns_dict[t]
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        mu_dict[t] = float(ret.dropna().mean()) * 252

    current_portfolio = sum(
        current_prices[t] * quantities.get(t, 0) for t in tickers
    )

    all_paths = np.zeros((n_simulations, n_days + 1))
    all_paths[:, 0] = current_portfolio

    for step in range(1, n_days + 1):
        portfolio_step = np.zeros(n_simulations)
        for ticker in tickers:
            qty   = quantities.get(ticker, 0)
            if qty == 0:
                continue
            mu    = mu_dict[ticker]
            sigma = garch_vols.get(ticker, 0.20)
            Z     = np.random.standard_normal(n_simulations)
            drift = (mu - 0.5 * sigma ** 2) * dt
            shock = sigma * np.sqrt(dt) * Z
            price_t = current_prices[ticker] * np.exp(drift * step + shock * np.sqrt(step))
            portfolio_step += price_t * qty
        all_paths[:, step] = portfolio_step

    final_values = all_paths[:, -1]

    return {
        "all_paths":         all_paths,
        "final_values":      final_values,
        "percentile_5":      float(np.percentile(final_values, 5)),
        "percentile_95":     float(np.percentile(final_values, 95)),
        "expected_value":    float(np.mean(final_values)),
        "current_portfolio": current_portfolio,
        "n_simulations":     n_simulations,
        "n_days":            n_days,
    }
