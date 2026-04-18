import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ── VaR Method 1: Historical Simulation ──────────────────────────────────────
def compute_var(returns: np.ndarray, portfolio_value: float, confidence: float = 0.95) -> float:
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    return abs(cutoff * portfolio_value)


# ── VaR Method 3: Parametric (Variance-Covariance) ───────────────────────────
def compute_parametric_var(returns: np.ndarray, portfolio_value: float,
                           confidence: float = 0.95) -> float:
    """
    Assumes returns are normally distributed.
    VaR = |z_α × σ × portfolio_value|
    """
    mu    = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z     = stats.norm.ppf(1 - confidence)
    return abs((mu + z * sigma) * portfolio_value)


def compute_cvar(returns: np.ndarray, portfolio_value: float, confidence: float = 0.95) -> float:
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    tail   = returns[returns <= cutoff]
    cvar_r = tail.mean() if len(tail) > 0 else cutoff
    return abs(cvar_r * portfolio_value)


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """Annualised Sharpe. Default risk-free = 6.5% (RBI repo rate)."""
    daily_rf = risk_free_rate / 252
    excess   = returns - daily_rf
    if returns.std() == 0:
        return 0.0
    return float((excess.mean() / returns.std()) * np.sqrt(252))


def compute_max_drawdown(portfolio_values: pd.Series) -> float:
    roll_max = portfolio_values.cummax()
    drawdown = (portfolio_values - roll_max) / (roll_max + 1e-9)
    return float(drawdown.min() * 100)


def compute_portfolio_returns(price_data_dict: dict, quantities: dict) -> pd.Series:
    portfolio_values = None
    for ticker, df in price_data_dict.items():
        if df.empty:
            continue
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        stock_value      = close * quantities.get(ticker, 0)
        portfolio_values = stock_value if portfolio_values is None \
                           else portfolio_values.add(stock_value, fill_value=0)
    if portfolio_values is None:
        return pd.Series(dtype=float)
    return portfolio_values.pct_change().dropna()


def compute_correlation_matrix(price_data_dict: dict) -> pd.DataFrame:
    close_df = {}
    for ticker, df in price_data_dict.items():
        if df.empty:
            continue
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close_df[ticker] = close
    return pd.DataFrame(close_df).pct_change().corr()


def full_risk_report(
    price_data_dict: dict,
    quantities: dict,
    mc_results: dict,
    portfolio_summary: dict,
) -> dict:
    port_returns = compute_portfolio_returns(price_data_dict, quantities)
    port_value   = portfolio_summary["current_total"]

    port_val_series = None
    for ticker, df in price_data_dict.items():
        if df.empty:
            continue
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        sv              = close * quantities.get(ticker, 0)
        port_val_series = sv if port_val_series is None \
                          else port_val_series.add(sv, fill_value=0)

    var_95       = compute_var(port_returns.values,  port_value, 0.95)
    var_99       = compute_var(port_returns.values,  port_value, 0.99)
    param_var_95 = compute_parametric_var(port_returns.values, port_value, 0.95)
    param_var_99 = compute_parametric_var(port_returns.values, port_value, 0.99)
    cvar_95      = compute_cvar(port_returns.values, port_value, 0.95)
    sharpe       = compute_sharpe_ratio(port_returns)
    max_dd       = compute_max_drawdown(port_val_series) if port_val_series is not None else 0.0
    corr_mat     = compute_correlation_matrix(price_data_dict)
    mc_var       = port_value - mc_results["percentile_5"]

    return {
        "var_95":            round(var_95,   2),
        "var_99":            round(var_99,   2),
        "param_var_95":      round(param_var_95, 2),
        "param_var_99":      round(param_var_99, 2),
        "cvar_95":           round(cvar_95,  2),
        "mc_var":            round(mc_var,   2),
        "sharpe":            round(sharpe,   4),
        "max_drawdown":      round(max_dd,   4),
        "portfolio_returns": port_returns.values,
        "portfolio_values":  port_val_series.values if port_val_series is not None else [],
        "corr_matrix":       corr_mat,
    }
