# import numpy as np
# import pandas as pd
# from arch import arch_model
# import warnings
# warnings.filterwarnings("ignore")


# def compute_garch_volatility(returns: pd.Series, horizon: int = 1) -> dict:
#     """
#     Fit GARCH(1,1) on log-returns and forecast next-day conditional variance.

#     σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

#     Returns dict with:
#         - 'params':       fitted model parameters (omega, alpha, beta)
#         - 'forecast_var': next-day conditional variance
#         - 'forecast_vol': next-day conditional volatility (annualised)
#         - 'fitted_vol':   in-sample conditional volatility series
#         - 'aic':          model AIC
#     """
#     clean_returns = returns.dropna() * 100   # scale for numerical stability

#     model = arch_model(
#         clean_returns,
#         vol="Garch",
#         p=1, q=1,
#         dist="normal",
#         mean="Zero",       # use Zero mean for residual modeling
#     )

#     try:
#         result = model.fit(disp="off", show_warning=False)
#     except Exception:
#         result = model.fit(disp="off", options={"maxiter": 500})

#     forecast     = result.forecast(horizon=horizon, reindex=False)
#     forecast_var = float(forecast.variance.iloc[-1, 0]) / 1e4   # rescale
#     forecast_vol = np.sqrt(forecast_var * 252)                   # annualised

#     params = {
#         "omega": float(result.params.get("omega", 0)),
#         "alpha": float(result.params.get("alpha[1]", 0)),
#         "beta":  float(result.params.get("beta[1]", 0)),
#     }

#     fitted_vol = np.sqrt(result.conditional_volatility ** 2 / 1e4)

#     return {
#         "params":       params,
#         "forecast_var": forecast_var,
#         "forecast_vol": forecast_vol,
#         "fitted_vol":   fitted_vol,
#         "aic":          result.aic,
#         "result_obj":   result,
#     }


# def get_adjusted_volatility(garch_output: dict, raf: float) -> float:
#     """
#     σ_adjusted = σ_historical × RAF
#     Uses GARCH forecast volatility as σ_historical.
#     """
#     return garch_output["forecast_vol"] * raf






# import numpy as np
# import pandas as pd
# from arch import arch_model
# import warnings
# warnings.filterwarnings("ignore")


# def compute_garch_volatility(returns: pd.Series, horizon: int = 1) -> dict:
#     """
#     Fit GARCH(1,1) on log-returns and forecast next-day conditional variance.
#     σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

#     Returns dict with params, forecast variance/volatility, fitted vol, AIC.
#     """
#     if isinstance(returns, pd.DataFrame):
#         returns = returns.iloc[:, 0]

#     clean_returns = returns.dropna() * 100   # scale for numerical stability

#     model = arch_model(
#         clean_returns,
#         vol="Garch",
#         p=1, q=1,
#         dist="normal",
#         mean="Zero",
#     )

#     try:
#         result = model.fit(disp="off", show_warning=False)
#     except Exception:
#         result = model.fit(disp="off", options={"maxiter": 500})

#     forecast     = result.forecast(horizon=horizon, reindex=False)
#     forecast_var = float(forecast.variance.iloc[-1, 0]) / 1e4
#     forecast_vol = np.sqrt(forecast_var * 252)     # annualised

#     params = {
#         "omega": float(result.params.get("omega",    0)),
#         "alpha": float(result.params.get("alpha[1]", 0)),
#         "beta":  float(result.params.get("beta[1]",  0)),
#     }

#     fitted_vol = np.sqrt(result.conditional_volatility ** 2 / 1e4)

#     return {
#         "params":       params,
#         "forecast_var": forecast_var,
#         "forecast_vol": forecast_vol,
#         "fitted_vol":   fitted_vol,
#         "aic":          result.aic,
#         "result_obj":   result,
#     }


# def get_adjusted_volatility(garch_output: dict, raf: float) -> float:
#     """
#     σ_adjusted = σ_historical × RAF
#     Uses GARCH forecast volatility as σ_historical.
#     """
#     return garch_output["forecast_vol"] * float(raf)





import numpy as np
import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")


def compute_garch_volatility(returns: pd.Series, horizon: int = 1) -> dict:
    """
    Fit GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    Returns params, forecast variance/volatility, fitted vol, AIC.
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]

    clean_returns = returns.dropna() * 100  # scale for numerical stability

    model = arch_model(clean_returns, vol="Garch", p=1, q=1, dist="normal", mean="Zero")
    try:
        result = model.fit(disp="off", show_warning=False)
    except Exception:
        result = model.fit(disp="off", options={"maxiter": 500})

    forecast     = result.forecast(horizon=horizon, reindex=False)
    forecast_var = float(forecast.variance.iloc[-1, 0]) / 1e4
    forecast_vol = np.sqrt(forecast_var * 252)  # annualised

    params = {
        "omega": float(result.params.get("omega",    0)),
        "alpha": float(result.params.get("alpha[1]", 0)),
        "beta":  float(result.params.get("beta[1]",  0)),
    }
    fitted_vol = np.sqrt(result.conditional_volatility ** 2 / 1e4)

    return {
        "params":       params,
        "forecast_var": forecast_var,
        "forecast_vol": forecast_vol,
        "fitted_vol":   fitted_vol,
        "aic":          result.aic,
        "result_obj":   result,
    }


def get_adjusted_volatility(garch_output: dict, raf: float) -> float:
    """σ_adjusted = σ_GARCH_forecast × RAF"""
    return garch_output["forecast_vol"] * float(raf)