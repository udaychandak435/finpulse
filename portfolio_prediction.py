# import numpy as np
# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")


# def compute_stock_prediction_summary(
#     ticker, current_price, predicted_ohlc, quantity, garch_vol, fundamentals
# ):
#     pred_close      = float(predicted_ohlc[3])
#     pct_change      = ((pred_close - current_price) / current_price) * 100
#     current_value   = current_price * quantity
#     predicted_value = pred_close * quantity
#     return {
#         "Ticker":           ticker,
#         "Current Price":    round(current_price, 2),
#         "Pred Open":        round(float(predicted_ohlc[0]), 2),
#         "Pred High":        round(float(predicted_ohlc[1]), 2),
#         "Pred Low":         round(float(predicted_ohlc[2]), 2),
#         "Predicted Close":  round(pred_close, 2),
#         "% Change":         round(pct_change, 4),
#         "Quantity":         quantity,
#         "Current Value":    round(current_value, 2),
#         "Predicted Value":  round(predicted_value, 2),
#         "GARCH Vol (%)":    round(garch_vol * 100, 4),
#         "Beta":             fundamentals.get("Beta", 1.0),
#         "RAF":              round(fundamentals.get("RAF", 1.0), 4),
#         "ROE":              fundamentals.get("ROE", 0.0),
#         "ROA":              fundamentals.get("ROA", 0.0),
#         "DE_Ratio":         fundamentals.get("DE_Ratio", 0.0),
#     }


# def compute_portfolio_summary(stock_summaries):
#     current_total   = sum(s["Current Value"]   for s in stock_summaries)
#     predicted_total = sum(s["Predicted Value"] for s in stock_summaries)
#     pct_change = ((predicted_total - current_total) / current_total) * 100 \
#                   if current_total > 0 else 0.0
#     weights = {
#         s["Ticker"]: round((s["Current Value"] / current_total) * 100, 2)
#         for s in stock_summaries
#     } if current_total > 0 else {}
#     return {
#         "current_total":   round(current_total, 2),
#         "predicted_total": round(predicted_total, 2),
#         "pct_change":      round(pct_change, 4),
#         "weights":         weights,
#     }


# def build_portfolio_trend(price_data_dict, quantities, predicted_closes, last_n_days=5):
#     records = []
#     sample_df = next((df for df in price_data_dict.values() if not df.empty), None)
#     if sample_df is None:
#         return pd.DataFrame()

#     recent_dates = sample_df.index[-last_n_days:]
#     for date in recent_dates:
#         pv = sum(
#             float(price_data_dict[t].loc[date, "Close"]) * quantities.get(t, 0)
#             for t in price_data_dict
#             if not price_data_dict[t].empty and date in price_data_dict[t].index
#         )
#         records.append({"Date": date, "Portfolio Value": pv, "Type": "Historical"})

#     next_date = recent_dates[-1] + pd.tseries.offsets.BDay(1)
#     pred_pv   = sum(predicted_closes.get(t, 0) * quantities.get(t, 0)
#                     for t in predicted_closes)
#     records.append({"Date": next_date, "Portfolio Value": pred_pv, "Type": "Predicted"})

#     return pd.DataFrame(records).set_index("Date")







# import numpy as np
# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")


# def compute_stock_prediction_summary(
#     ticker, current_price, predicted_ohlc,
#     quantity, garch_vol, fundamentals
# ) -> dict:
#     pred_close      = float(predicted_ohlc[3])
#     pct_change      = ((pred_close - current_price) / current_price) * 100
#     current_value   = current_price * quantity
#     predicted_value = pred_close * quantity

#     return {
#         "Ticker":           ticker,
#         "Current Price":    round(current_price, 2),
#         "Pred Open":        round(float(predicted_ohlc[0]), 2),
#         "Pred High":        round(float(predicted_ohlc[1]), 2),
#         "Pred Low":         round(float(predicted_ohlc[2]), 2),
#         "Predicted Close":  round(pred_close, 2),
#         "% Change":         round(pct_change, 4),
#         "Quantity":         quantity,
#         "Current Value":    round(current_value, 2),
#         "Predicted Value":  round(predicted_value, 2),
#         "GARCH Vol (%)":    round(garch_vol * 100, 4),
#         "Beta":             fundamentals.get("Beta", 1.0),
#         "RAF":              round(fundamentals.get("RAF", 1.0), 4),
#         "ROE":              fundamentals.get("ROE", 0.0),
#         "ROA":              fundamentals.get("ROA", 0.0),
#         "DE_Ratio":         fundamentals.get("DE_Ratio", 0.0),
#     }


# def compute_portfolio_summary(stock_summaries: list) -> dict:
#     current_total   = sum(s["Current Value"]   for s in stock_summaries)
#     predicted_total = sum(s["Predicted Value"] for s in stock_summaries)
#     pct_change = ((predicted_total - current_total) / current_total) * 100 \
#                   if current_total > 0 else 0.0
#     weights = {
#         s["Ticker"]: round((s["Current Value"] / current_total) * 100, 2)
#         for s in stock_summaries
#     } if current_total > 0 else {}

#     return {
#         "current_total":   round(current_total, 2),
#         "predicted_total": round(predicted_total, 2),
#         "pct_change":      round(pct_change, 4),
#         "weights":         weights,
#     }


# def build_portfolio_trend(
#     price_data_dict:  dict,
#     quantities:       dict,
#     predicted_closes: dict,
#     last_n_days:      int = 5,
# ) -> pd.DataFrame:
#     """
#     Build a DataFrame: last N days historical portfolio value + 1 predicted day.
#     """
#     sample_df = next((df for df in price_data_dict.values() if not df.empty), None)
#     if sample_df is None:
#         return pd.DataFrame()

#     recent_dates = sample_df.index[-last_n_days:]
#     records = []

#     for date in recent_dates:
#         pv = 0.0
#         for t, df in price_data_dict.items():
#             if not df.empty and date in df.index:
#                 close_val = df.loc[date, "Close"]
#                 if isinstance(close_val, pd.Series):
#                     close_val = close_val.iloc[0]
#                 pv += float(close_val) * quantities.get(t, 0)
#         records.append({"Date": date, "Portfolio Value": pv, "Type": "Historical"})

#     next_date = recent_dates[-1] + pd.tseries.offsets.BDay(1)
#     pred_pv   = sum(
#         predicted_closes.get(t, 0) * quantities.get(t, 0)
#         for t in predicted_closes
#     )
#     records.append({"Date": next_date, "Portfolio Value": pred_pv, "Type": "Predicted"})

#     return pd.DataFrame(records).set_index("Date")




# =============================================================================
# portfolio_prediction.py
# =============================================================================
import numpy as np
import pandas as pd


# =============================================================================
# PER-STOCK SUMMARY
# =============================================================================
def compute_stock_prediction_summary(
    ticker        : str,
    current_price : float,
    predicted_ohlc: np.ndarray,
    quantity      : int,
    garch_vol     : float,
    fundamentals  : dict,
) -> dict:
    pred_close   = float(predicted_ohlc[3])
    pred_open    = float(predicted_ohlc[0])
    pred_high    = float(predicted_ohlc[1])
    pred_low     = float(predicted_ohlc[2])
    pct_change   = ((pred_close - current_price) / current_price) * 100
    current_val  = current_price * quantity
    predicted_val= pred_close   * quantity
    pnl          = predicted_val - current_val
    signal       = "BUY" if pct_change > 0.5 else ("SELL" if pct_change < -0.5 else "HOLD")

    return {
        "Ticker"          : ticker,
        "Current Price"   : round(current_price,  2),
        "Pred Open"       : round(pred_open,       2),
        "Pred High"       : round(pred_high,       2),
        "Pred Low"        : round(pred_low,        2),
        "Pred Close"      : round(pred_close,      2),
        "Change (%)"      : round(pct_change,      3),
        "Signal"          : signal,
        "Quantity"        : quantity,
        "Current Value"   : round(current_val,     2),
        "Predicted Value" : round(predicted_val,   2),
        "P&L"             : round(pnl,             2),
        "GARCH Vol (%)"   : round(garch_vol * 100, 3),
        "Beta"            : fundamentals.get("Beta",     1.0),
        "PE Ratio"        : fundamentals.get("PE_Ratio", None),
        "DE Ratio"        : fundamentals.get("DE_Ratio", None),
        "RAF"             : round(fundamentals.get("RAF", 1.0), 4),
    }


# =============================================================================
# PORTFOLIO SUMMARY
# =============================================================================
def compute_portfolio_summary(stock_summaries: list) -> dict:
    current_total   = sum(s["Current Value"]   for s in stock_summaries)
    predicted_total = sum(s["Predicted Value"] for s in stock_summaries)
    total_pnl       = predicted_total - current_total
    pct_change      = ((predicted_total - current_total) / current_total * 100
                       if current_total > 0 else 0.0)

    weights = {
        s["Ticker"]: round((s["Current Value"] / current_total) * 100, 2)
        if current_total > 0 else 0.0
        for s in stock_summaries
    }

    return {
        "current_total"  : round(current_total,   2),
        "predicted_total": round(predicted_total, 2),
        "total_pnl"      : round(total_pnl,       2),
        "pct_change"     : round(pct_change,       3),
        "weights"        : weights,
        "n_stocks"       : len(stock_summaries),
    }


# =============================================================================
# PORTFOLIO TREND  (last N days + D+1 + D+2)
# =============================================================================
def build_portfolio_trend(
    price_data_dict      : dict,
    quantities           : dict,
    predicted_closes_d1  : dict,
    predicted_closes_d2  : dict = None,
    last_n_days          : int  = 5,
) -> pd.DataFrame:
    tickers = list(price_data_dict.keys())

    # Historical portfolio value
    close_frames = []
    for t in tickers:
        col = price_data_dict[t]["Close"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        close_frames.append(col.rename(t))

    close_df = pd.concat(close_frames, axis=1).dropna().tail(last_n_days)

    portfolio_vals = []
    for date, row in close_df.iterrows():
        val = sum(float(row[t]) * quantities[t] for t in tickers if t in row.index)
        portfolio_vals.append({
            "Date":            date,
            "Portfolio Value": round(val, 2),
            "Type":            "Historical",
        })

    trend_df = pd.DataFrame(portfolio_vals)

    # Business day offsets
    try:
        from pandas.tseries.offsets import BDay
        last_date = close_df.index[-1]
        d1_date   = last_date + BDay(1)
        d2_date   = last_date + BDay(2)
    except Exception:
        last_date = close_df.index[-1]
        d1_date   = last_date + pd.Timedelta(days=1)
        d2_date   = last_date + pd.Timedelta(days=2)

    # D+1 row
    d1_val = sum(predicted_closes_d1[t] * quantities[t] for t in tickers)
    trend_df = pd.concat([
        trend_df,
        pd.DataFrame([{
            "Date":            d1_date,
            "Portfolio Value": round(d1_val, 2),
            "Type":            "Predicted",
        }]),
    ], ignore_index=True)

    # D+2 row
    if predicted_closes_d2 is not None:
        d2_val = sum(predicted_closes_d2[t] * quantities[t] for t in tickers)
        trend_df = pd.concat([
            trend_df,
            pd.DataFrame([{
                "Date":            d2_date,
                "Portfolio Value": round(d2_val, 2),
                "Type":            "Predicted",
            }]),
        ], ignore_index=True)

    trend_df["Date"] = pd.to_datetime(trend_df["Date"])
    return trend_df.sort_values("Date").reset_index(drop=True)