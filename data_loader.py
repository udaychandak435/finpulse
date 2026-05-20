# # import yfinance as yf
# # import pandas as pd
# # import numpy as np
# # from datetime import datetime, timedelta
# # import warnings
# # warnings.filterwarnings("ignore")

# # def fetch_stock_data(ticker: str, years: int = 3) -> pd.DataFrame:
# #     """
# #     Fetch OHLCV data from yfinance with fallback handling.
# #     Returns a clean DataFrame indexed by Date.
# #     """
# #     end_date = datetime.today()
# #     start_date = end_date - timedelta(days=years * 365)

# #     try:
# #         df = yf.download(
# #             ticker,
# #             start=start_date.strftime("%Y-%m-%d"),
# #             end=end_date.strftime("%Y-%m-%d"),
# #             progress=False,
# #             auto_adjust=True,
# #         )
# #         if df.empty:
# #             raise ValueError(f"No data returned for {ticker}")

# #         # Flatten MultiIndex columns if present
# #         if isinstance(df.columns, pd.MultiIndex):
# #             df.columns = df.columns.get_level_values(0)

# #         df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
# #         df.index = pd.to_datetime(df.index)
# #         df.sort_index(inplace=True)
# #         return df

# #     except Exception as e:
# #         print(f"[data_loader] yfinance failed for {ticker}: {e}. Attempting fallback...")
# #         return _fallback_fetch(ticker, start_date, end_date)


# # def _fallback_fetch(ticker: str, start_date, end_date) -> pd.DataFrame:
# #     """
# #     Fallback: attempt scraping from Google Finance via pandas_datareader.
# #     If unavailable, returns an empty DataFrame.
# #     """
# #     try:
# #         import pandas_datareader.data as web
# #         df = web.DataReader(ticker, "av-daily", start=start_date, end=end_date)
# #         df.rename(columns={
# #             "open": "Open", "high": "High",
# #             "low": "Low", "close": "Close", "volume": "Volume"
# #         }, inplace=True)
# #         df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
# #         df.index = pd.to_datetime(df.index)
# #         return df
# #     except Exception as e:
# #         print(f"[data_loader] Fallback also failed for {ticker}: {e}")
# #         return pd.DataFrame()


# # def fetch_fundamental_data(ticker: str) -> dict:
# #     """
# #     Fetch key fundamental ratios using yfinance Ticker info.
# #     Returns a dictionary of financial metrics.
# #     """
# #     try:
# #         info = yf.Ticker(ticker).info
# #         de_ratio   = info.get("debtToEquity", 0.0) or 0.0
# #         roe        = info.get("returnOnEquity", 0.0) or 0.0
# #         roa        = info.get("returnOnAssets", 0.0) or 0.0
# #         op_margin  = info.get("operatingMargins", 0.0) or 0.0
# #         rev_growth = info.get("revenueGrowth", 0.0) or 0.0
# #         beta       = info.get("beta", 1.0) or 1.0

# #         # Risk Adjustment Factor (RAF)
# #         denominator = (1 + roa + roe)
# #         raf = (1 + de_ratio) / denominator if denominator != 0 else 1.0

# #         return {
# #             "DE_Ratio":      de_ratio,
# #             "ROE":           roe,
# #             "ROA":           roa,
# #             "OperatingMargin": op_margin,
# #             "RevenueGrowth": rev_growth,
# #             "Beta":          beta,
# #             "RAF":           raf,
# #         }
# #     except Exception as e:
# #         print(f"[data_loader] Fundamental data failed for {ticker}: {e}")
# #         return {
# #             "DE_Ratio": 0.0, "ROE": 0.0, "ROA": 0.0,
# #             "OperatingMargin": 0.0, "RevenueGrowth": 0.0,
# #             "Beta": 1.0, "RAF": 1.0,
# #         }


# # def fetch_all_stocks_parallel(tickers: list, years: int = 3) -> dict:
# #     """
# #     Fetch OHLCV + fundamentals for all tickers in parallel using ThreadPoolExecutor.
# #     Returns dict: {ticker: {"price_data": df, "fundamentals": dict}}
# #     """
# #     from concurrent.futures import ThreadPoolExecutor

# #     results = {}

# #     def _fetch_one(ticker):
# #         price_data   = fetch_stock_data(ticker, years)
# #         fundamentals = fetch_fundamental_data(ticker)
# #         return ticker, price_data, fundamentals

# #     with ThreadPoolExecutor(max_workers=max(1, len(tickers))) as executor:
# #         futures = {executor.submit(_fetch_one, t): t for t in tickers}
# #         for future in futures:
# #             ticker, price_data, fundamentals = future.result()
# #             results[ticker] = {
# #                 "price_data":   price_data,
# #                 "fundamentals": fundamentals,
# #             }

# #     return results







# import yfinance as yf
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings("ignore")


# def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Flatten MultiIndex columns, deduplicate, and ensure all
#     OHLCV columns are 1D Series. Handles all yfinance version quirks.
#     """
#     if df.empty:
#         return df

#     # Flatten MultiIndex columns
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)

#     # Deduplicate columns (keep first occurrence)
#     df = df.loc[:, ~df.columns.duplicated()]

#     # Keep only required columns
#     required = ["Open", "High", "Low", "Close", "Volume"]
#     available = [c for c in required if c in df.columns]
#     df = df[available]

#     # Squeeze any column that is still a DataFrame into a Series
#     for col in df.columns:
#         if isinstance(df[col], pd.DataFrame):
#             df[col] = df[col].iloc[:, 0]

#     df = df.dropna()
#     df.index = pd.to_datetime(df.index)
#     df.sort_index(inplace=True)
#     return df


# def fetch_stock_data(ticker: str, years: int = 3) -> pd.DataFrame:
#     """
#     Fetch OHLCV data from yfinance with fallback handling.
#     Returns a clean DataFrame indexed by Date.
#     """
#     end_date   = datetime.today()
#     start_date = end_date - timedelta(days=years * 365)

#     try:
#         df = yf.download(
#             ticker,
#             start=start_date.strftime("%Y-%m-%d"),
#             end=end_date.strftime("%Y-%m-%d"),
#             progress=False,
#             auto_adjust=True,
#             actions=False,
#         )
#         if df.empty:
#             raise ValueError(f"No data returned for {ticker}")

#         df = _sanitize_df(df)

#         if df.empty:
#             raise ValueError(f"DataFrame empty after sanitization for {ticker}")

#         return df

#     except Exception as e:
#         print(f"[data_loader] yfinance failed for {ticker}: {e}. Attempting fallback...")
#         return _fallback_fetch(ticker, start_date, end_date)


# def _fallback_fetch(ticker: str, start_date, end_date) -> pd.DataFrame:
#     """
#     Fallback: attempt via pandas_datareader.
#     Returns empty DataFrame if unavailable.
#     """
#     try:
#         import pandas_datareader.data as web
#         df = web.DataReader(ticker, "av-daily", start=start_date, end=end_date)
#         df.rename(columns={
#             "open": "Open", "high": "High",
#             "low": "Low", "close": "Close", "volume": "Volume"
#         }, inplace=True)
#         df = _sanitize_df(df)
#         return df
#     except Exception as e:
#         print(f"[data_loader] Fallback also failed for {ticker}: {e}")
#         return pd.DataFrame()


# def fetch_fundamental_data(ticker: str) -> dict:
#     """
#     Fetch key fundamental ratios using yfinance Ticker info.
#     Returns a dictionary of financial metrics with safe defaults.
#     """
#     try:
#         info = yf.Ticker(ticker).info

#         de_ratio   = float(info.get("debtToEquity",    0.0) or 0.0)
#         roe        = float(info.get("returnOnEquity",  0.0) or 0.0)
#         roa        = float(info.get("returnOnAssets",  0.0) or 0.0)
#         op_margin  = float(info.get("operatingMargins",0.0) or 0.0)
#         rev_growth = float(info.get("revenueGrowth",   0.0) or 0.0)
#         beta       = float(info.get("beta",            1.0) or 1.0)

#         # Risk Adjustment Factor: RAF = (1 + D/E) / (1 + ROA + ROE)
#         denominator = 1 + roa + roe
#         raf = (1 + de_ratio) / denominator if denominator != 0 else 1.0

#         # Clamp RAF to a reasonable range to avoid extreme scaling
#         raf = float(np.clip(raf, 0.5, 5.0))

#         return {
#             "DE_Ratio":        de_ratio,
#             "ROE":             roe,
#             "ROA":             roa,
#             "OperatingMargin": op_margin,
#             "RevenueGrowth":   rev_growth,
#             "Beta":            beta,
#             "RAF":             raf,
#         }
#     except Exception as e:
#         print(f"[data_loader] Fundamental data failed for {ticker}: {e}")
#         return {
#             "DE_Ratio": 0.0, "ROE": 0.0, "ROA": 0.0,
#             "OperatingMargin": 0.0, "RevenueGrowth": 0.0,
#             "Beta": 1.0, "RAF": 1.0,
#         }


# def fetch_all_stocks_parallel(tickers: list, years: int = 3) -> dict:
#     """
#     Fetch OHLCV + fundamentals for all tickers in parallel
#     using ThreadPoolExecutor.
#     Returns: {ticker: {"price_data": df, "fundamentals": dict}}
#     """
#     from concurrent.futures import ThreadPoolExecutor

#     results = {}

#     def _fetch_one(ticker):
#         price_data   = fetch_stock_data(ticker, years)
#         fundamentals = fetch_fundamental_data(ticker)
#         return ticker, price_data, fundamentals

#     with ThreadPoolExecutor(max_workers=max(1, len(tickers))) as executor:
#         futures = {executor.submit(_fetch_one, t): t for t in tickers}
#         for future in futures:
#             ticker, price_data, fundamentals = future.result()
#             results[ticker] = {
#                 "price_data":   price_data,
#                 "fundamentals": fundamentals,
#             }

#     return results






import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns, deduplicate, and ensure all
    OHLCV columns are 1D Series. Handles all yfinance version quirks.
    """
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]

    required  = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in required if c in df.columns]
    df        = df[available]

    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def fetch_stock_data(ticker: str, years: int = 1) -> pd.DataFrame:
    """
    Fetch OHLCV data for a SINGLE ticker — call this sequentially only.
    Never call inside a thread to avoid yfinance race conditions.
    """
    from datetime import datetime, timedelta
    end_date = datetime.today() + timedelta(days=1)   # +1 so today's/Friday's data is included


    start_date = end_date - timedelta(days=years * 365)

    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            actions=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        df = _sanitize_df(df)

        if df.empty:
            raise ValueError(f"DataFrame empty after sanitization for {ticker}")

        return df

    except Exception as e:
        print(f"[data_loader] yfinance failed for {ticker}: {e}. Attempting fallback...")
        return _fallback_fetch(ticker, start_date, end_date)


def _fallback_fetch(ticker: str, start_date, end_date) -> pd.DataFrame:
    try:
        import pandas_datareader.data as web
        df = web.DataReader(ticker, "av-daily", start=start_date, end=end_date)
        df.rename(columns={
            "open": "Open", "high": "High",
            "low":  "Low",  "close": "Close", "volume": "Volume"
        }, inplace=True)
        df = _sanitize_df(df)
        return df
    except Exception as e:
        print(f"[data_loader] Fallback also failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_fundamental_data(ticker: str) -> dict:
    """
    Fetch fundamentals for a SINGLE ticker — call this sequentially only.
    Never call inside a thread.
    Pulls comprehensive KPI data from yfinance for financial health analysis.
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info

        de_ratio   = float(info.get("debtToEquity",     0.0) or 0.0)
        roe        = float(info.get("returnOnEquity",   0.0) or 0.0)
        roa        = float(info.get("returnOnAssets",   0.0) or 0.0)
        op_margin  = float(info.get("operatingMargins", 0.0) or 0.0)
        rev_growth = float(info.get("revenueGrowth",    0.0) or 0.0)
        beta       = float(info.get("beta",             1.0) or 1.0)

        denominator = 1 + roa + roe
        raf = (1 + de_ratio) / denominator if denominator != 0 else 1.0
        raf = float(np.clip(raf, 0.5, 5.0))

        # ── Extended KPI data ─────────────────────────────────────────
        # Income Statement KPIs
        total_revenue    = float(info.get("totalRevenue",     0.0) or 0.0)
        gross_profit     = float(info.get("grossProfits",     0.0) or 0.0)
        net_income       = float(info.get("netIncomeToCommon",0.0) or 0.0)
        ebitda           = float(info.get("ebitda",           0.0) or 0.0)
        gross_margin     = float(info.get("grossMargins",     0.0) or 0.0)
        profit_margin    = float(info.get("profitMargins",    0.0) or 0.0)
        earnings_growth  = float(info.get("earningsGrowth",   0.0) or 0.0)

        # Balance Sheet KPIs
        total_cash       = float(info.get("totalCash",           0.0) or 0.0)
        total_debt       = float(info.get("totalDebt",           0.0) or 0.0)
        current_ratio    = float(info.get("currentRatio",        0.0) or 0.0)
        quick_ratio      = float(info.get("quickRatio",          0.0) or 0.0)
        book_value       = float(info.get("bookValue",           0.0) or 0.0)
        total_assets     = float(info.get("totalAssets",         0.0) or 0.0)
        total_liabilities = total_debt  # approximation

        # Cash Flow KPIs
        operating_cf     = float(info.get("operatingCashflow",  0.0) or 0.0)
        free_cf          = float(info.get("freeCashflow",       0.0) or 0.0)

        # Hybrid / Valuation KPIs
        pe_ratio         = float(info.get("trailingPE",         0.0) or 0.0)
        forward_pe       = float(info.get("forwardPE",          0.0) or 0.0)
        peg_ratio        = float(info.get("pegRatio",           0.0) or 0.0)
        price_to_book    = float(info.get("priceToBook",        0.0) or 0.0)
        enterprise_value = float(info.get("enterpriseValue",    0.0) or 0.0)
        ev_to_ebitda     = float(info.get("enterpriseToEbitda", 0.0) or 0.0)
        dividend_yield   = float(info.get("dividendYield",      0.0) or 0.0)
        payout_ratio     = float(info.get("payoutRatio",        0.0) or 0.0)
        employees        = float(info.get("fullTimeEmployees",  0.0) or 0.0)

        # Derived metrics
        working_capital  = total_cash - total_debt if total_cash and total_debt else 0.0
        rev_per_employee = (total_revenue / employees) if employees > 0 else 0.0
        net_profit_margin = profit_margin
        ebitda_margin    = (ebitda / total_revenue * 100) if total_revenue > 0 else 0.0

        return {
            "DE_Ratio":        de_ratio,
            "ROE":             roe,
            "ROA":             roa,
            "OperatingMargin": op_margin,
            "RevenueGrowth":   rev_growth,
            "Beta":            beta,
            "RAF":             raf,
            # ── Extended KPIs ──
            "TotalRevenue":     total_revenue,
            "GrossProfit":      gross_profit,
            "NetIncome":        net_income,
            "EBITDA":           ebitda,
            "GrossMargin":      gross_margin,
            "NetProfitMargin":  net_profit_margin,
            "EBITDAMargin":     ebitda_margin,
            "EarningsGrowth":   earnings_growth,
            "CurrentRatio":     current_ratio,
            "QuickRatio":       quick_ratio,
            "TotalCash":        total_cash,
            "TotalDebt":        total_debt,
            "TotalAssets":      total_assets,
            "BookValue":        book_value,
            "WorkingCapital":   working_capital,
            "OperatingCF":      operating_cf,
            "FreeCF":           free_cf,
            "PE_Ratio":         pe_ratio,
            "ForwardPE":        forward_pe,
            "PEG_Ratio":        peg_ratio,
            "PriceToBook":      price_to_book,
            "EV":               enterprise_value,
            "EV_to_EBITDA":     ev_to_ebitda,
            "DividendYield":    dividend_yield,
            "PayoutRatio":      payout_ratio,
            "Employees":        employees,
            "RevPerEmployee":   rev_per_employee,
        }
    except Exception as e:
        print(f"[data_loader] Fundamental data failed for {ticker}: {e}")
        return {
            "DE_Ratio": 0.0, "ROE": 0.0, "ROA": 0.0,
            "OperatingMargin": 0.0, "RevenueGrowth": 0.0,
            "Beta": 1.0, "RAF": 1.0,
            "TotalRevenue": 0.0, "GrossProfit": 0.0, "NetIncome": 0.0,
            "EBITDA": 0.0, "GrossMargin": 0.0, "NetProfitMargin": 0.0,
            "EBITDAMargin": 0.0, "EarningsGrowth": 0.0,
            "CurrentRatio": 0.0, "QuickRatio": 0.0,
            "TotalCash": 0.0, "TotalDebt": 0.0, "TotalAssets": 0.0,
            "BookValue": 0.0, "WorkingCapital": 0.0,
            "OperatingCF": 0.0, "FreeCF": 0.0,
            "PE_Ratio": 0.0, "ForwardPE": 0.0, "PEG_Ratio": 0.0,
            "PriceToBook": 0.0, "EV": 0.0, "EV_to_EBITDA": 0.0,
            "DividendYield": 0.0, "PayoutRatio": 0.0,
            "Employees": 0.0, "RevPerEmployee": 0.0,
        }


def fetch_all_stocks_sequential(tickers: list, years: int = 3) -> dict:
    """
    ─────────────────────────────────────────────────────────────────────
    STEP 1 — Fetch ALL price data and fundamentals SEQUENTIALLY.

    WHY SEQUENTIAL?
    yfinance uses a shared internal HTTP session. Calling yf.download()
    from multiple threads simultaneously causes race conditions where
    all threads receive the same ticker's data — every stock shows
    the same price. Fetching sequentially first is the correct pattern
    (matches multitheread-2.py reference implementation).

    STEP 2 — After this function returns, pass the pre-fetched data_store
    into ThreadPoolExecutor workers for CPU-heavy ML processing.
    ─────────────────────────────────────────────────────────────────────

    Returns: {ticker: {"price_data": df, "fundamentals": dict}}
    """
    results = {}

    print("[data_loader] Fetching price data sequentially to prevent race conditions...")
    for ticker in tickers:
        print(f"[data_loader]  -> Downloading {ticker}...")
        price_data   = fetch_stock_data(ticker, 1)
        fundamentals = fetch_fundamental_data(ticker)
        results[ticker] = {
            "price_data":   price_data,
            "fundamentals": fundamentals,
        }
        print(f"[data_loader]  [OK] {ticker}: {len(price_data)} rows fetched")

    print(f"[data_loader] All {len(tickers)} stocks fetched. Ready for parallel ML processing.")
    return results
