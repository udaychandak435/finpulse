import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def _to_series(obj, name: str = "") -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        obj = pd.Series(obj)
    obj.name = name
    return obj


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    series = _to_series(series)
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series   = _to_series(series)
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9):
    series      = _to_series(series)
    ema_fast    = compute_ema(series, fast)
    ema_slow    = compute_ema(series, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    series = _to_series(series)
    return series - series.shift(period)


def compute_rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    series  = _to_series(series)
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window=window).std()


# ── NEW: Bollinger Bands ─────────────────────────────────────────────────────
def compute_bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0):
    """Returns (upper_band, middle_band, lower_band, bandwidth)."""
    series = _to_series(series)
    middle = series.rolling(window=window).mean()
    std    = series.rolling(window=window).std()
    upper  = middle + n_std * std
    lower  = middle - n_std * std
    bandwidth = (upper - lower) / (middle + 1e-9)
    return upper, middle, lower, bandwidth


# ── NEW: Average True Range (ATR) ────────────────────────────────────────────
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    """Average True Range — measures volatility using price range."""
    high  = _to_series(high)
    low   = _to_series(low)
    close = _to_series(close)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low  - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


# ── NEW: Stochastic Oscillator ────────────────────────────────────────────────
def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                       k_period: int = 14, d_period: int = 3):
    """Returns (%K, %D) — momentum oscillator comparing close to high-low range."""
    high  = _to_series(high)
    low   = _to_series(low)
    close = _to_series(close)
    low_n  = low.rolling(window=k_period).min()
    high_n = high.rolling(window=k_period).max()
    k_pct  = 100 * (close - low_n) / (high_n - low_n + 1e-9)
    d_pct  = k_pct.rolling(window=d_period).mean()
    return k_pct, d_pct


# ── NEW: On-Balance Volume (OBV) ─────────────────────────────────────────────
def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    close  = _to_series(close)
    volume = _to_series(volume)
    direction = np.sign(close.diff()).fillna(0)
    obv = (volume * direction).cumsum()
    return obv


def build_features(df: pd.DataFrame, raf: float = 1.0) -> pd.DataFrame:
    feat = df.copy()

    if isinstance(feat.columns, pd.MultiIndex):
        feat.columns = feat.columns.get_level_values(0)
    feat = feat.loc[:, ~feat.columns.duplicated()]

    close  = _to_series(feat["Close"],  "Close")
    high   = _to_series(feat["High"],   "High")
    low    = _to_series(feat["Low"],    "Low")
    open_  = _to_series(feat["Open"],   "Open")
    volume = _to_series(feat["Volume"], "Volume")

    feat["Close"]  = close
    feat["High"]   = high
    feat["Low"]    = low
    feat["Open"]   = open_
    feat["Volume"] = volume

    # EMA
    feat["EMA12"] = compute_ema(close, 12)
    feat["EMA26"] = compute_ema(close, 26)
    feat["EMA50"] = compute_ema(close, 50)

    # RSI
    feat["RSI14"] = compute_rsi(close, 14)

    # MACD
    feat["MACD"], feat["MACD_Signal"], feat["MACD_Hist"] = compute_macd(close)

    # Momentum
    feat["Momentum10"]   = compute_momentum(close, 10)

    # Log returns & volatility
    feat["LogReturn"]    = np.log(close / close.shift(1))
    feat["RollingVol20"] = compute_rolling_volatility(close, 20)
    feat["AdjVolatility"]= feat["RollingVol20"] * float(raf)

    # Price spread
    feat["HL_Spread"]    = high - low
    feat["OC_Spread"]    = open_ - close

    # ── NEW INDICATORS ────────────────────────────────────────────────────────
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower, bb_bw = compute_bollinger_bands(close)
    feat["BB_Upper"]     = bb_upper
    feat["BB_Lower"]     = bb_lower
    feat["BB_Bandwidth"] = bb_bw
    feat["BB_Position"]  = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ATR
    feat["ATR14"] = compute_atr(high, low, close, 14)

    # Stochastic Oscillator
    feat["Stoch_K"], feat["Stoch_D"] = compute_stochastic(high, low, close)

    # OBV
    feat["OBV"] = compute_obv(close, volume)
    # ──────────────────────────────────────────────────────────────────────────

    # Lag features
    for lag in [1, 2, 3]:
        feat[f"Close_lag{lag}"] = close.shift(lag)
        feat[f"Vol_lag{lag}"]   = volume.shift(lag)

    feat.dropna(inplace=True)

    feat.reset_index(drop=False, inplace=True)
    for candidate in ["Date", "index", "Datetime"]:
        if candidate in feat.columns:
            feat.set_index(candidate, inplace=True)
            break

    return feat


def create_sequences(df: pd.DataFrame, seq_len: int = 30):
    """
    TARGET: Log returns of OHLC (not raw prices).
    Reconstruction: P_t = P_{t-1} * exp(predicted_log_return)
    Returns: X, y_returns, scaler_X, scaler_y, feature_cols, raw_ohlc_array
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    ohlc_cols    = ["Open", "High", "Low", "Close"]
    feature_cols = [c for c in df.columns if c not in ohlc_cols + ["Volume"]]

    X_df = df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)

    y_df      = df[ohlc_cols].copy().apply(pd.to_numeric, errors="coerce")
    y_returns = np.log(y_df / y_df.shift(1)).fillna(0)

    X_raw = X_df.values.astype(np.float32)
    y_raw = y_returns.values.astype(np.float32)

    scaler_X = MinMaxScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i - seq_len : i])
        y_seq.append(y_scaled[i])

    raw_ohlc = y_df.values[seq_len:]

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        scaler_X,
        scaler_y,
        feature_cols,
        raw_ohlc,
    )
