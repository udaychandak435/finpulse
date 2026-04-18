# hyperparameters.py
# ─────────────────────────────────────────────────────────────────────────────
# Per-Stock Hyperparameter Configuration
#
# HOW TO ADD A NEW STOCK:
#   1. Add a new entry to STOCK_HYPERPARAMS below using the ticker as key
#   2. Set all 8 parameters (see PARAMETER GUIDE at the bottom)
#   3. Save and re-run — no other file needs to be changed
#
# If a ticker is NOT listed here, DEFAULT_PARAMS is used automatically.
# ─────────────────────────────────────────────────────────────────────────────

STOCK_HYPERPARAMS = {

    "RELIANCE.NS": {
        "seq_len":        45,
        "epochs":         120,
        "batch_size":     32,
        "learning_rate":  0.0008,
        "dropout":        0.35,   # ↑ was 0.25 — more regularisation
        "lstm_units":     128,
        "gru_units":      128,
        "conv_filters":   64,
    },

    "TCS.NS": {
        "seq_len":        30,
        "epochs":         100,
        "batch_size":     32,
        "learning_rate":  0.001,
        "dropout":        0.30,   # ↑ was 0.20
        "lstm_units":     64,
        "gru_units":      64,
        "conv_filters":   64,
    },

    "INFY.NS": {
        "seq_len":        35,
        "epochs":         110,
        "batch_size":     16,
        "learning_rate":  0.0009,
        "dropout":        0.30,   # ↑ was 0.20
        "lstm_units":     96,
        "gru_units":      96,
        "conv_filters":   64,
    },

    "HDFCBANK.NS": {
        "seq_len":        40,
        "epochs":         100,
        "batch_size":     32,
        "learning_rate":  0.001,
        "dropout":        0.38,   # ↑ was 0.30
        "lstm_units":     64,
        "gru_units":      128,
        "conv_filters":   128,
    },

    "WIPRO.NS": {
        "seq_len":        30,
        "epochs":         90,
        "batch_size":     32,
        "learning_rate":  0.001,
        "dropout":        0.30,   # ↑ was 0.20
        "lstm_units":     64,
        "gru_units":      64,
        "conv_filters":   64,
    },

    "ICICIBANK.NS": {
        "seq_len":        40,
        "epochs":         110,
        "batch_size":     32,
        "learning_rate":  0.0008,
        "dropout":        0.35,   # ↑ was 0.25
        "lstm_units":     96,
        "gru_units":      96,
        "conv_filters":   64,
    },

    "BHARTIARTL.NS": {
        "seq_len":        35,
        "epochs":         100,
        "batch_size":     32,
        "learning_rate":  0.001,
        "dropout":        0.30,   # ↑ was 0.20
        "lstm_units":     64,
        "gru_units":      64,
        "conv_filters":   64,
    },

    "ITC.NS": {
        "seq_len":        30,
        "epochs":         90,
        "batch_size":     64,
        "learning_rate":  0.001,
        "dropout":        0.25,   # ↑ was 0.15
        "lstm_units":     64,
        "gru_units":      64,
        "conv_filters":   32,
    },

    "SBIN.NS": {
        "seq_len":        40,
        "epochs":         100,
        "batch_size":     32,
        "learning_rate":  0.001,
        "dropout":        0.35,   # ↑ was 0.25
        "lstm_units":     96,
        "gru_units":      96,
        "conv_filters":   64,
    },

    "ADANIENT.NS": {
        "seq_len":        50,
        "epochs":         130,
        "batch_size":     16,
        "learning_rate":  0.0006,
        "dropout":        0.40,   # ↑ was 0.35
        "lstm_units":     128,
        "gru_units":      128,
        "conv_filters":   128,
    },

    "HINDUNILVR.NS": {
        "seq_len":        30,
        "epochs":         90,
        "batch_size":     64,
        "learning_rate":  0.001,
        "dropout":        0.25,   # ↑ was 0.15
        "lstm_units":     64,
        "gru_units":      64,
        "conv_filters":   32,
    },

    "BAJFINANCE.NS": {
        "seq_len":        45,
        "epochs":         120,
        "batch_size":     32,
        "learning_rate":  0.0008,
        "dropout":        0.38,   # ↑ was 0.30
        "lstm_units":     128,
        "gru_units":      128,
        "conv_filters":   64,
    },

    # ── ADD NEW STOCKS BELOW THIS LINE ────────────────────────────────────────
    # Copy-paste this template and fill in values:
    #
    # "TICKER.NS": {
    #     "seq_len":       30,
    #     "epochs":        100,
    #     "batch_size":    32,
    #     "learning_rate": 0.001,
    #     "dropout":       0.20,
    #     "lstm_units":    64,
    #     "gru_units":     64,
    #     "conv_filters":  64,
    # },
}

# ─────────────────────────────────────────────────────────────────────────────
# Default fallback — used when ticker is NOT in STOCK_HYPERPARAMS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "seq_len":        30,
    "epochs":         100,
    "batch_size":     32,
    "learning_rate":  0.001,
    "dropout":        0.30,   # ↑ was 0.20 — more regularisation
    "lstm_units":     64,
    "gru_units":      64,
    "conv_filters":   64,
}


def get_hyperparams(ticker: str) -> dict:
    """
    Returns hyperparameters for the given ticker.
    Falls back to DEFAULT_PARAMS if ticker is not configured.
    """
    if ticker in STOCK_HYPERPARAMS:
        return STOCK_HYPERPARAMS[ticker].copy()
    print(f"[hyperparams] No config found for {ticker} — using DEFAULT_PARAMS")
    return DEFAULT_PARAMS.copy()


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER GUIDE
# ─────────────────────────────────────────────────────────────────────────────
# seq_len        : Lookback window in days. Higher = more context but slower.
#                  Recommended range: 20–60
#
# epochs         : Max training iterations. EarlyStopping stops early if
#                  val_loss plateaus. Recommended range: 80–150
#
# batch_size     : Samples per gradient update. Smaller = noisier gradients
#                  but can generalise better. Options: 16, 32, 64
#
# learning_rate  : Adam optimizer step size.
#                  Recommended range: 0.0005–0.002
#
# dropout        : Fraction of units randomly dropped after recurrent blocks.
#                  Higher for noisy/volatile stocks. Range: 0.10–0.40
#
# lstm_units     : Units in the LSTM layer. Options: 32, 64, 96, 128
#
# gru_units      : Units in each BiGRU layer (doubled by Bidirectional wrapper).
#                  Options: 32, 64, 96, 128
#
# conv_filters   : Filters in Conv1D layers (second layer uses 2×).
#                  Options: 32, 64, 128
# ─────────────────────────────────────────────────────────────────────────────
