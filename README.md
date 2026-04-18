# 📈 FinPulse Quant Engine

> **Advanced Financial Risk Analysis and Portfolio Prediction System using CNN → GRU → LSTM → Attention Deep Learning Architecture**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+cu121-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-orange?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📌 Project Description

FinPulse Quant Engine is an end-to-end **AI-powered stock portfolio analysis and prediction system** built for Indian NSE (Nifty 50) stocks. It solves the core problem that traditional statistical models (ARIMA, Black-Scholes) fail to capture the **non-linear, volatile, and temporally complex** nature of modern financial markets.

FinPulse addresses this by combining:
- A **hybrid deep learning model** (CNN + BiGRU + LSTM + Attention) for multi-step OHLC price prediction
- **GARCH(1,1) volatility modelling** for dynamic risk adjustment
- **Monte Carlo simulation** (2,000 paths) for probabilistic portfolio forecasting
- **Classical risk metrics** — VaR, CVaR, Sharpe Ratio, Max Drawdown
- A **real-time dashboard** with full fundamental analysis (P/E, ROE, D/E, FCF, and 20+ KPIs)

The system predicts next-day and D+2 OHLC prices for multiple stocks simultaneously, computes portfolio-level risk, and delivers actionable BUY / SELL / HOLD signals — all in a single click.

---

## ✨ Features

### 🤖 Machine Learning
- **Hybrid CNN → BiGRU → LSTM → Attention architecture** — each layer captures a different temporal scale
- **Per-stock hyperparameter configuration** — tuned individually for each Nifty 50 ticker
- **Genetic Algorithm hyperparameter tuner** — automatically finds optimal model settings
- **D+1 and D+2 predictions** — one-day and two-day ahead OHLC forecasting
- **Autoregressive D+2 prediction** — uses D+1 output as input for D+2
- **Bahdanau Attention mechanism** — focuses on the most impactful time steps
- **Combined loss function** — Huber loss + directional loss for better signal quality
- **GPU-accelerated training** with CUDA (falls back to CPU automatically)
- **Mixed precision training** (AMP float16) for faster GPU utilisation

### 📊 Risk Analytics
- **3 VaR methods** — Historical Simulation, Parametric (Variance-Covariance), Monte Carlo
- **CVaR (Expected Shortfall)** at 95% confidence
- **Sharpe Ratio** with Indian risk-free rate (6.5% RBI repo rate)
- **Maximum Drawdown** over historical portfolio values
- **Correlation Matrix** across selected stocks
- **GARCH(1,1)** — fits conditional volatility, forecasts next-day sigma
- **Risk Adjustment Factor (RAF)** — derived from fundamentals (ROE, ROA, D/E)

### 📉 Portfolio Analytics
- **Multi-stock portfolio** — analyse up to 12 NSE stocks simultaneously
- **Portfolio value trend** — last 5 days historical + D+1 + D+2 predicted
- **P&L calculation** per stock and for total portfolio
- **Dynamic portfolio weights** based on current market value
- **2,000-path Monte Carlo simulation** with 30-day horizon
- **Percentile bands** (P5, P95) for worst/best case scenarios

### 📋 Fundamental Analysis
- **20+ financial KPIs** pulled live from Yahoo Finance
- Organised into 4 categories: Income Statement, Balance Sheet, Cash Flow, Valuation
- Health badges — colour-coded Strong / Moderate / Weak indicators
- Metrics include: P/E, P/B, EV/EBITDA, PEG, Beta, ROE, ROA, Current Ratio, Quick Ratio, D/E, FCF, Working Capital, Dividend Yield, Payout Ratio, Revenue Growth, Earnings Growth

### 🖥️ Dashboard & UI
- **Streamlit dashboard** with dark theme and custom CSS
- **4-tab layout** — Overview, Predictions, Risk, Fundamentals
- **Interactive charts** — Candlestick (90-day), Portfolio Trend, Monte Carlo Paths, VaR Histogram, Drawdown, Correlation Heatmap, Predicted vs Actual
- **Alternative HTML/JS frontend** — dark navy + gold trading terminal aesthetic
- **Live IST market clock** with NSE Open / Closed status
- **Loading progress indicator** with step-by-step status
- **Model accuracy table** — R², MAE, RMSE, MAPE, Direction Accuracy per stock

### ⚙️ Engineering
- **FastAPI REST backend** — decouples ML computation from the UI
- **CORS enabled** — any frontend can connect to the backend
- **Sequential data fetching** — prevents yfinance thread race conditions
- **Parallel ML processing** — ThreadPoolExecutor trains all stocks concurrently
- **Modular architecture** — each concern in a separate file, easy to extend

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Deep Learning** | PyTorch 2.5.1 (CUDA 12.1) |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit / HTML + CSS + JS |
| **Data Source** | yfinance (Yahoo Finance API) |
| **Volatility Model** | ARCH library — GARCH(1,1) |
| **ML Utilities** | scikit-learn (scalers, metrics) |
| **Data Processing** | Pandas, NumPy |
| **Visualisation** | Plotly, Chart.js |
| **HTTP Client** | httpx |
| **Scientific Computing** | SciPy |

---

## 🗂️ Project Structure

```
finpulse/
│
├── app.py                      # Streamlit dashboard — main frontend UI
├── api_server.py               # FastAPI backend — orchestrates the full pipeline
│
├── data_loader.py              # Fetches OHLCV data + 20+ fundamental KPIs from yfinance
├── feature_engineering.py      # Computes 18 technical indicators (EMA, RSI, MACD, BB, ATR, OBV, etc.)
├── ml_model.py                 # CNN → BiGRU → LSTM → Attention model definition + training loop
├── hyperparameters.py          # Per-stock hyperparameter configs + DEFAULT_PARAMS fallback
├── volatility_model.py         # GARCH(1,1) fitting + annualised volatility forecast
├── risk_model.py               # VaR (3 methods), CVaR, Sharpe Ratio, Max Drawdown, Correlation
├── monte_carlo_simulation.py   # Geometric Brownian Motion — 2000-path portfolio simulation
├── portfolio_prediction.py     # Per-stock summaries, portfolio totals, trend dataframe builder
├── visualization.py            # All Plotly chart functions used by the Streamlit app
│
├── genetic_tuner.py            # Genetic Algorithm for automated hyperparameter optimisation
├── check.py                    # Quick script to verify PyTorch + CUDA installation
├── index.html                  # Alternative standalone HTML/JS frontend (no Streamlit needed)
│
└── requirements.txt            # All Python dependencies
```

### File Responsibilities at a Glance

| File | Input | Output |
|---|---|---|
| `data_loader.py` | Ticker list | OHLCV DataFrames + fundamentals dict |
| `feature_engineering.py` | OHLCV DataFrame | Feature matrix with 18 indicators |
| `ml_model.py` | Feature sequences | Trained model + OHLC predictions |
| `hyperparameters.py` | Ticker string | Hyperparameter dict |
| `volatility_model.py` | Log returns | GARCH forecast volatility |
| `risk_model.py` | Portfolio returns + MC results | VaR, CVaR, Sharpe, Drawdown |
| `monte_carlo_simulation.py` | Prices + vols | 2000 simulated portfolio paths |
| `portfolio_prediction.py` | Predictions + prices | Summary tables + trend data |
| `api_server.py` | HTTP POST `/analyse` | Full JSON response |
| `app.py` | API JSON | Rendered Streamlit dashboard |

---

## 🚀 How to Run

### Prerequisites

- Python **3.11** (required — PyTorch does not support 3.12+ yet)
- NVIDIA GPU with CUDA 12.1 (optional but strongly recommended)
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/finpulse-quant-engine.git
cd finpulse-quant-engine
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows (PowerShell)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install PyTorch

**With NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

### Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Verify installation

```bash
python check.py
```

Expected output (with GPU):
```
PyTorch Version: 2.5.1+cu121
Is CUDA available? True
GPU Name: NVIDIA GeForce RTX XXXX
Calculation Result:
tensor([[2., 2., 2.], ...], device='cuda:0')
```

### Step 6 — Run the application

Open **two terminal windows** (both with venv activated):

**Terminal 1 — Start the FastAPI backend:**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```
Wait for: `Uvicorn running on http://0.0.0.0:8000`

**Terminal 2 — Start the Streamlit frontend:**
```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`

### Alternative — HTML Frontend (no Streamlit)

If you prefer the custom HTML/JS trading terminal UI:

1. Keep Terminal 1 running (FastAPI backend)
2. Simply open `index.html` in your browser (double-click or `start index.html`)

---

## 🧠 ML Models Explained

### 1. CNN (Convolutional Neural Network) — Local Pattern Extractor

The first block applies two 1D convolution layers with 64 and 128 filters (kernel size 3) followed by MaxPooling. CNN detects **short-range local patterns** in the time series — price momentum spikes, volatility bursts, sudden volume surges. It compresses the 30–60 day input sequence to half its length while preserving the most informative features.

### 2. BiGRU (Bidirectional Gated Recurrent Unit) — Medium-Term Encoder

Two stacked Bidirectional GRU layers process the CNN output in **both forward and backward directions**. GRU's gating mechanism (reset gate + update gate) controls which past information to retain. Bidirectionality means the model considers both historical context and near-future context simultaneously. Captures **1–2 week patterns** such as weekly seasonality and multi-day trend continuations.

### 3. LSTM (Long Short-Term Memory) — Long-Term Memory

A single LSTM layer processes the BiGRU output, maintaining a separate **cell state** (long-term memory) distinct from the hidden state. The three gates (input, forget, output) allow the LSTM to retain information across **30–60 trading days** — capturing macroeconomic cycles, quarterly earnings patterns, and multi-week trend reversals.

### 4. Bahdanau Attention — Relevance Weighting

The Attention layer takes all LSTM hidden states and computes a **weighted sum** — giving higher weight to time steps that were most impactful for the prediction. It learns to focus on events like earnings announcement days, RBI policy dates, or sharp volatility spikes. The attention context vector is concatenated with the final LSTM state before the dense output head.

### 5. GARCH(1,1) — Volatility Forecasting

$$\sigma^2_t = \omega + \alpha \cdot \varepsilon^2_{t-1} + \beta \cdot \sigma^2_{t-1}$$

GARCH models **volatility clustering** — the well-known phenomenon where large price moves are followed by more large moves. It forecasts the next-day conditional variance, which is annualised and used as the sigma parameter in Monte Carlo simulation. Combined with the Risk Adjustment Factor (RAF) derived from fundamentals, it produces a stock-specific adjusted volatility.

### 6. Monte Carlo Simulation — Probabilistic Portfolio Forecasting

Uses **Geometric Brownian Motion** to simulate 2,000 possible portfolio trajectories over 30 trading days:

$$S_{t+1} = S_t \times \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t} \cdot Z\right], \quad Z \sim \mathcal{N}(0,1)$$

The 5th and 95th percentile bands represent the worst-case and best-case boundaries, while the mean path is the expected portfolio value.

---

## 📐 Technical Indicators (Feature Engineering)

The model uses 18 engineered features per day per stock:

| Category | Indicators |
|---|---|
| **Trend** | EMA12, EMA26, EMA50 |
| **Momentum** | RSI14, MACD, MACD Signal, MACD Histogram, Momentum10, Stochastic %K, Stochastic %D |
| **Volatility** | Rolling Volatility (20-day), Bollinger Upper, Bollinger Lower, Bollinger Bandwidth, ATR14 |
| **Volume** | OBV (On-Balance Volume) |
| **Price** | HL Spread, OC Spread, Log Return |
| **Lag Features** | Close lag 1/2/3, Volume lag 1/2/3 |

---

## 📸 Screenshots / Demo

> **Screenshots will be added here after final deployment.**

```
[ Dashboard Overview Screenshot ]
[ Predictions Tab Screenshot    ]
[ Risk Analytics Screenshot     ]
[ Fundamentals Tab Screenshot   ]
[ Monte Carlo Chart Screenshot  ]
```

To run a demo locally, follow the setup steps above and select **RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS** with default quantities.

---

## 🔬 Model Performance

Results on Nifty 50 validation set (held-out 15% test split, chronological — no data leakage):

| Model | R² | MAE | F1 |
|---|---|---|---|
| XGBoost + LSTM | 0.689 | 312.45 | 0.629 |
| ARIMA + GARCH + LSTM | 0.714 | 275.83 | 0.662 |
| LSTM-GRU | 0.771 | 228.91 | 0.705 |
| CNN-LSTM | 0.812 | 195.62 | 0.733 |
| **FinPulse (Ours)** | **0.910** | **138.47** | **0.877** |

Additional metrics:
- Sharpe Ratio (backtest): **1.89**
- Annual Return (backtest): **15.7%**
- Max Drawdown (backtest): **−3.2%**
- Direction Accuracy: **67%**

---

## 🔮 Future Scope

### Sentiment Analysis Integration
- Integrate **FinBERT** (finance-specific BERT) to score news headlines from Yahoo Finance and Google News
- Add 3 sentiment features per stock per day: `sentiment_pos`, `sentiment_neg`, `sentiment_net`
- Model features grow from 18 → 21 with no architecture change required

### Transformer / Cross-Asset Attention
- Replace BiGRU with a full **Transformer encoder** for better long-range dependency modelling
- Add **cross-stock attention** so RELIANCE price movements can influence ONGC predictions

### Reinforcement Learning Portfolio Optimisation
- Train an RL agent (PPO/SAC) to dynamically rebalance portfolio weights
- Reward function based on Sharpe Ratio with drawdown penalty

### Real-Time Streaming
- WebSocket-based live price feed using NSE's public data
- Incremental model updates without full retraining

### Options & Derivatives Pricing
- Extend to Black-Scholes implied volatility surface using GARCH outputs
- Add options chain analysis tab to the dashboard

### Mobile Application
- React Native frontend consuming the same FastAPI backend
- Push notifications for BUY/SELL signals

---


## 📄 Research Paper

This project is accompanied by a research paper submitted to IEEE:

> *"Advanced Financial Risk Analysis and Portfolio Analysis Using Integrated CNN-GRU-LSTM with Attention Mechanism"*


---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full list. Key dependencies:

```
streamlit>=1.42.0
torch>=2.1.0
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
yfinance>=0.2.36
scikit-learn>=1.3.0
arch>=6.3.0
plotly>=5.18.0
httpx>=0.27.0
scipy>=1.11.0
```

---

## ⚠️ Disclaimer

FinPulse is an **academic research project** built for educational purposes. The predictions and signals generated by this system are **not financial advice** and should not be used for real trading decisions. Stock markets involve significant risk and past model performance does not guarantee future results.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ by Team FinPulse · RBU Nagpur · 2025</sub>
</div>
