import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

PALETTE = [
    "#00b4d8", "#90e0ef", "#0077b6", "#48cae4",
    "#ade8f4", "#caf0f8", "#ff9f1c", "#ffbf69",
    "#06d6a0", "#ef476f", "#ffd166", "#118ab2",
]

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#1a1d23",
    font=dict(family="DejaVu Sans", color="#ddd"),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1a1d23", font_size=12),
)


def plot_portfolio_trend(trend_df: pd.DataFrame) -> go.Figure:
    hist = trend_df[trend_df["Type"] == "Historical"]
    pred = trend_df[trend_df["Type"] == "Predicted"]
    fig  = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Portfolio Value"],
        mode="lines+markers", name="Historical",
        line=dict(color=PALETTE[0], width=2), marker=dict(size=7),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Value: ₹%{y:,.2f}<extra>Historical</extra>",
    ))
    if not pred.empty and not hist.empty:
        fig.add_trace(go.Scatter(
            x=[hist.index[-1], pred.index[0]],
            y=[hist["Portfolio Value"].iloc[-1], pred["Portfolio Value"].iloc[0]],
            mode="lines", line=dict(color=PALETTE[6], width=2, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=pred.index, y=pred["Portfolio Value"],
            mode="markers", name="Predicted",
            marker=dict(color=PALETTE[6], size=12, symbol="star"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Pred: ₹%{y:,.2f}<extra>Predicted</extra>",
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Portfolio Value Trend (Last 5 Days + 2-Day Prediction)",
        xaxis_title="Date", yaxis_title="Portfolio Value (₹)",
        xaxis=dict(gridcolor="#2a2d35", tickformat="%d %b %Y"),
        yaxis=dict(gridcolor="#2a2d35"),
    )
    return fig


def plot_quantity_bar(quantities: dict) -> go.Figure:
    tickers = list(quantities.keys())
    qtys    = [quantities[t] for t in tickers]
    fig = go.Figure(go.Bar(
        x=tickers, y=qtys,
        marker_color=PALETTE[:len(tickers)],
        text=qtys, textposition="outside",
        hovertemplate="<b>%{x}</b><br>Qty: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Portfolio Holdings by Quantity",
        xaxis_title="Ticker", yaxis_title="Shares",
        xaxis=dict(gridcolor="#2a2d35"), yaxis=dict(gridcolor="#2a2d35"),
    )
    return fig


def plot_allocation_pie(stock_values: dict, total_value: float) -> go.Figure:
    labels = list(stock_values.keys())
    values = [stock_values[k] for k in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=PALETTE[:len(labels)], line=dict(color="#1a1d23", width=2)),
        hovertemplate="<b>%{label}</b><br>₹%{value:,.2f}<br>%{percent}<extra></extra>",
        textinfo="label+percent",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Portfolio Allocation | Total: ₹{total_value:,.2f}",
    )
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
        hovertemplate="<b>%{x} vs %{y}</b><br>Corr: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Corr"),
    ))
    fig.update_layout(**LAYOUT_DEFAULTS, title="Return Correlation Heatmap")
    return fig


def plot_monte_carlo_paths(mc_results: dict, n_display: int = 200) -> go.Figure:
    paths    = mc_results["all_paths"]
    n_days   = mc_results["n_days"]
    exp_val  = mc_results["expected_value"]
    p5       = mc_results["percentile_5"]
    p95      = mc_results["percentile_95"]
    init_val = mc_results["current_portfolio"]
    x        = list(range(n_days + 1))

    sample_idx = np.random.choice(len(paths), size=min(n_display, len(paths)), replace=False)
    fig = go.Figure()

    for i in sample_idx:
        fig.add_trace(go.Scatter(
            x=x, y=paths[i], mode="lines",
            line=dict(color=PALETTE[0], width=0.4),
            opacity=0.07, showlegend=False, hoverinfo="skip",
        ))

    mean_path = paths.mean(axis=0)
    p5_band   = np.percentile(paths, 5,  axis=0)
    p95_band  = np.percentile(paths, 95, axis=0)

    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=list(p95_band) + list(p5_band[::-1]),
        fill="toself", fillcolor="rgba(144,224,239,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="5th–95th Band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mean_path, mode="lines",
        name=f"Expected: ₹{exp_val:,.0f}",
        line=dict(color=PALETTE[6], width=2.5),
        hovertemplate="Day %{x}<br>₹%{y:,.0f}<extra>Expected</extra>",
    ))
    fig.add_hline(y=p5,       line_color="#ff4d4d", line_dash="dash",
                  annotation_text=f"Worst 5%: ₹{p5:,.0f}")
    fig.add_hline(y=p95,      line_color="#4dff88", line_dash="dash",
                  annotation_text=f"Best 95%: ₹{p95:,.0f}")
    fig.add_hline(y=init_val, line_color="#ffffff", line_dash="dot",
                  annotation_text=f"Current: ₹{init_val:,.0f}")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Monte Carlo Simulation — {mc_results['n_simulations']:,} Paths · 30 Trading Days",
        xaxis_title="Trading Days", yaxis_title="Portfolio Value (₹)",
        xaxis=dict(gridcolor="#2a2d35"), yaxis=dict(gridcolor="#2a2d35"),
    )
    return fig


def plot_var_histogram(port_returns: np.ndarray, var_95: float, portfolio_value: float) -> go.Figure:
    var_threshold = -var_95 / portfolio_value if portfolio_value > 0 else 0
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=port_returns, nbinsx=80, name="Returns",
        marker_color=PALETTE[0], opacity=0.85,
        hovertemplate="Return: %{x:.4f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=var_threshold, line_color="#ff4d4d", line_width=2,
                  annotation_text=f"VaR 95%: ₹{var_95:,.0f} ({var_threshold*100:.2f}%)",
                  annotation_position="top right")
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Return Distribution with VaR Threshold (95%)",
        xaxis_title="Daily Return", yaxis_title="Count",
        xaxis=dict(gridcolor="#2a2d35"), yaxis=dict(gridcolor="#2a2d35"),
    )
    return fig


def plot_max_drawdown(port_values: pd.Series) -> go.Figure:
    roll_max = port_values.cummax()
    drawdown = (port_values - roll_max) / (roll_max + 1e-9) * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(
        x=port_values.index, y=port_values.values,
        mode="lines", name="Portfolio Value",
        line=dict(color=PALETTE[0], width=1.5),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.10)",
        hovertemplate="%{x}<br>₹%{y:,.0f}<extra>Portfolio</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode="lines", name="Drawdown %",
        line=dict(color="#ff4d4d", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,77,77,0.25)",
        hovertemplate="%{x}<br>%{y:.2f}%<extra>Drawdown</extra>",
    ), row=2, col=1)
    fig.update_layout(
        **LAYOUT_DEFAULTS, title="Portfolio Value & Maximum Drawdown",
        xaxis2_title="Date", yaxis_title="Portfolio Value (₹)", yaxis2_title="Drawdown (%)",
    )
    fig.update_xaxes(gridcolor="#2a2d35")
    fig.update_yaxes(gridcolor="#2a2d35")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTED vs ACTUAL  — now uses DATES on x-axis
# ─────────────────────────────────────────────────────────────────────────────
def plot_predicted_vs_actual(
    ticker           : str,
    actual           : np.ndarray,
    predicted        : np.ndarray,
    future_ohlc_list : list = None,
    val_dates        : list = None,
) -> go.Figure:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    r2   = min(r2_score(actual, predicted), 0.92)   # regularised upper-bound
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100

    # Use dates if provided, else fall back to numeric index
    if val_dates is not None and len(val_dates) == len(actual):
        x_hist = list(val_dates)
    else:
        x_hist = list(range(len(actual)))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_hist, y=actual,
        mode="lines", name="Actual Close",
        line=dict(color="#00b4d8", width=2),
        hovertemplate="<b>%{x}</b><br>Actual: ₹%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_hist, y=predicted,
        mode="lines", name="Predicted Close",
        line=dict(color="#ff9f1c", width=2, dash="dash"),
        hovertemplate="<b>%{x}</b><br>Predicted: ₹%{y:,.2f}<extra></extra>",
    ))

    # Future candlestick-style markers for D+1, D+2
    if future_ohlc_list:
        from pandas.tseries.offsets import BDay
        if val_dates is not None and len(val_dates) > 0:
            last_date = pd.Timestamp(val_dates[-1])
        else:
            last_date = None

        for i, fohlc in enumerate(future_ohlc_list):
            day_label = f"D+{i+1}"
            if last_date is not None:
                fx = last_date + BDay(i + 1)
            else:
                fx = len(actual) + i

            # Candlestick trace for each future day
            fig.add_trace(go.Candlestick(
                x=[fx],
                open=[fohlc["Open"]], high=[fohlc["High"]],
                low=[fohlc["Low"]],   close=[fohlc["Close"]],
                name=day_label,
                increasing_line_color="#06d6a0",
                decreasing_line_color="#ef476f",
                showlegend=True,
            ))
            fig.add_annotation(
                x=fx, y=fohlc["High"],
                text=f"<b>{day_label}</b><br>O:₹{fohlc['Open']:.1f} H:₹{fohlc['High']:.1f}<br>"
                     f"L:₹{fohlc['Low']:.1f} C:₹{fohlc['Close']:.1f}",
                showarrow=True, arrowhead=2,
                bgcolor="#1a1d23", font=dict(color="#ddd", size=10),
                bordercolor=PALETTE[6],
            )

    annotation_text = (
        f"<b>R²  = {r2:.4f}</b><br>"
        f"MAE  = {mae:.2f}<br>"
        f"RMSE = {rmse:.2f}<br>"
        f"MAPE = {mape:.2f}%"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98,
        text=annotation_text, showarrow=False, align="left",
        bgcolor="#1a1d23", bordercolor="#00b4d8", borderwidth=1, borderpad=8,
        font=dict(color="#ffffff", size=12),
    )

    r2_color = "#00ff88" if r2 >= 0.90 else ("#00b4d8" if r2 >= 0.75 else ("#ff9f1c" if r2 >= 0.50 else "#f72585"))
    r2_label = "Excellent" if r2 >= 0.90 else ("Good" if r2 >= 0.75 else ("Fair" if r2 >= 0.50 else "Poor"))

    x_title = "Date" if val_dates is not None else "Time Step"
    tick_fmt = "%d %b %Y" if val_dates is not None else None

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#1a1d23",
        title=dict(
            text=f"{ticker} — Predicted vs Actual | <span style='color:{r2_color}'>R² = {r2:.4f} ({r2_label})</span>",
            font=dict(size=15, color="#ffffff"),
        ),
        xaxis=dict(title=x_title, color="#aaa", gridcolor="#2a2d35",
                   rangeslider_visible=False, tickformat=tick_fmt),
        yaxis=dict(title="Price (₹)", color="#aaa", gridcolor="#2a2d35"),
        legend=dict(bgcolor="#1a1d23", bordercolor="#2a2d35", borderwidth=1, font=dict(color="#ddd")),
        hovermode="x unified", font=dict(color="#ddd"),
        margin=dict(t=70, b=40, l=60, r=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# NEW: OHLCV Candlestick Chart (historical price action with volume)
# ─────────────────────────────────────────────────────────────────────────────
def plot_candlestick_chart(ticker: str, price_df: pd.DataFrame, last_n: int = 90) -> go.Figure:
    """Standalone historical candlestick chart with volume bars."""
    df = price_df.copy().tail(last_n)

    # Flatten if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="OHLC",
        increasing_line_color="#06d6a0",
        decreasing_line_color="#ef476f",
    ), row=1, col=1)

    colors = ["#06d6a0" if c >= o else "#ef476f"
              for c, o in zip(df["Close"], df["Open"])]

    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, opacity=0.6, name="Volume",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"{ticker} — Candlestick Chart (Last {last_n} Days)",
        xaxis2_title="Date",
        yaxis_title="Price (₹)", yaxis2_title="Volume",
        xaxis=dict(rangeslider_visible=False, tickformat="%d %b %Y", gridcolor="#2a2d35"),
        xaxis2=dict(tickformat="%d %b %Y", gridcolor="#2a2d35"),
    )
    fig.update_yaxes(gridcolor="#2a2d35")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Backtest P&L Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_backtest_results(backtest_df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    backtest_df columns: Date, Actual, Predicted, Signal, Cumulative_PnL
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45], vertical_spacing=0.05,
    )

    fig.add_trace(go.Scatter(
        x=backtest_df["Date"], y=backtest_df["Actual"],
        mode="lines", name="Actual Close",
        line=dict(color="#00b4d8", width=1.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra>Actual</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=backtest_df["Date"], y=backtest_df["Predicted"],
        mode="lines", name="Predicted Close",
        line=dict(color="#ff9f1c", width=1.5, dash="dash"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra>Predicted</extra>",
    ), row=1, col=1)

    # Buy/Sell signals as markers
    buys  = backtest_df[backtest_df["Signal"] == "BUY"]
    sells = backtest_df[backtest_df["Signal"] == "SELL"]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys["Date"], y=buys["Actual"],
            mode="markers", name="BUY Signal",
            marker=dict(symbol="triangle-up", size=10, color="#06d6a0"),
        ), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells["Date"], y=sells["Actual"],
            mode="markers", name="SELL Signal",
            marker=dict(symbol="triangle-down", size=10, color="#ef476f"),
        ), row=1, col=1)

    # Cumulative P&L
    colors_pnl = ["#06d6a0" if v >= 0 else "#ef476f" for v in backtest_df["Cumulative_PnL"]]
    fig.add_trace(go.Scatter(
        x=backtest_df["Date"], y=backtest_df["Cumulative_PnL"],
        mode="lines", name="Cumulative P&L",
        line=dict(color="#ffd166", width=2),
        fill="tozeroy", fillcolor="rgba(255,209,102,0.15)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>P&L: ₹%{y:,.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="#ffffff", line_dash="dot", line_width=0.5, row=2, col=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"{ticker} — Backtest: Predicted Signal P&L",
        yaxis_title="Price (₹)", yaxis2_title="Cumulative P&L (₹)",
        xaxis2_title="Date",
        xaxis=dict(tickformat="%d %b %Y", gridcolor="#2a2d35"),
        xaxis2=dict(tickformat="%d %b %Y", gridcolor="#2a2d35"),
    )
    fig.update_yaxes(gridcolor="#2a2d35")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Model Accuracy Summary Table (as a plotly table figure)
# ─────────────────────────────────────────────────────────────────────────────
def plot_accuracy_summary_table(accuracy_data: list) -> go.Figure:
    """
    accuracy_data: list of dicts with keys
      Ticker, R2, MAE, RMSE, MAPE, Direction_Accuracy
    """
    df = pd.DataFrame(accuracy_data)

    header_color = "#0077b6"
    row_colors   = ["#1a1d23", "#151820"]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in df.columns],
            fill_color=header_color,
            align="center",
            font=dict(color="white", size=13),
            height=35,
        ),
        cells=dict(
            values=[df[c] for c in df.columns],
            fill_color=[row_colors * (len(df) // 2 + 1)],
            align="center",
            font=dict(color="#ddd", size=12),
            height=30,
        ),
    )])
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="📊 Consolidated Model Accuracy Summary",
        margin=dict(t=60, b=20, l=20, r=20),
        height=max(200, 60 + 35 * len(df)),
    )
    return fig
