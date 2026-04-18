"""
FinPulse Quant Engine — Streamlit Dashboard (FastAPI backend)
==============================================================
Launch:
  1) uvicorn api_server:app --host 0.0.0.0 --port 8000
  2) streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import httpx
import warnings
warnings.filterwarnings("ignore")

API_BASE = "http://localhost:8000"

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinPulse Quant Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
.block-container {padding-top: 1.5rem;}
div[data-testid="metric-container"] {
    background-color: #1a1d23;
    border: 1px solid #2a2d35;
    border-radius: 8px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

from visualization import (
    plot_portfolio_trend, plot_quantity_bar, plot_allocation_pie,
    plot_correlation_heatmap, plot_monte_carlo_paths,
    plot_var_histogram, plot_max_drawdown, plot_predicted_vs_actual,
    plot_candlestick_chart, plot_backtest_results,
    plot_accuracy_summary_table,
)

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:0.5rem 0 1rem 0;'>
  <h1 style='margin:0;font-size:2rem;'>📈 FinPulse Quant Engine</h1>
  <p style='color:#aaa;margin:4px 0 0 0;font-size:0.95rem;'>
    Multi-Stock Portfolio Prediction · GARCH Volatility · Monte Carlo Risk Analytics
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# STOCK SELECTION (simplified — no MC slider, no trading days slider)
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "WIPRO.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS",
    "SBIN.NS", "ADANIENT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS",
]

st.markdown("### 🎯 Stock Selection & Portfolio Setup")
selected_tickers = st.multiselect(
    "Select NSE Stocks",
    options=DEFAULT_TICKERS,
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"],
)

quantities = {}
if selected_tickers:
    st.markdown("#### 🔢 Quantity per Stock")
    n_cols   = min(len(selected_tickers), 4)
    qty_cols = st.columns(n_cols)
    for i, ticker in enumerate(selected_tickers):
        with qty_cols[i % n_cols]:
            quantities[ticker] = st.number_input(
                ticker, min_value=1, max_value=100000,
                value=10, step=1, key=f"qty_{ticker}",
            )

st.divider()
run_button = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

if not run_button:
    st.info("⬆ Select stocks, set quantities, and click **Run Full Analysis** to begin.")
    st.stop()
if not selected_tickers:
    st.error("Please select at least one stock.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# CALL FASTAPI BACKEND
# ═════════════════════════════════════════════════════════════════════════════
progress_bar = st.progress(0, text="Initializing...")
status_text  = st.empty()

def update_progress(pct, msg):
    progress_bar.progress(pct, text=msg)
    status_text.markdown(f"*{msg}*")

update_progress(5, "📡 Sending request to FastAPI backend…")

try:
    resp = httpx.post(
        f"{API_BASE}/analyse",
        json={"tickers": selected_tickers, "quantities": quantities},
        timeout=600.0,
    )
    if resp.status_code != 200:
        st.error(f"Backend error ({resp.status_code}): {resp.text}")
        st.stop()
    data = resp.json()
except httpx.ConnectError:
    st.error("⚠️ Cannot connect to FastAPI backend. Make sure `uvicorn api_server:app --port 8000` is running.")
    st.stop()
except Exception as e:
    st.error(f"Request failed: {e}")
    st.stop()

update_progress(70, "📊 Processing results…")

# ═════════════════════════════════════════════════════════════════════════════
# UNPACK RESPONSE
# ═════════════════════════════════════════════════════════════════════════════
valid_tickers     = data["valid_tickers"]
error_tickers     = data["error_tickers"]
quantities        = data["quantities"]
current_prices    = data["current_prices"]
stock_summaries   = data["stock_summaries"]
stock_results     = data["stock_results"]
ps                = data["portfolio_summary"]
trend_data        = data["trend_data"]
mc_data           = data["mc_results"]
risk_report       = data["risk_report"]
corr_data         = data["corr_matrix"]
stock_values      = data["stock_values"]
fundamentals_dict = data["fundamentals"]
candle_data       = data["candlestick_data"]
accuracy_summary  = data["accuracy_summary"]

for et in error_tickers:
    st.warning(f"⚠️ Model failed for {et}")

if not valid_tickers:
    st.error("All stock models failed.")
    st.stop()

# Rebuild trend_df
trend_df = pd.DataFrame(trend_data)
trend_df["Date"] = pd.to_datetime(trend_df["Date"], errors="coerce", utc=True)
trend_df["Date"] = trend_df["Date"].dt.tz_localize(None)
trend_df.dropna(subset=["Date"], inplace=True)
trend_df.set_index("Date", inplace=True)

# Rebuild corr_matrix
corr_matrix = pd.DataFrame(
    corr_data["values"],
    columns=corr_data["columns"],
    index=corr_data["columns"],
)

# Rebuild mc_results for plotting
mc_results_plot = {
    "all_paths":         np.array(mc_data["all_paths"]),
    "final_values":      np.array(mc_data["final_values"]),
    "percentile_5":      mc_data["percentile_5"],
    "percentile_95":     mc_data["percentile_95"],
    "expected_value":    mc_data["expected_value"],
    "current_portfolio": mc_data["current_portfolio"],
    "n_simulations":     mc_data["n_simulations"],
    "n_days":            mc_data["n_days"],
}

update_progress(90, "🎨 Rendering dashboard…")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Stock Predictions",
    "🕯️ Candlestick Charts",
    "💼 Portfolio Panel",
    "⚠️ Risk Analytics",
    "🎲 Monte Carlo",
    "🔬 Backtest & Accuracy",
    "📉 Training Charts",
    "🏥 Financial Health KPIs",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Stock Predictions (with DATES on x-axis + candlestick future)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📈 Per-Stock Prediction Results")
    st.dataframe(pd.DataFrame(stock_summaries), use_container_width=True)
    st.divider()
    st.markdown("### 🔮 Predicted vs Actual + 2-Day Forecast")

    for t in valid_tickers:
        res = stock_results[t]
        o1  = res["predicted_ohlc"]
        o2  = res["predicted_ohlc2"]
        flist = [
            {"Open": o1[0], "High": o1[1], "Low": o1[2], "Close": o1[3]},
            {"Open": o2[0], "High": o2[1], "Low": o2[2], "Close": o2[3]},
        ]
        val_dates = res.get("val_dates")

        fig = plot_predicted_vs_actual(
            t,
            np.array(res["val_actual_close"]),
            np.array(res["val_preds_close"]),
            future_ohlc_list=flist,
            val_dates=val_dates,
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{t} Current", f"₹{current_prices[t]:,.2f}")
        c2.metric(
            f"{t} Day +1",
            f"₹{flist[0]['Close']:,.2f}",
            f"{((flist[0]['Close'] - current_prices[t]) / current_prices[t] * 100):+.2f}%",
        )
        c3.metric(
            f"{t} Day +2",
            f"₹{flist[1]['Close']:,.2f}",
            f"{((flist[1]['Close'] - current_prices[t]) / current_prices[t] * 100):+.2f}%",
        )
        st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Candlestick Charts (historical OHLCV)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🕯️ Historical Candlestick Charts")
    st.caption("Last 90 trading days · OHLC with volume bars")

    for t in valid_tickers:
        cd = candle_data[t]
        df_candle = pd.DataFrame({
            "Open":   cd["open"],
            "High":   cd["high"],
            "Low":    cd["low"],
            "Close":  cd["close"],
            "Volume": cd["volume"],
        }, index=pd.to_datetime(cd["dates"]))
        fig = plot_candlestick_chart(t, df_candle, last_n=90)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Portfolio Panel
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 💼 Portfolio Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Portfolio",   f"₹{ps['current_total']:,.2f}")
    c2.metric("Predicted Portfolio", f"₹{ps['predicted_total']:,.2f}",
              f"{ps['pct_change']:+.2f}%")
    c3.metric("Expected P&L",
              f"₹{ps['predicted_total'] - ps['current_total']:,.2f}",
              f"{ps['pct_change']:+.2f}%")
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_portfolio_trend(trend_df), use_container_width=True)
    with col_b:
        st.plotly_chart(
            plot_allocation_pie(stock_values, ps["current_total"]),
            use_container_width=True,
        )
    st.plotly_chart(plot_quantity_bar(quantities), use_container_width=True)

    st.markdown("#### 📋 Weight Breakdown")
    weight_df = pd.DataFrame([
        {
            "Ticker":             t,
            "Weight (%)":         ps["weights"].get(t, 0),
            "Current Value (Rs)": stock_values[t],
        }
        for t in valid_tickers
    ]).sort_values("Weight (%)", ascending=False)
    st.dataframe(weight_df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Risk Analytics (now with 3 VaR methods + Parametric VaR)
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## ⚠️ Risk Analytics")
    st.markdown("""
    <div style='background:#1a1d23;border:1px solid #2a2d35;border-radius:10px;padding:12px;margin-bottom:16px;'>
        <b style='color:#00b4d8;'>3 VaR Methods:</b>
        <span style='color:#aaa;'> ① Historical Simulation · ② Parametric (Variance-Covariance) · ③ Monte Carlo VaR</span>
    </div>
    """, unsafe_allow_html=True)

    if risk_report:
        st.markdown("#### ① Historical VaR")
        h1, h2, h3 = st.columns(3)
        h1.metric("Historical VaR 95% (1d)", f"₹{risk_report.get('var_95', 0):,.0f}")
        h2.metric("Historical VaR 99% (1d)", f"₹{risk_report.get('var_99', 0):,.0f}")
        h3.metric("CVaR 95% (Expected Shortfall)", f"₹{risk_report.get('cvar_95', 0):,.0f}")

        st.markdown("#### ② Parametric VaR (Normal Distribution)")
        p1, p2 = st.columns(2)
        p1.metric("Parametric VaR 95% (1d)", f"₹{risk_report.get('param_var_95', 0):,.0f}")
        p2.metric("Parametric VaR 99% (1d)", f"₹{risk_report.get('param_var_99', 0):,.0f}")

        st.markdown("#### ③ Monte Carlo VaR")
        m1, m2 = st.columns(2)
        m1.metric("MC VaR (30-day)", f"₹{risk_report.get('mc_var', 0):,.0f}")
        m2.metric("Sharpe Ratio",    f"{risk_report.get('sharpe', 0):.3f}")

        st.divider()
        r5, r6 = st.columns(2)
        r5.metric("Max Drawdown", f"{risk_report.get('max_drawdown', 0):.2f}%")
        r6.metric("Portfolio Value", f"₹{ps['current_total']:,.0f}")

    st.divider()
    st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)

    if risk_report and len(risk_report.get("portfolio_returns", [])) > 0:
        st.plotly_chart(
            plot_var_histogram(
                np.array(risk_report["portfolio_returns"]),
                risk_report.get("var_95", 0),
                ps["current_total"],
            ),
            use_container_width=True,
        )

    if risk_report and len(risk_report.get("portfolio_values", [])) > 0:
        st.plotly_chart(
            plot_max_drawdown(pd.Series(risk_report["portfolio_values"])),
            use_container_width=True,
        )

    st.markdown("#### 📊 Per-Stock GARCH Volatility")
    garch_df = pd.DataFrame([
        {
            "Ticker":            t,
            "GARCH Vol (Ann %)": round(stock_results[t]["garch_vol"] * 100, 2),
            "RAF":               round(fundamentals_dict[t].get("RAF", 1.0), 4),
            "Beta":              fundamentals_dict[t].get("Beta", 1.0),
            "DE Ratio":          fundamentals_dict[t].get("DE_Ratio", 0.0),
        }
        for t in valid_tickers
    ])
    st.dataframe(garch_df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Monte Carlo
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🎲 Monte Carlo Simulation")
    st.caption(f"{mc_data['n_simulations']:,} paths · 30 trading days · GBM with GARCH volatility")
    st.plotly_chart(plot_monte_carlo_paths(mc_results_plot), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Portfolio", f"₹{mc_data['current_portfolio']:,.0f}")
    m2.metric("Expected (30-day)", f"₹{mc_data['expected_value']:,.0f}")
    m3.metric("Worst 5%",         f"₹{mc_data['percentile_5']:,.0f}")
    m4.metric("Best 95%",         f"₹{mc_data['percentile_95']:,.0f}")

    mc_gain = mc_data["expected_value"] - mc_data["current_portfolio"]
    mc_pct  = (mc_gain / mc_data["current_portfolio"] * 100
               if mc_data["current_portfolio"] > 0 else 0)
    st.info(f"📈 Expected change over 30 days: **₹{mc_gain:,.0f}** ({mc_pct:+.2f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — Backtest & Model Accuracy
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 🔬 Historical Backtesting & Model Accuracy")
    st.markdown("""
    <div style='background:#1a1d23;border:1px solid #2a2d35;border-radius:10px;padding:12px;margin-bottom:16px;'>
        <p style='color:#aaa;margin:0;font-size:0.9rem;'>
            Backtesting simulates trading on the <b>validation set</b>: if the model predicts price
            will rise → BUY signal; if fall → SELL signal. Cumulative P&L shows how the strategy
            would have performed historically. Direction Accuracy = % of days the model correctly
            predicted up/down movement.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Consolidated accuracy table
    st.markdown("### 📊 Consolidated Model Accuracy")
    fig_acc = plot_accuracy_summary_table(accuracy_summary)
    st.plotly_chart(fig_acc, use_container_width=True)

    st.divider()

    # Per-stock backtest charts
    for t in valid_tickers:
        res = stock_results[t]
        bt  = res.get("backtest", [])
        if not bt:
            continue

        bt_df = pd.DataFrame(bt)
        bt_df["Date"] = pd.to_datetime(bt_df["Date"])

        st.markdown(f"### {t}")

        # Metrics row
        bc1, bc2, bc3, bc4 = st.columns(4)
        n_trades = len(bt_df)
        wins     = len(bt_df[bt_df["Daily_PnL"] > 0])
        final_pnl = bt_df["Cumulative_PnL"].iloc[-1] if len(bt_df) > 0 else 0
        dir_acc   = res.get("direction_accuracy", 0)

        bc1.metric("Total Trades", f"{n_trades}")
        bc2.metric("Win Rate", f"{(wins/max(n_trades,1)*100):.1f}%")
        bc3.metric("Direction Accuracy", f"{dir_acc:.1f}%")
        bc4.metric("Final P&L", f"₹{final_pnl:,.2f}",
                   f"{'🟢' if final_pnl >= 0 else '🔴'}")

        fig_bt = plot_backtest_results(bt_df, t)
        st.plotly_chart(fig_bt, use_container_width=True)
        st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — Training Loss Curves
# ═════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## 📉 Model Training History")
    import plotly.graph_objects as go

    for t in valid_tickers:
        h  = stock_results[t]["history"]
        hp = stock_results[t]["hyperparams"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=h["loss"], mode="lines", name="Train Loss",
            line=dict(color="#00b4d8", width=2),
        ))
        fig.add_trace(go.Scatter(
            y=h["val_loss"], mode="lines", name="Val Loss",
            line=dict(color="#ff9f1c", width=2, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1d23",
            title=f"{t} — Training Loss | seq_len={hp['seq_len']} epochs_run={len(h['loss'])}",
            xaxis_title="Epoch", yaxis_title="Loss",
            font=dict(color="#ddd"), hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 — Financial Health KPIs (kept from previous version)
# ═════════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown("## 🏥 Financial Health KPIs")
    st.markdown("""
    <div style='background:#1a1d23;border:1px solid #2a2d35;border-radius:10px;padding:16px;margin-bottom:20px;'>
        <p style='color:#aaa;font-size:0.95rem;margin:0;'>
            Most KPIs below are derived directly from a company's <b style='color:#00b4d8;'>Income Statement</b>,
            <b style='color:#06d6a0;'>Balance Sheet</b>, and <b style='color:#ff9f1c;'>Cash Flow Statement</b>.
            Each KPI includes a <b>health indicator</b> showing what the metric signals about financial condition.
        </p>
    </div>
    """, unsafe_allow_html=True)

    def _fmt_cr(val):
        if val == 0: return "N/A"
        cr = val / 1e7
        return f"₹{cr:,.0f} Cr" if abs(cr) >= 100 else f"₹{cr:,.2f} Cr"

    def _fmt_pct(val):
        if val == 0: return "N/A"
        return f"{val*100:.2f}%" if abs(val) < 10 else f"{val:.2f}%"

    def _badge(label, color):
        return f"<span style='background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;'>{label}</span>"

    def _kpi_card(title, value_str, color, badge_html, tooltip):
        return f"""
        <div style='background:#1a1d23;border:1px solid #2a2d35;border-radius:8px;padding:12px;height:100%;'>
            <div style='color:#aaa;font-size:0.8rem;'>{title}</div>
            <div style='font-size:1.3rem;font-weight:700;color:{color};'>{value_str}</div>
            {badge_html}
            <div style='color:#666;font-size:0.75rem;margin-top:6px;'>{tooltip}</div>
        </div>"""

    for t in valid_tickers:
        f = fundamentals_dict[t]
        with st.expander(f"📊 {t}", expanded=(t == valid_tickers[0])):

            # ── Income Statement ─────────────────────────────────────
            st.markdown("<h4 style='color:#00b4d8;border-bottom:2px solid #00b4d8;padding-bottom:4px;'>📄 1. Income Statement KPIs</h4>", unsafe_allow_html=True)

            ic1, ic2, ic3, ic4 = st.columns(4)
            npm = f.get("NetProfitMargin", 0)
            npm_h = "Healthy" if npm > 0.10 else ("Moderate" if npm > 0.05 else "Weak")
            npm_c = "#06d6a0" if npm > 0.10 else ("#ff9f1c" if npm > 0.05 else "#ff4d4d")
            ic1.markdown(_kpi_card("Net Profit Margin", _fmt_pct(npm), "#00b4d8", _badge(npm_h, npm_c), "💡 >10% strong, 5-10% fair, <5% thin."), unsafe_allow_html=True)

            gm = f.get("GrossMargin", 0)
            gm_h = "Strong" if gm > 0.40 else ("Moderate" if gm > 0.20 else "Low")
            gm_c = "#06d6a0" if gm > 0.40 else ("#ff9f1c" if gm > 0.20 else "#ff4d4d")
            ic2.markdown(_kpi_card("Gross Margin", _fmt_pct(gm), "#00b4d8", _badge(gm_h, gm_c), "💡 Revenue - COGS / Revenue. >40% = moat."), unsafe_allow_html=True)

            ebitda_val = f.get("EBITDA", 0)
            eb_h = "Profitable" if ebitda_val > 0 else "Unprofitable"
            eb_c = "#06d6a0" if ebitda_val > 0 else "#ff4d4d"
            ic3.markdown(_kpi_card("EBITDA", _fmt_cr(ebitda_val), "#00b4d8", _badge(eb_h, eb_c), "💡 Core operating profitability."), unsafe_allow_html=True)

            rg = f.get("RevenueGrowth", 0)
            rg_h = "High Growth" if rg > 0.15 else ("Growing" if rg > 0 else "Declining")
            rg_c = "#06d6a0" if rg > 0.15 else ("#ff9f1c" if rg > 0 else "#ff4d4d")
            ic4.markdown(_kpi_card("Revenue Growth", _fmt_pct(rg), "#00b4d8", _badge(rg_h, rg_c), "💡 YoY revenue change."), unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            ic5, ic6, ic7, ic8 = st.columns(4)
            ic5.markdown(_kpi_card("Total Revenue", _fmt_cr(f.get("TotalRevenue", 0)), "#00b4d8", "", "💡 Top-line income."), unsafe_allow_html=True)
            ni = f.get("NetIncome", 0)
            ni_h = "Profit" if ni > 0 else "Loss"
            ni_c = "#06d6a0" if ni > 0 else "#ff4d4d"
            ic6.markdown(_kpi_card("Net Income", _fmt_cr(ni), "#00b4d8", _badge(ni_h, ni_c), "💡 Bottom line after all expenses."), unsafe_allow_html=True)
            om = f.get("OperatingMargin", 0)
            om_h = "Strong" if om > 0.15 else ("Fair" if om > 0.05 else "Weak")
            om_c = "#06d6a0" if om > 0.15 else ("#ff9f1c" if om > 0.05 else "#ff4d4d")
            ic7.markdown(_kpi_card("Operating Margin", _fmt_pct(om), "#00b4d8", _badge(om_h, om_c), "💡 Core ops profit as % of revenue."), unsafe_allow_html=True)
            eg = f.get("EarningsGrowth", 0)
            eg_h = "Accelerating" if eg > 0.10 else ("Stable" if eg > 0 else "Declining")
            eg_c = "#06d6a0" if eg > 0.10 else ("#ff9f1c" if eg > 0 else "#ff4d4d")
            ic8.markdown(_kpi_card("Earnings Growth", _fmt_pct(eg), "#00b4d8", _badge(eg_h, eg_c), "💡 YoY net earnings growth."), unsafe_allow_html=True)

            st.divider()

            # ── Balance Sheet ────────────────────────────────────────
            st.markdown("<h4 style='color:#06d6a0;border-bottom:2px solid #06d6a0;padding-bottom:4px;'>🏦 2. Balance Sheet KPIs</h4>", unsafe_allow_html=True)
            bc1, bc2, bc3, bc4 = st.columns(4)
            cr_val = f.get("CurrentRatio", 0)
            cr_h = "Strong" if cr_val > 1.5 else ("Adequate" if cr_val > 1.0 else "Risky")
            cr_c = "#06d6a0" if cr_val > 1.5 else ("#ff9f1c" if cr_val > 1.0 else "#ff4d4d")
            bc1.markdown(_kpi_card("Current Ratio", f"{cr_val:.2f}x", "#06d6a0", _badge(cr_h, cr_c), "💡 >1.5 liquid, <1 risky."), unsafe_allow_html=True)
            qr = f.get("QuickRatio", 0)
            qr_h = "Liquid" if qr > 1.0 else ("Tight" if qr > 0.5 else "Illiquid")
            qr_c = "#06d6a0" if qr > 1.0 else ("#ff9f1c" if qr > 0.5 else "#ff4d4d")
            bc2.markdown(_kpi_card("Quick Ratio", f"{qr:.2f}x", "#06d6a0", _badge(qr_h, qr_c), "💡 Excludes inventory. >1 = safe."), unsafe_allow_html=True)
            de = f.get("DE_Ratio", 0)
            de_h = "Conservative" if de < 50 else ("Moderate" if de < 150 else "High Leverage")
            de_c = "#06d6a0" if de < 50 else ("#ff9f1c" if de < 150 else "#ff4d4d")
            bc3.markdown(_kpi_card("Debt-to-Equity", f"{de:.1f}%", "#06d6a0", _badge(de_h, de_c), "💡 <50% conservative, >150% leveraged."), unsafe_allow_html=True)
            wc = f.get("WorkingCapital", 0)
            wc_h = "Positive" if wc > 0 else "Negative"
            wc_c = "#06d6a0" if wc > 0 else "#ff4d4d"
            bc4.markdown(_kpi_card("Working Capital", _fmt_cr(wc), "#06d6a0", _badge(wc_h, wc_c), "💡 Current Assets − Liabilities."), unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            bc5, bc6, bc7, bc8 = st.columns(4)
            bc5.markdown(_kpi_card("Total Cash", _fmt_cr(f.get("TotalCash", 0)), "#06d6a0", "", "💡 Cash & equivalents on hand."), unsafe_allow_html=True)
            bc6.markdown(_kpi_card("Total Debt", _fmt_cr(f.get("TotalDebt", 0)), "#06d6a0", "", "💡 All borrowings."), unsafe_allow_html=True)
            roe_val = f.get("ROE", 0)
            roe_h = "Excellent" if roe_val > 0.20 else ("Good" if roe_val > 0.10 else "Low")
            roe_c = "#06d6a0" if roe_val > 0.20 else ("#ff9f1c" if roe_val > 0.10 else "#ff4d4d")
            bc7.markdown(_kpi_card("ROE", _fmt_pct(roe_val), "#06d6a0", _badge(roe_h, roe_c), "💡 >20% excellent capital efficiency."), unsafe_allow_html=True)
            roa_val = f.get("ROA", 0)
            roa_h = "Efficient" if roa_val > 0.10 else ("Fair" if roa_val > 0.05 else "Low")
            roa_c = "#06d6a0" if roa_val > 0.10 else ("#ff9f1c" if roa_val > 0.05 else "#ff4d4d")
            bc8.markdown(_kpi_card("ROA", _fmt_pct(roa_val), "#06d6a0", _badge(roa_h, roa_c), "💡 >10% asset-light & efficient."), unsafe_allow_html=True)

            st.divider()

            # ── Cash Flow ────────────────────────────────────────────
            st.markdown("<h4 style='color:#ff9f1c;border-bottom:2px solid #ff9f1c;padding-bottom:4px;'>💰 3. Cash Flow Statement KPIs</h4>", unsafe_allow_html=True)
            cf1, cf2 = st.columns(2)
            ocf = f.get("OperatingCF", 0)
            ocf_h = "Cash Generating" if ocf > 0 else "Cash Burning"
            ocf_c = "#06d6a0" if ocf > 0 else "#ff4d4d"
            cf1.markdown(_kpi_card("Operating Cash Flow", _fmt_cr(ocf), "#ff9f1c", _badge(ocf_h, ocf_c), "💡 Cash from core operations. Lifeblood metric."), unsafe_allow_html=True)
            fcf = f.get("FreeCF", 0)
            fcf_h = "FCF Positive" if fcf > 0 else "FCF Negative"
            fcf_c = "#06d6a0" if fcf > 0 else "#ff4d4d"
            cf2.markdown(_kpi_card("Free Cash Flow", _fmt_cr(fcf), "#ff9f1c", _badge(fcf_h, fcf_c), "💡 OCF minus CapEx. Positive = sustainable."), unsafe_allow_html=True)

            st.divider()

            # ── Hybrid & Valuation ───────────────────────────────────
            st.markdown("<h4 style='color:#f72585;border-bottom:2px solid #f72585;padding-bottom:4px;'>🔗 4. Hybrid & Valuation KPIs</h4>", unsafe_allow_html=True)
            hc1, hc2, hc3, hc4 = st.columns(4)
            pe = f.get("PE_Ratio", 0)
            pe_h = "Value" if 0 < pe < 15 else ("Fair" if 15 <= pe < 25 else ("Growth Premium" if pe >= 25 else "N/A"))
            pe_c = "#06d6a0" if 0 < pe < 15 else ("#ff9f1c" if 15 <= pe < 25 else ("#f72585" if pe >= 25 else "#666"))
            hc1.markdown(_kpi_card("P/E Ratio", f"{pe:.1f}x", "#f72585", _badge(pe_h, pe_c), "💡 <15 value, 15-25 fair, >25 growth."), unsafe_allow_html=True)
            pb = f.get("PriceToBook", 0)
            pb_h = "Undervalued" if 0 < pb < 1 else ("Fair" if 1 <= pb < 3 else "Premium")
            pb_c = "#06d6a0" if 0 < pb < 1 else ("#ff9f1c" if 1 <= pb < 3 else "#f72585")
            hc2.markdown(_kpi_card("Price-to-Book", f"{pb:.2f}x", "#f72585", _badge(pb_h, pb_c), "💡 <1 undervalued, high = intangibles."), unsafe_allow_html=True)
            eveb = f.get("EV_to_EBITDA", 0)
            eveb_h = "Cheap" if 0 < eveb < 10 else ("Fair" if 10 <= eveb < 15 else "Expensive")
            eveb_c = "#06d6a0" if 0 < eveb < 10 else ("#ff9f1c" if 10 <= eveb < 15 else "#f72585")
            hc3.markdown(_kpi_card("EV/EBITDA", f"{eveb:.1f}x", "#f72585", _badge(eveb_h, eveb_c), "💡 <10 cheap, 10-15 fair, >15 premium."), unsafe_allow_html=True)
            dy = f.get("DividendYield", 0)
            dy_h = "Income Stock" if dy > 0.03 else ("Modest" if dy > 0 else "No Dividend")
            dy_c = "#06d6a0" if dy > 0.03 else ("#ff9f1c" if dy > 0 else "#666")
            hc4.markdown(_kpi_card("Dividend Yield", _fmt_pct(dy), "#f72585", _badge(dy_h, dy_c), "💡 >3% income stock."), unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            hc5, hc6, hc7, hc8 = st.columns(4)
            bt = f.get("Beta", 1.0)
            bt_h = "Defensive" if bt < 0.8 else ("Market-like" if bt < 1.2 else "Aggressive")
            bt_c = "#06d6a0" if bt < 0.8 else ("#ff9f1c" if bt < 1.2 else "#f72585")
            hc5.markdown(_kpi_card("Beta", f"{bt:.2f}", "#f72585", _badge(bt_h, bt_c), "💡 <1 defensive, >1 volatile."), unsafe_allow_html=True)
            peg = f.get("PEG_Ratio", 0)
            peg_h = "Undervalued" if 0 < peg < 1 else ("Fair" if 1 <= peg < 2 else "Overvalued")
            peg_c = "#06d6a0" if 0 < peg < 1 else ("#ff9f1c" if 1 <= peg < 2 else "#f72585")
            hc6.markdown(_kpi_card("PEG Ratio", f"{peg:.2f}", "#f72585", _badge(peg_h, peg_c), "💡 <1 bargain growth."), unsafe_allow_html=True)
            rpe = f.get("RevPerEmployee", 0)
            hc7.markdown(_kpi_card("Rev/Employee", _fmt_cr(rpe), "#f72585", "", "💡 Workforce productivity."), unsafe_allow_html=True)
            pr = f.get("PayoutRatio", 0)
            pr_h = "Sustainable" if 0 < pr < 0.60 else ("High" if pr >= 0.60 else "None")
            pr_c = "#06d6a0" if 0 < pr < 0.60 else ("#ff9f1c" if pr >= 0.60 else "#666")
            hc8.markdown(_kpi_card("Payout Ratio", _fmt_pct(pr), "#f72585", _badge(pr_h, pr_c), "💡 <60% sustainable dividend."), unsafe_allow_html=True)

    # ── Source Reference Table & Legend ───────────────────────────────────────
    st.divider()
    st.markdown("### 📋 KPI Source Reference")
    src_df = pd.DataFrame([
        {"Source": "📄 Income Statement", "Data Points": "Revenue, COGS, Net Income, Interest, Taxes", "KPIs": "Profit Margins, EBITDA, Revenue Growth"},
        {"Source": "🏦 Balance Sheet", "Data Points": "Assets, Liabilities, Equity, Inventory", "KPIs": "Current/Quick Ratio, D/E, Working Capital, ROE, ROA"},
        {"Source": "💰 Cash Flow", "Data Points": "Cash from Operations, CapEx", "KPIs": "OCF, FCF, Burn Rate"},
        {"Source": "🔗 Hybrid", "Data Points": "Revenue + Headcount, Price + Earnings", "KPIs": "P/E, EV/EBITDA, PEG, Beta, Dividend Yield"},
    ])
    st.dataframe(src_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style='background:#1a1d23;border:1px solid #2a2d35;border-radius:10px;padding:16px;margin-top:16px;'>
        <h4 style='margin:0 0 8px 0;color:#ddd;'>🔑 Health Badge Legend</h4>
        <p style='margin:4px 0;'><span style='background:#06d6a0;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;'>Green</span> &nbsp; Strong / Healthy</p>
        <p style='margin:4px 0;'><span style='background:#ff9f1c;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;'>Amber</span> &nbsp; Moderate / Fair</p>
        <p style='margin:4px 0;'><span style='background:#ff4d4d;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;'>Red</span> &nbsp; Weak / Risky</p>
        <p style='margin:4px 0;'><span style='background:#f72585;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;'>Pink</span> &nbsp; Premium / Aggressive</p>
    </div>
    """, unsafe_allow_html=True)

update_progress(100, "✅ Analysis complete!")
status_text.success("✅ FinPulse analysis complete!")
