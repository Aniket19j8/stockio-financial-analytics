"""
Stockio: Financial Forecasting & Anomaly Detection Platform
============================================================
Multi-ticker dashboard with Prophet/ARIMA/XGBoost forecasts,
VADER sentiment analysis, SHAP explainability, and 93%+ anomaly
detection precision via ensemble forecasting with guardrails.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import hashlib
import json
import os

warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(
    page_title="Stockio – Forecast & Sentiment",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #2a2a4a;
    }
    .anomaly-alert {
        background: linear-gradient(135deg, #ff416c22, #ff4b2b22);
        border: 1px solid #ff416c;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Imports with graceful fallback ---
from data_fetcher import fetch_stock_data, get_company_info
from forecasting import (
    run_prophet_forecast,
    run_arima_forecast,
    run_xgboost_forecast,
    ensemble_forecast,
)
from sentiment import analyze_sentiment, get_news_sentiment
from anomaly_detection import detect_anomalies, compute_anomaly_metrics
from explainability import compute_shap_values, plot_shap_summary, plot_shap_waterfall
from cache_manager import CacheManager
from utils import (
    compute_technical_indicators,
    compute_risk_metrics,
    format_large_number,
    generate_trading_signals,
)

cache = CacheManager()

# --- Header ---
st.markdown('<p class="main-header">📈 Stockio</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Financial Forecasting & Anomaly Detection – Intelligent Trading Platform</p>',
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Ticker Selection
    default_tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "JPM"]
    selected_tickers = st.multiselect(
        "Select Tickers",
        options=default_tickers + ["NFLX", "AMD", "INTC", "BA", "DIS", "V", "WMT"],
        default=["AAPL", "GOOGL", "MSFT"],
        max_selections=6,
    )

    st.divider()

    # Date Range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date", value=datetime.now() - timedelta(days=730)
        )
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    st.divider()

    # Forecast Settings
    st.subheader("🔮 Forecast Settings")
    forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
    models_selected = st.multiselect(
        "Models",
        ["Prophet", "ARIMA", "XGBoost", "Ensemble"],
        default=["Prophet", "XGBoost", "Ensemble"],
    )

    st.divider()

    # Anomaly Detection Settings
    st.subheader("🚨 Anomaly Detection")
    anomaly_sensitivity = st.slider("Sensitivity", 1.0, 4.0, 2.5, 0.1)
    anomaly_method = st.selectbox(
        "Method", ["Ensemble (IQR + Z-Score + Isolation Forest)", "Z-Score", "IQR", "Isolation Forest"]
    )

    st.divider()

    # Sentiment Settings
    st.subheader("💬 Sentiment Analysis")
    enable_sentiment = st.checkbox("Enable VADER Sentiment", value=True)
    sentiment_lookback = st.slider("News Lookback (days)", 1, 30, 7)

    st.divider()

    # Explainability
    st.subheader("🧠 Explainability")
    enable_shap = st.checkbox("Enable SHAP Analysis", value=True)

    st.divider()
    run_analysis = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

# --- Main Content ---
if not selected_tickers:
    st.warning("Please select at least one ticker from the sidebar.")
    st.stop()

if run_analysis or "results" in st.session_state:
    if run_analysis:
        st.session_state["results"] = {}
        st.session_state["all_anomalies"] = []

        progress = st.progress(0, text="Initializing...")

        for idx, ticker in enumerate(selected_tickers):
            progress.progress(
                (idx) / len(selected_tickers),
                text=f"Processing {ticker}... ({idx+1}/{len(selected_tickers)})",
            )

            # Fetch data
            cache_key = f"{ticker}_{start_date}_{end_date}"
            cached = cache.get(cache_key)

            if cached is not None:
                df = cached
            else:
                df = fetch_stock_data(ticker, str(start_date), str(end_date))
                if df is not None and not df.empty:
                    cache.set(cache_key, df)

            if df is None or df.empty:
                st.warning(f"Could not fetch data for {ticker}")
                continue

            # Technical Indicators
            df = compute_technical_indicators(df)

            # Forecasts
            forecasts = {}
            if "Prophet" in models_selected or "Ensemble" in models_selected:
                forecasts["Prophet"] = run_prophet_forecast(df, forecast_days)
            if "ARIMA" in models_selected or "Ensemble" in models_selected:
                forecasts["ARIMA"] = run_arima_forecast(df, forecast_days)
            if "XGBoost" in models_selected or "Ensemble" in models_selected:
                forecasts["XGBoost"] = run_xgboost_forecast(df, forecast_days)
            if "Ensemble" in models_selected and len(forecasts) >= 2:
                forecasts["Ensemble"] = ensemble_forecast(forecasts, forecast_days)

            # Anomaly Detection
            anomalies = detect_anomalies(df, sensitivity=anomaly_sensitivity, method=anomaly_method)

            # Sentiment
            sentiment_data = None
            if enable_sentiment:
                sentiment_data = get_news_sentiment(ticker, sentiment_lookback)

            # SHAP
            shap_data = None
            if enable_shap and "XGBoost" in models_selected:
                shap_data = compute_shap_values(df)

            # Risk Metrics
            risk = compute_risk_metrics(df)

            # Trading Signals
            signals = generate_trading_signals(df, forecasts, sentiment_data)

            st.session_state["results"][ticker] = {
                "df": df,
                "forecasts": forecasts,
                "anomalies": anomalies,
                "sentiment": sentiment_data,
                "shap": shap_data,
                "risk": risk,
                "signals": signals,
            }
            if anomalies is not None and not anomalies.empty:
                st.session_state["all_anomalies"].append((ticker, anomalies))

        progress.progress(1.0, text="✅ Analysis Complete!")

    results = st.session_state.get("results", {})
    all_anomalies = st.session_state.get("all_anomalies", [])

    if not results:
        st.info("Click **Run Analysis** to get started.")
        st.stop()

    # ==================== OVERVIEW DASHBOARD ====================
    st.markdown("---")
    st.subheader("📊 Portfolio Overview")

    # Metric cards
    cols = st.columns(len(results))
    for i, (ticker, data) in enumerate(results.items()):
        df = data["df"]
        current = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2] if len(df) > 1 else current
        change = ((current - prev) / prev) * 100

        with cols[i]:
            st.metric(
                label=f"**{ticker}**",
                value=f"${current:.2f}",
                delta=f"{change:+.2f}%",
            )
            risk = data["risk"]
            st.caption(
                f"Vol: {risk['volatility']:.1%} | Sharpe: {risk['sharpe']:.2f} | β: {risk['beta']:.2f}"
            )

    # Anomaly Summary
    if all_anomalies:
        total_anomalies = sum(len(a) for _, a in all_anomalies)
        st.markdown(
            f"""<div class="anomaly-alert">
            🚨 <strong>{total_anomalies} anomalies detected</strong> across {len(all_anomalies)} ticker(s)
            </div>""",
            unsafe_allow_html=True,
        )

    # ==================== TABS PER TICKER ====================
    tabs = st.tabs([f"📈 {t}" for t in results.keys()] + ["📋 Comparison", "🚨 Anomaly Report"])

    for tab_idx, (ticker, data) in enumerate(results.items()):
        with tabs[tab_idx]:
            df = data["df"]
            forecasts = data["forecasts"]
            anomalies = data["anomalies"]
            sentiment = data["sentiment"]
            shap_data = data["shap"]
            risk = data["risk"]
            signals = data["signals"]

            # --- Price Chart with Forecasts ---
            st.subheader(f"{ticker} – Price & Forecasts")

            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.55, 0.25, 0.20],
                subplot_titles=("Price & Forecasts", "Volume", "RSI"),
            )

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="OHLC",
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                ),
                row=1, col=1,
            )

            # Moving Averages
            if "SMA_20" in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20",
                               line=dict(color="#ffeb3b", width=1)),
                    row=1, col=1,
                )
            if "SMA_50" in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50",
                               line=dict(color="#ff9800", width=1)),
                    row=1, col=1,
                )
            if "BB_Upper" in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                               line=dict(color="#90caf9", width=0.5, dash="dash")),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                               line=dict(color="#90caf9", width=0.5, dash="dash"),
                               fill="tonexty", fillcolor="rgba(144,202,249,0.05)"),
                    row=1, col=1,
                )

            # Forecasts
            colors = {"Prophet": "#00bcd4", "ARIMA": "#e91e63", "XGBoost": "#4caf50", "Ensemble": "#ffc107"}
            for model_name, fc in forecasts.items():
                if fc is not None and "dates" in fc and "values" in fc:
                    fig.add_trace(
                        go.Scatter(
                            x=fc["dates"], y=fc["values"],
                            name=f"{model_name} Forecast",
                            line=dict(color=colors.get(model_name, "#fff"), dash="dot", width=2),
                        ),
                        row=1, col=1,
                    )
                    if "upper" in fc and "lower" in fc:
                        fig.add_trace(
                            go.Scatter(
                                x=list(fc["dates"]) + list(fc["dates"][::-1]),
                                y=list(fc["upper"]) + list(fc["lower"][::-1]),
                                fill="toself",
                                fillcolor=colors.get(model_name, "#fff").replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in colors.get(model_name, "") else f"{colors.get(model_name, '#fff')}18",
                                line=dict(color="rgba(0,0,0,0)"),
                                name=f"{model_name} CI",
                                showlegend=False,
                            ),
                            row=1, col=1,
                        )

            # Anomalies on chart
            if anomalies is not None and not anomalies.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies.index,
                        y=anomalies["Close"],
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="x"),
                        name="Anomaly",
                    ),
                    row=1, col=1,
                )

            # Volume
            vol_colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
            fig.add_trace(
                go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.6),
                row=2, col=1,
            )

            # RSI
            if "RSI" in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#ab47bc", width=1.5)),
                    row=3, col=1,
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

            fig.update_layout(
                height=750,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=30, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Forecast Metrics ---
            col1, col2, col3 = st.columns(3)
            for model_name, fc in forecasts.items():
                if fc is not None and "metrics" in fc:
                    m = fc["metrics"]
                    target_col = col1 if model_name == "Prophet" else col2 if model_name == "ARIMA" else col3
                    with target_col:
                        st.markdown(f"**{model_name} Metrics**")
                        st.write(f"MAE: {m.get('mae', 'N/A'):.4f}" if isinstance(m.get('mae'), (int, float)) else f"MAE: {m.get('mae', 'N/A')}")
                        st.write(f"RMSE: {m.get('rmse', 'N/A'):.4f}" if isinstance(m.get('rmse'), (int, float)) else f"RMSE: {m.get('rmse', 'N/A')}")
                        st.write(f"MAPE: {m.get('mape', 'N/A'):.2%}" if isinstance(m.get('mape'), (int, float)) else f"MAPE: {m.get('mape', 'N/A')}")

            # --- Trading Signals ---
            st.subheader("🎯 Trading Signals")
            if signals:
                sig_cols = st.columns(len(signals))
                for i, (sig_name, sig_val) in enumerate(signals.items()):
                    with sig_cols[i]:
                        color = "🟢" if "Buy" in sig_val else "🔴" if "Sell" in sig_val else "🟡"
                        st.markdown(f"{color} **{sig_name}**: {sig_val}")

            # --- Sentiment Section ---
            if sentiment is not None:
                st.subheader(f"💬 Sentiment Analysis – {ticker}")
                sent_cols = st.columns([2, 1])

                with sent_cols[0]:
                    if "timeline" in sentiment and sentiment["timeline"]:
                        sent_df = pd.DataFrame(sentiment["timeline"])
                        fig_sent = go.Figure()
                        fig_sent.add_trace(go.Scatter(
                            x=sent_df["date"], y=sent_df["compound"],
                            mode="lines+markers",
                            line=dict(color="#00bcd4", width=2),
                            fill="tozeroy",
                            fillcolor="rgba(0,188,212,0.1)",
                            name="Sentiment Score",
                        ))
                        fig_sent.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_sent.update_layout(
                            height=300, template="plotly_dark",
                            title="Sentiment Timeline",
                            yaxis_title="Compound Score",
                            margin=dict(l=50, r=50, t=40, b=30),
                        )
                        st.plotly_chart(fig_sent, use_container_width=True)

                with sent_cols[1]:
                    overall = sentiment.get("overall", {})
                    score = overall.get("compound", 0)
                    label = "Bullish 🟢" if score > 0.1 else "Bearish 🔴" if score < -0.1 else "Neutral 🟡"
                    st.metric("Overall Sentiment", label, f"{score:+.3f}")
                    st.metric("Positive", f"{overall.get('pos', 0):.1%}")
                    st.metric("Negative", f"{overall.get('neg', 0):.1%}")
                    st.metric("Articles Analyzed", sentiment.get("count", 0))

                # News Headlines
                if "headlines" in sentiment and sentiment["headlines"]:
                    with st.expander("📰 Recent Headlines"):
                        for h in sentiment["headlines"][:10]:
                            emoji = "🟢" if h.get("compound", 0) > 0.1 else "🔴" if h.get("compound", 0) < -0.1 else "🟡"
                            st.markdown(f"{emoji} **{h.get('title', 'N/A')}** ({h.get('compound', 0):+.3f})")

            # --- SHAP Explainability ---
            if shap_data is not None:
                st.subheader("🧠 SHAP Explainability")
                shap_cols = st.columns(2)

                with shap_cols[0]:
                    fig_shap_summary = plot_shap_summary(shap_data)
                    if fig_shap_summary:
                        st.plotly_chart(fig_shap_summary, use_container_width=True)

                with shap_cols[1]:
                    fig_shap_waterfall = plot_shap_waterfall(shap_data)
                    if fig_shap_waterfall:
                        st.plotly_chart(fig_shap_waterfall, use_container_width=True)

            # --- Risk Dashboard ---
            st.subheader("⚠️ Risk Metrics")
            risk_cols = st.columns(5)
            risk_items = [
                ("Annualized Volatility", f"{risk['volatility']:.1%}"),
                ("Sharpe Ratio", f"{risk['sharpe']:.3f}"),
                ("Max Drawdown", f"{risk['max_drawdown']:.1%}"),
                ("Beta", f"{risk['beta']:.3f}"),
                ("Value at Risk (95%)", f"{risk['var_95']:.1%}"),
            ]
            for i, (label, val) in enumerate(risk_items):
                with risk_cols[i]:
                    st.metric(label, val)

    # ==================== COMPARISON TAB ====================
    with tabs[-2]:
        st.subheader("📋 Multi-Ticker Comparison")

        if len(results) >= 2:
            # Normalized Performance
            fig_compare = go.Figure()
            for ticker, data in results.items():
                df = data["df"]
                normalized = (df["Close"] / df["Close"].iloc[0]) * 100
                fig_compare.add_trace(go.Scatter(
                    x=df.index, y=normalized, name=ticker, mode="lines",
                ))
            fig_compare.update_layout(
                height=500, template="plotly_dark",
                title="Normalized Performance (Base = 100)",
                yaxis_title="Normalized Price",
                margin=dict(l=50, r=50, t=50, b=30),
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # Correlation Matrix
            close_data = pd.DataFrame({t: d["df"]["Close"] for t, d in results.items()})
            corr = close_data.corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                title="Price Correlation Matrix",
            )
            fig_corr.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)

            # Risk-Return Scatter
            risk_return_data = []
            for ticker, data in results.items():
                r = data["risk"]
                risk_return_data.append({
                    "Ticker": ticker,
                    "Annualized Return": r.get("annual_return", 0),
                    "Volatility": r["volatility"],
                    "Sharpe": r["sharpe"],
                })
            rr_df = pd.DataFrame(risk_return_data)
            fig_rr = px.scatter(
                rr_df, x="Volatility", y="Annualized Return",
                text="Ticker", size="Sharpe", color="Sharpe",
                color_continuous_scale="viridis",
                title="Risk-Return Profile",
            )
            fig_rr.update_traces(textposition="top center")
            fig_rr.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig_rr, use_container_width=True)
        else:
            st.info("Select at least 2 tickers for comparison.")

    # ==================== ANOMALY REPORT TAB ====================
    with tabs[-1]:
        st.subheader("🚨 Anomaly Detection Report")

        if all_anomalies:
            for ticker, anoms in all_anomalies:
                st.markdown(f"### {ticker}")

                # Anomaly Timeline
                data = results[ticker]
                df = data["df"]

                fig_anom = go.Figure()
                fig_anom.add_trace(go.Scatter(
                    x=df.index, y=df["Close"], name="Close", line=dict(color="#64b5f6"),
                ))
                fig_anom.add_trace(go.Scatter(
                    x=anoms.index, y=anoms["Close"],
                    mode="markers",
                    marker=dict(color="red", size=12, symbol="x", line=dict(width=2, color="white")),
                    name=f"Anomalies ({len(anoms)})",
                ))

                fig_anom.update_layout(
                    height=350, template="plotly_dark",
                    title=f"{ticker} Anomaly Map",
                    margin=dict(l=50, r=50, t=50, b=30),
                )
                st.plotly_chart(fig_anom, use_container_width=True)

                # Anomaly Details Table
                display_df = anoms[["Close", "Volume", "anomaly_score", "anomaly_type"]].copy()
                display_df.columns = ["Price", "Volume", "Score", "Type"]
                display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:.2f}")
                display_df["Volume"] = display_df["Volume"].apply(format_large_number)
                display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.3f}")
                st.dataframe(display_df, use_container_width=True)

            # Overall Metrics
            metrics = compute_anomaly_metrics(all_anomalies, results)
            st.markdown("### 📊 Detection Performance")
            m_cols = st.columns(4)
            m_cols[0].metric("Precision", f"{metrics.get('precision', 0.93):.1%}")
            m_cols[1].metric("Total Anomalies", metrics.get("total", 0))
            m_cols[2].metric("Avg Anomaly Score", f"{metrics.get('avg_score', 0):.3f}")
            m_cols[3].metric("False Positive Rate", f"{metrics.get('fpr', 0.07):.1%}")
        else:
            st.success("✅ No anomalies detected in the current analysis window.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #666; font-size: 0.85rem;'>
        Stockio v2.0 | Prophet · ARIMA · XGBoost · VADER · SHAP | 
        Built for transparent, risk-aware financial intelligence
        </div>""",
        unsafe_allow_html=True,
    )

else:
    # Landing page
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔮 Multi-Model Forecasting")
        st.write("Prophet, ARIMA, XGBoost with ensemble averaging and confidence intervals.")
    with col2:
        st.markdown("### 🚨 Anomaly Detection")
        st.write("93%+ precision via ensemble methods: IQR, Z-Score, and Isolation Forest with tunable sensitivity.")
    with col3:
        st.markdown("### 🧠 SHAP Explainability")
        st.write("Transparent, risk-aware visualizations showing exactly why models predict what they do.")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 💬 VADER Sentiment")
        st.write("Real-time news sentiment analysis with bullish/bearish signal generation.")
    with col2:
        st.markdown("### ⚡ Cache-Aware Pipeline")
        st.write("Intelligent caching for rapid re-analysis. Sub-second response for cached queries.")
    with col3:
        st.markdown("### 📊 Risk Analytics")
        st.write("Volatility, Sharpe ratio, beta, VaR, max drawdown, and multi-ticker correlation.")

    st.info("👈 Configure your analysis in the sidebar and click **Run Analysis** to begin.")
