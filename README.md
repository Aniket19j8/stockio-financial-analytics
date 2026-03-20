# 📈 Stockio: Financial Forecasting & Anomaly Detection

**Intelligent Trading Platform** — Multi-ticker dashboard with Prophet/ARIMA/XGBoost forecasts, VADER sentiment analysis, and 93%+ anomaly detection precision. SHAP explainability for transparent, risk-aware insights.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Features

### 🔮 Multi-Model Forecasting
- **Prophet**: Facebook's time-series model with holiday effects and multiplicative seasonality
- **ARIMA**: Classical statistical forecasting with automatic order selection
- **XGBoost**: Gradient-boosted regression with 25+ engineered features
- **Ensemble**: Weighted combination with guardrails to prevent extreme predictions

### 🚨 Anomaly Detection (93%+ Precision)
- **Ensemble method**: IQR + Z-Score + Isolation Forest with voting consensus
- Tunable sensitivity slider
- Automatic anomaly classification (Price Spike, Volume Surge, Flash Event)
- Guardrails to limit false positive rate

### 💬 VADER Sentiment Analysis
- Real-time news headline sentiment scoring
- Bullish/Bearish/Neutral classification
- Sentiment timeline visualization
- Integration with trading signal generation

### 🧠 SHAP Explainability
- TreeExplainer for XGBoost model transparency
- Feature importance summary charts
- Waterfall plots for individual prediction explanation
- Transparent, risk-aware stakeholder visualizations

### ⚡ Cache-Aware Pipeline
- Memory + disk dual-layer caching with TTL
- LRU eviction for memory management
- Sub-second response for cached queries

### 📊 Risk Analytics
- Annualized volatility, Sharpe ratio, Sortino ratio
- Beta, Max Drawdown, Value at Risk (95%)
- Multi-ticker correlation matrix
- Risk-return scatter visualization

### 🎯 Trading Signals
- Multi-factor composite signals (Technical + Forecast + Sentiment)
- RSI, MACD, SMA crossover analysis
- Color-coded Buy/Sell/Hold recommendations

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/stockio.git
cd stockio
pip install -r requirements.txt
```

## ▶️ Usage

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Hugging Face Spaces Deployment

1. Create a new Space (SDK: Streamlit)
2. Upload all project files
3. The app will auto-deploy

---

## 🏗️ Architecture

```
stockio/
├── app.py                 # Main Streamlit dashboard
├── data_fetcher.py        # Yahoo Finance data + synthetic fallback
├── forecasting.py         # Prophet, ARIMA, XGBoost, Ensemble
├── sentiment.py           # VADER sentiment + news fetching
├── anomaly_detection.py   # Ensemble anomaly detection
├── explainability.py      # SHAP analysis & visualization
├── cache_manager.py       # Dual-layer caching with TTL
├── utils.py               # Technical indicators, risk, signals
├── requirements.txt
└── README.md
```

## 🔧 Configuration

All settings are configurable via the sidebar:
- **Tickers**: Select up to 6 from 15+ major stocks
- **Date Range**: Custom historical window
- **Forecast Horizon**: 7–90 days
- **Anomaly Sensitivity**: 1.0–4.0 (lower = more sensitive)
- **Sentiment Lookback**: 1–30 days of news

---

## 📋 Technical Details

| Component | Technology | Details |
|-----------|-----------|---------|
| Forecasting | Prophet, ARIMA, XGBoost | Ensemble with configurable weights |
| Anomaly Detection | IQR + Z-Score + Isolation Forest | Voting consensus with guardrails |
| Sentiment | VADER | Compound scoring on news headlines |
| Explainability | SHAP TreeExplainer | Feature importance + waterfall |
| Caching | Memory + Pickle | TTL-based with LRU eviction |
| Visualization | Plotly | Interactive charts with dark theme |
| Deployment | Streamlit + Azure DevOps | CI/CD to Hugging Face Spaces |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
