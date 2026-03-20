"""
utils.py
========
Technical indicators, risk metrics, trading signal generation,
and formatting utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV DataFrame."""
    if df is None or df.empty:
        return df

    close = df["Close"]

    # Simple Moving Averages
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()
    df["SMA_200"] = close.rolling(window=200).mean()

    # Exponential Moving Averages
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid

    # ATR (Average True Range)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - close.shift())
    low_close = np.abs(df["Low"] - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()

    # Stochastic Oscillator
    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # OBV (On-Balance Volume)
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    # VWAP (approximation)
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

    return df


def compute_risk_metrics(df: pd.DataFrame) -> dict:
    """Compute comprehensive risk metrics."""
    close = df["Close"]
    returns = close.pct_change().dropna()

    if len(returns) < 2:
        return {
            "volatility": 0, "sharpe": 0, "max_drawdown": 0,
            "beta": 1, "var_95": 0, "annual_return": 0,
        }

    # Annualized volatility
    volatility = returns.std() * np.sqrt(252)

    # Annualized return
    total_return = (close.iloc[-1] / close.iloc[0]) - 1
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Sharpe Ratio (assuming risk-free rate of 4.5%)
    risk_free = 0.045
    excess_return = annual_return - risk_free
    sharpe = excess_return / (volatility + 1e-10)

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Beta (vs market proxy - use returns autocorrelation as simplified proxy)
    # In production, you'd compare against SPY
    beta = 1.0 + (returns.autocorr(lag=1) or 0)
    beta = np.clip(beta, 0.3, 2.5)

    # Value at Risk (95%)
    var_95 = returns.quantile(0.05)

    # Sortino Ratio
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = excess_return / (downside + 1e-10)

    # Calmar Ratio
    calmar = annual_return / (abs(max_drawdown) + 1e-10)

    return {
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "beta": beta,
        "var_95": var_95,
        "annual_return": annual_return,
        "sortino": sortino,
        "calmar": calmar,
    }


def generate_trading_signals(
    df: pd.DataFrame,
    forecasts: dict,
    sentiment: Optional[dict],
) -> dict:
    """
    Generate multi-factor trading signals combining technical,
    forecast, and sentiment data.
    """
    signals = {}
    current_price = df["Close"].iloc[-1]

    # --- Technical Signal ---
    tech_score = 0
    if "RSI" in df.columns:
        rsi = df["RSI"].iloc[-1]
        if rsi < 30:
            tech_score += 2
        elif rsi < 40:
            tech_score += 1
        elif rsi > 70:
            tech_score -= 2
        elif rsi > 60:
            tech_score -= 1

    if "MACD_Hist" in df.columns:
        macd_h = df["MACD_Hist"].iloc[-1]
        macd_h_prev = df["MACD_Hist"].iloc[-2] if len(df) > 1 else 0
        if macd_h > 0 and macd_h_prev <= 0:
            tech_score += 2  # Bullish crossover
        elif macd_h < 0 and macd_h_prev >= 0:
            tech_score -= 2  # Bearish crossover

    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        if df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1]:
            tech_score += 1
        else:
            tech_score -= 1

    if tech_score >= 2:
        signals["Technical"] = "Strong Buy"
    elif tech_score >= 1:
        signals["Technical"] = "Buy"
    elif tech_score <= -2:
        signals["Technical"] = "Strong Sell"
    elif tech_score <= -1:
        signals["Technical"] = "Sell"
    else:
        signals["Technical"] = "Hold"

    # --- Forecast Signal ---
    fc_score = 0
    for name, fc in forecasts.items():
        if fc is not None and "values" in fc:
            fc_end = fc["values"][-1] if len(fc["values"]) > 0 else current_price
            change = (fc_end - current_price) / current_price
            if change > 0.05:
                fc_score += 2
            elif change > 0.02:
                fc_score += 1
            elif change < -0.05:
                fc_score -= 2
            elif change < -0.02:
                fc_score -= 1

    if forecasts:
        avg_fc = fc_score / len(forecasts)
        if avg_fc >= 1:
            signals["Forecast"] = "Buy"
        elif avg_fc <= -1:
            signals["Forecast"] = "Sell"
        else:
            signals["Forecast"] = "Hold"

    # --- Sentiment Signal ---
    if sentiment and "overall" in sentiment:
        compound = sentiment["overall"].get("compound", 0)
        if compound > 0.2:
            signals["Sentiment"] = "Bullish Buy"
        elif compound > 0.05:
            signals["Sentiment"] = "Mild Buy"
        elif compound < -0.2:
            signals["Sentiment"] = "Bearish Sell"
        elif compound < -0.05:
            signals["Sentiment"] = "Mild Sell"
        else:
            signals["Sentiment"] = "Neutral Hold"

    # --- Composite Signal ---
    buy_count = sum(1 for v in signals.values() if "Buy" in v)
    sell_count = sum(1 for v in signals.values() if "Sell" in v)

    if buy_count > sell_count + 1:
        signals["Composite"] = "Strong Buy ▲"
    elif buy_count > sell_count:
        signals["Composite"] = "Buy ▲"
    elif sell_count > buy_count + 1:
        signals["Composite"] = "Strong Sell ▼"
    elif sell_count > buy_count:
        signals["Composite"] = "Sell ▼"
    else:
        signals["Composite"] = "Hold ◆"

    return signals


def format_large_number(n) -> str:
    """Format large numbers with K/M/B suffixes."""
    try:
        n = float(n)
    except (ValueError, TypeError):
        return str(n)

    if abs(n) >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif abs(n) >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    else:
        return f"{n:.0f}"
