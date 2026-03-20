"""
data_fetcher.py
===============
Fetches stock data from Yahoo Finance with fallback synthetic generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker. Falls back to synthetic data if yfinance
    is unavailable or the download fails.
    """
    df = None

    if HAS_YFINANCE:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index)
                return df
        except Exception:
            pass

    # Fallback: generate realistic synthetic data
    return _generate_synthetic_data(ticker, start, end)


def _generate_synthetic_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for demo purposes."""
    np.random.seed(hash(ticker) % 2**31)

    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Simulate price with geometric Brownian motion
    base_prices = {
        "AAPL": 170, "GOOGL": 140, "MSFT": 380, "TSLA": 250,
        "AMZN": 180, "META": 500, "NVDA": 800, "JPM": 190,
        "NFLX": 600, "AMD": 160, "INTC": 35, "BA": 200,
        "DIS": 100, "V": 280, "WMT": 170,
    }
    base = base_prices.get(ticker, 100)
    mu = 0.0003  # daily drift
    sigma = 0.018  # daily volatility

    returns = np.random.normal(mu, sigma, n)
    # Add some regime changes for anomaly detection interest
    for _ in range(3):
        shock_idx = np.random.randint(n // 4, 3 * n // 4)
        returns[shock_idx] = np.random.choice([-1, 1]) * np.random.uniform(0.04, 0.08)

    prices = base * np.exp(np.cumsum(returns))

    # OHLC from close
    high = prices * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_prices = low + (high - low) * np.random.uniform(0.2, 0.8, n)

    # Volume
    base_vol = np.random.uniform(20e6, 80e6)
    volume = np.abs(np.random.normal(base_vol, base_vol * 0.3, n)).astype(int)
    # Volume spikes near price anomalies
    vol_spikes = np.where(np.abs(returns) > 0.03)[0]
    for idx in vol_spikes:
        volume[idx] = int(volume[idx] * np.random.uniform(2, 5))

    df = pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": volume,
    }, index=dates)

    return df


def get_company_info(ticker: str) -> dict:
    """Get basic company info."""
    if HAS_YFINANCE:
        try:
            info = yf.Ticker(ticker).info
            return {
                "name": info.get("shortName", ticker),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
            }
        except Exception:
            pass

    return {
        "name": ticker,
        "sector": "Technology",
        "industry": "N/A",
        "market_cap": 0,
        "pe_ratio": 0,
        "dividend_yield": 0,
    }
