"""
sentiment.py
============
VADER sentiment analysis on financial news headlines.
Includes synthetic news generation for demo when APIs are unavailable.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of a single text string using VADER."""
    if HAS_VADER:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return scores
    else:
        # Simple keyword-based fallback
        positive = ["up", "gain", "surge", "rally", "bull", "profit", "beat", "growth",
                     "strong", "positive", "high", "record", "outperform", "upgrade"]
        negative = ["down", "loss", "drop", "crash", "bear", "decline", "miss", "weak",
                     "negative", "low", "fear", "sell", "downgrade", "cut", "risk"]

        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        total = pos_count + neg_count + 1

        pos_score = pos_count / total
        neg_score = neg_count / total
        compound = (pos_score - neg_score)

        return {
            "pos": pos_score,
            "neg": neg_score,
            "neu": 1 - pos_score - neg_score,
            "compound": compound,
        }


def get_news_sentiment(ticker: str, lookback_days: int = 7) -> dict:
    """
    Fetch news headlines and compute sentiment.
    Tries RSS feeds first, falls back to synthetic headlines for demo.
    """
    headlines = []

    # Try Google News RSS
    if HAS_FEEDPARSER:
        try:
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.now()
                if (datetime.now() - pub_date).days <= lookback_days:
                    scores = analyze_sentiment(entry.title)
                    headlines.append({
                        "title": entry.title,
                        "date": pub_date.strftime("%Y-%m-%d"),
                        **scores,
                    })
        except Exception:
            pass

    # Fallback: generate synthetic headlines
    if not headlines:
        headlines = _generate_synthetic_headlines(ticker, lookback_days)

    if not headlines:
        return None

    # Compute overall sentiment
    compounds = [h["compound"] for h in headlines]
    overall = {
        "compound": np.mean(compounds),
        "pos": np.mean([h.get("pos", 0) for h in headlines]),
        "neg": np.mean([h.get("neg", 0) for h in headlines]),
        "neu": np.mean([h.get("neu", 0) for h in headlines]),
    }

    # Build timeline
    df = pd.DataFrame(headlines)
    if "date" in df.columns:
        timeline = df.groupby("date").agg({"compound": "mean"}).reset_index().to_dict("records")
    else:
        timeline = []

    return {
        "overall": overall,
        "headlines": headlines,
        "timeline": timeline,
        "count": len(headlines),
    }


def _generate_synthetic_headlines(ticker: str, lookback_days: int) -> list:
    """Generate realistic synthetic financial headlines for demo."""
    np.random.seed(hash(ticker) % 2**31)

    templates_positive = [
        f"{ticker} shares surge on strong earnings beat",
        f"Analysts upgrade {ticker} with bullish price target",
        f"{ticker} reports record quarterly revenue growth",
        f"Institutional investors increase {ticker} holdings",
        f"{ticker} announces expansion into new markets",
        f"Strong demand drives {ticker} stock to new highs",
        f"{ticker} beats analyst expectations for Q4 results",
        f"Positive momentum continues for {ticker} amid sector rally",
        f"{ticker} secures major partnership deal worth billions",
        f"Market optimism grows as {ticker} unveils innovation roadmap",
    ]

    templates_negative = [
        f"{ticker} falls after disappointing guidance outlook",
        f"Concerns mount over {ticker} valuation amid market uncertainty",
        f"{ticker} faces headwinds from rising interest rates",
        f"Analysts downgrade {ticker} citing competitive pressures",
        f"{ticker} misses revenue estimates in latest quarter",
        f"Regulatory scrutiny weighs on {ticker} stock performance",
        f"{ticker} warns of supply chain disruptions ahead",
        f"Insider selling raises red flags for {ticker} investors",
    ]

    templates_neutral = [
        f"{ticker} trading sideways as market awaits Fed decision",
        f"Volume remains steady for {ticker} in mixed session",
        f"{ticker} holds steady amid broader market fluctuations",
        f"Investors weigh mixed signals from {ticker} earnings call",
        f"{ticker} maintains dividend as board reviews strategy",
    ]

    headlines = []
    for day_offset in range(lookback_days):
        date = (datetime.now() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        n_articles = np.random.randint(1, 4)

        for _ in range(n_articles):
            r = np.random.random()
            if r < 0.4:
                title = np.random.choice(templates_positive)
            elif r < 0.7:
                title = np.random.choice(templates_neutral)
            else:
                title = np.random.choice(templates_negative)

            scores = analyze_sentiment(title)
            headlines.append({"title": title, "date": date, **scores})

    return headlines
