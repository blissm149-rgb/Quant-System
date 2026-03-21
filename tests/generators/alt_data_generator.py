"""Section 10.2: Alternative Data Generators

Synthetic analyst estimates, news sentiment, and options chains
for testing alternative data modules.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def make_analyst_estimates(
    tickers: Optional[List[str]] = None,
    n_dates: int = 60,
    n_analysts: int = 5,
    seed: int = 42,
    revision_probability: float = 0.15,
) -> pd.DataFrame:
    """Generate synthetic analyst estimates with realistic revision patterns.

    Returns DataFrame with columns:
        date, ticker, analyst_id, eps_estimate, revenue_estimate, target_price, revision
    """
    tickers = tickers or ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    analysts = [f"analyst_{i}" for i in range(n_analysts)]

    rows = []
    # Initialize base estimates per (ticker, analyst)
    base_eps: Dict[str, Dict[str, float]] = {}
    for ticker in tickers:
        base_eps[ticker] = {}
        for analyst in analysts:
            base_eps[ticker][analyst] = rng.uniform(1.0, 10.0)

    for dt in dates:
        for ticker in tickers:
            for analyst in analysts:
                is_revision = rng.random() < revision_probability
                if is_revision:
                    # Revision: shift estimate by ±5-15%
                    direction = rng.choice([-1, 1])
                    magnitude = rng.uniform(0.05, 0.15)
                    base_eps[ticker][analyst] *= (1 + direction * magnitude)

                eps = base_eps[ticker][analyst] + rng.normal(0, 0.1)
                revenue = eps * rng.uniform(8, 15) * 1e9
                target = eps * rng.uniform(15, 30)

                rows.append({
                    "date": dt,
                    "ticker": ticker,
                    "analyst_id": analyst,
                    "eps_estimate": round(eps, 2),
                    "revenue_estimate": round(revenue, 0),
                    "target_price": round(target, 2),
                    "revision": is_revision,
                })

    return pd.DataFrame(rows)


def make_news_sentiment(
    tickers: Optional[List[str]] = None,
    n_dates: int = 60,
    articles_per_day: int = 5,
    signal_to_noise: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic news sentiment with configurable signal-to-noise ratio.

    Returns DataFrame with columns:
        date, ticker, headline, sentiment_score, confidence, source
    """
    tickers = tickers or ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    sources = ["Reuters", "Bloomberg", "CNBC", "WSJ", "FT"]

    # True underlying sentiment per ticker (signal)
    true_sentiment = {t: rng.uniform(-0.5, 0.5) for t in tickers}

    rows = []
    for dt in dates:
        for ticker in tickers:
            # Slowly drift underlying sentiment
            true_sentiment[ticker] += rng.normal(0, 0.02)
            true_sentiment[ticker] = np.clip(true_sentiment[ticker], -1, 1)

            n_articles = rng.poisson(articles_per_day)
            for _ in range(max(1, n_articles)):
                # Observed sentiment = signal + noise
                signal = true_sentiment[ticker] * signal_to_noise
                noise = rng.normal(0, 1 - signal_to_noise)
                score = np.clip(signal + noise, -1.0, 1.0)
                confidence = rng.uniform(0.4, 1.0)

                rows.append({
                    "date": dt,
                    "ticker": ticker,
                    "headline": f"{ticker} news article",
                    "sentiment_score": round(score, 4),
                    "confidence": round(confidence, 3),
                    "source": rng.choice(sources),
                })

    return pd.DataFrame(rows)


def make_options_chain(
    ticker: str = "AAPL",
    spot_price: float = 150.0,
    n_strikes: int = 20,
    n_expiries: int = 5,
    risk_free_rate: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic options chain with valid no-arbitrage IV surface.

    Returns DataFrame with columns:
        ticker, expiry, strike, option_type, bid, ask, mid, implied_vol,
        delta, gamma, open_interest, volume
    """
    rng = np.random.default_rng(seed)

    # Expiries: 7d, 30d, 60d, 90d, 180d
    expiry_days = [7, 30, 60, 90, 180][:n_expiries]
    base_date = pd.Timestamp("2023-06-15")

    # Strike range: 80% to 120% of spot
    strikes = np.linspace(spot_price * 0.80, spot_price * 1.20, n_strikes)

    # Base ATM vol
    atm_vol = 0.25

    rows = []
    for days_to_exp in expiry_days:
        expiry = base_date + pd.Timedelta(days=days_to_exp)
        T = days_to_exp / 365.0

        for strike in strikes:
            moneyness = np.log(strike / spot_price)

            # IV smile: higher for OTM, lower for ATM
            # Skew: slightly higher for low strikes (put skew)
            vol_smile = atm_vol + 0.10 * moneyness**2 - 0.05 * moneyness
            vol_smile += rng.normal(0, 0.005)  # small noise
            vol_smile = max(vol_smile, 0.05)  # floor

            # Term structure: longer expiry → slightly higher vol
            vol_term = vol_smile * (1 + 0.02 * np.sqrt(T))

            for opt_type in ["call", "put"]:
                # Rough B-S approximation for mid price
                d1 = (np.log(spot_price / strike) + (risk_free_rate + 0.5 * vol_term**2) * T) / (vol_term * np.sqrt(T + 1e-10))
                from scipy.stats import norm
                if opt_type == "call":
                    price = spot_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * T) * norm.cdf(d1 - vol_term * np.sqrt(T))
                    delta = norm.cdf(d1)
                else:
                    price = strike * np.exp(-risk_free_rate * T) * norm.cdf(-(d1 - vol_term * np.sqrt(T))) - spot_price * norm.cdf(-d1)
                    delta = norm.cdf(d1) - 1

                price = max(price, 0.01)
                spread = price * rng.uniform(0.02, 0.10)
                gamma = norm.pdf(d1) / (spot_price * vol_term * np.sqrt(T + 1e-10))

                rows.append({
                    "ticker": ticker,
                    "expiry": expiry,
                    "strike": round(strike, 2),
                    "option_type": opt_type,
                    "bid": round(price - spread / 2, 2),
                    "ask": round(price + spread / 2, 2),
                    "mid": round(price, 2),
                    "implied_vol": round(vol_term, 4),
                    "delta": round(delta, 4),
                    "gamma": round(gamma, 6),
                    "open_interest": int(rng.uniform(100, 10000)),
                    "volume": int(rng.uniform(0, 5000)),
                })

    return pd.DataFrame(rows)
