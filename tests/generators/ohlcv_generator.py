"""Enhanced OHLCV data generator with regime support.

Extends the base make_ohlcv() from conftest with market regime simulation,
configurable volatility regimes, and multi-regime sequences.
"""

import numpy as np
import pandas as pd


class MarketRegimeSimulator:
    """Generates synthetic market data matching specific regimes."""

    def __init__(self, tickers=None, seed=42):
        self.tickers = tickers or ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        self.rng = np.random.default_rng(seed)

    def bull_market(self, days=60, annual_return=0.15, vol=0.12):
        """Low vol, positive drift."""
        daily_drift = annual_return / 252
        daily_vol = vol / np.sqrt(252)
        return self._generate(days, daily_drift, daily_vol)

    def bear_market(self, days=30, drawdown=0.30, vol=0.35):
        """High vol, negative drift, fat tails."""
        daily_drift = np.log(1 - drawdown) / days
        daily_vol = vol / np.sqrt(252)
        return self._generate(days, daily_drift, daily_vol, fat_tails=True)

    def sideways(self, days=90, vol=0.08):
        """Very low vol, mean-reverting."""
        daily_vol = vol / np.sqrt(252)
        return self._generate(days, 0.0, daily_vol)

    def crash(self, days=5, drawdown=0.15, vol=0.80):
        """Extreme vol, sharp decline."""
        daily_drift = np.log(1 - drawdown) / days
        daily_vol = vol / np.sqrt(252)
        return self._generate(days, daily_drift, daily_vol, fat_tails=True)

    def recovery(self, days=60, annual_return=0.25, vol=0.20):
        """Post-crash recovery with elevated vol."""
        daily_drift = annual_return / 252
        daily_vol = vol / np.sqrt(252)
        return self._generate(days, daily_drift, daily_vol)

    def regime_sequence(self, regimes):
        """Chain multiple regimes together.

        Parameters
        ----------
        regimes : list of (name, kwargs) tuples
            Each tuple is (regime_method_name, kwargs_dict).
            e.g. [("bull_market", {"days": 60}), ("crash", {"days": 5})]

        Returns
        -------
        pd.DataFrame
            MultiIndex(date, ticker) OHLCV data spanning all regimes.
        """
        frames = []
        current_date = pd.Timestamp("2020-01-02")
        last_prices = {t: self.rng.uniform(50, 200) for t in self.tickers}

        for regime_name, kwargs in regimes:
            method = getattr(self, regime_name)
            df = method(**kwargs)
            days = len(df.index.get_level_values("date").unique())

            # Rebase dates to continue from current_date
            dates_unique = pd.bdate_range(start=current_date, periods=days)
            new_rows = []
            for ticker in self.tickers:
                if ticker not in df.index.get_level_values("ticker"):
                    continue
                ticker_data = df.xs(ticker, level="ticker")
                scale = last_prices[ticker] / ticker_data["close"].iloc[0]
                for i, (_, row) in enumerate(ticker_data.iterrows()):
                    if i >= len(dates_unique):
                        break
                    new_rows.append({
                        "date": dates_unique[i],
                        "ticker": ticker,
                        "open": row["open"] * scale,
                        "high": row["high"] * scale,
                        "low": row["low"] * scale,
                        "close": row["close"] * scale,
                        "adj_close": row["close"] * scale,
                        "volume": int(row["volume"]),
                    })
                if new_rows:
                    last_prices[ticker] = new_rows[-1]["close"]

            if new_rows:
                frames.append(pd.DataFrame(new_rows))
            current_date = dates_unique[-1] + pd.offsets.BDay(1)

        result = pd.concat(frames, ignore_index=True)
        return result.set_index(["date", "ticker"]).sort_index()

    def _generate(self, days, drift, vol, fat_tails=False):
        """Generate OHLCV for all tickers with given parameters."""
        dates = pd.bdate_range("2020-01-02", periods=days)
        rows = []
        for ticker in self.tickers:
            base = self.rng.uniform(50, 200)
            if fat_tails:
                rets = self.rng.standard_t(df=3, size=days) * vol + drift
            else:
                rets = self.rng.normal(drift, vol, size=days)
            prices = base * np.cumprod(1 + rets)
            prices = np.maximum(prices, 0.01)

            for i, dt in enumerate(dates):
                p = prices[i]
                intraday = abs(self.rng.normal(0, vol * 0.7))
                rows.append({
                    "date": dt,
                    "ticker": ticker,
                    "open": p * (1 + self.rng.uniform(-0.005, 0.005)),
                    "high": p * (1 + intraday),
                    "low": p * (1 - intraday),
                    "close": p,
                    "adj_close": p,
                    "volume": int(self.rng.uniform(1e6, 1e8)),
                })
        df = pd.DataFrame(rows)
        return df.set_index(["date", "ticker"]).sort_index()


def make_bull_market(tickers=None, days=60, seed=42, **kwargs):
    """Convenience function for bull market data."""
    sim = MarketRegimeSimulator(tickers=tickers, seed=seed)
    return sim.bull_market(days=days, **kwargs)


def make_bear_market(tickers=None, days=30, seed=42, **kwargs):
    """Convenience function for bear market data."""
    sim = MarketRegimeSimulator(tickers=tickers, seed=seed)
    return sim.bear_market(days=days, **kwargs)


def make_crash(tickers=None, days=5, seed=42, **kwargs):
    """Convenience function for crash scenario."""
    sim = MarketRegimeSimulator(tickers=tickers, seed=seed)
    return sim.crash(days=days, **kwargs)


def make_recovery(tickers=None, days=60, seed=42, **kwargs):
    """Convenience function for recovery scenario."""
    sim = MarketRegimeSimulator(tickers=tickers, seed=seed)
    return sim.recovery(days=days, **kwargs)
