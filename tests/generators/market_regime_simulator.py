"""Section 11.2: Market Regime Simulator

Generates synthetic market data matching specific regimes for E2E testing.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MarketRegimeSimulator:
    """Generates synthetic market data matching specific regimes."""

    def __init__(self, tickers: Optional[List[str]] = None, seed: int = 42):
        self.tickers = tickers or ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _generate_ohlcv(
        self,
        days: int,
        drift: float,
        vol: float,
        start_date: str = "2020-01-02",
        base_prices: Optional[Dict[str, float]] = None,
        fat_tails: bool = False,
    ) -> pd.DataFrame:
        """Generate OHLCV data with given drift and volatility.

        Args:
            days: Number of trading days to simulate.
            drift: Daily expected return (e.g. 0.001 for 0.1%/day).
            vol: Daily volatility (e.g. 0.02 for 2%/day).
            start_date: Start date string.
            base_prices: Optional starting prices per ticker.
            fat_tails: If True, use t-distribution for fatter tails.
        """
        dates = pd.bdate_range(start_date, periods=days)
        base_prices = base_prices or {t: self._rng.uniform(50, 300) for t in self.tickers}

        rows = []
        for ticker in self.tickers:
            price = base_prices[ticker]
            for dt in dates:
                if fat_tails:
                    # t-distribution with df=4 for fat tails
                    ret = drift + vol * (self._rng.standard_t(4))
                else:
                    ret = self._rng.normal(drift, vol)

                price *= (1 + ret)
                price = max(price, 0.01)  # floor at 1 cent

                intraday_range = abs(self._rng.normal(0, vol)) * price
                high = price + intraday_range / 2
                low = max(price - intraday_range / 2, 0.01)
                open_price = price + self._rng.normal(0, vol * 0.3) * price

                rows.append({
                    "date": dt,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": price,
                    "adj_close": price,
                    "volume": int(self._rng.uniform(1e6, 1e8)),
                })

        df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
        return df

    def bull_market(
        self,
        days: int = 60,
        annual_return: float = 0.15,
        vol: float = 0.12,
        **kwargs,
    ) -> pd.DataFrame:
        """Low vol, positive drift bull market."""
        daily_drift = annual_return / 252
        daily_vol = vol / np.sqrt(252)
        return self._generate_ohlcv(days, daily_drift, daily_vol, **kwargs)

    def bear_market(
        self,
        days: int = 30,
        drawdown: float = 0.30,
        vol: float = 0.35,
        **kwargs,
    ) -> pd.DataFrame:
        """High vol, negative drift bear market with fat tails."""
        # Target total drawdown over the period
        daily_drift = np.log(1 - drawdown) / days
        daily_vol = vol / np.sqrt(252)
        return self._generate_ohlcv(days, daily_drift, daily_vol, fat_tails=True, **kwargs)

    def sideways(
        self,
        days: int = 90,
        vol: float = 0.08,
        **kwargs,
    ) -> pd.DataFrame:
        """Very low vol, mean-reverting sideways market."""
        daily_vol = vol / np.sqrt(252)
        return self._generate_ohlcv(days, 0.0, daily_vol, **kwargs)

    def crash(
        self,
        days: int = 5,
        drawdown: float = 0.15,
        vol: float = 0.80,
        **kwargs,
    ) -> pd.DataFrame:
        """Extreme vol, sharp decline flash crash."""
        daily_drift = np.log(1 - drawdown) / days
        daily_vol = vol / np.sqrt(252)
        return self._generate_ohlcv(days, daily_drift, daily_vol, fat_tails=True, **kwargs)

    def recovery(
        self,
        days: int = 60,
        recovery_pct: float = 0.20,
        vol: float = 0.20,
        **kwargs,
    ) -> pd.DataFrame:
        """Post-crash recovery with elevated vol."""
        daily_drift = np.log(1 + recovery_pct) / days
        daily_vol = vol / np.sqrt(252)
        return self._generate_ohlcv(days, daily_drift, daily_vol, **kwargs)

    def regime_sequence(
        self,
        regimes: List[Tuple[str, Dict]],
        start_date: str = "2020-01-02",
    ) -> pd.DataFrame:
        """Chain multiple regimes together into a continuous dataset.

        Args:
            regimes: List of (regime_name, kwargs) tuples.
                     regime_name is one of: "bull", "bear", "sideways", "crash", "recovery"
            start_date: Starting date for the sequence.

        Returns:
            Combined OHLCV DataFrame with continuous price series.
        """
        regime_methods = {
            "bull": self.bull_market,
            "bear": self.bear_market,
            "sideways": self.sideways,
            "crash": self.crash,
            "recovery": self.recovery,
        }

        all_dfs = []
        current_date = pd.Timestamp(start_date)
        base_prices = {t: self._rng.uniform(50, 300) for t in self.tickers}

        for regime_name, kwargs in regimes:
            method = regime_methods[regime_name]
            df = method(
                start_date=current_date.strftime("%Y-%m-%d"),
                base_prices=base_prices,
                **kwargs,
            )
            all_dfs.append(df)

            # Update base prices to last close and advance date
            dates = df.index.get_level_values("date").unique()
            current_date = dates[-1] + pd.Timedelta(days=1)
            for ticker in self.tickers:
                if (dates[-1], ticker) in df.index:
                    base_prices[ticker] = df.loc[(dates[-1], ticker), "close"]

        return pd.concat(all_dfs).sort_index()
