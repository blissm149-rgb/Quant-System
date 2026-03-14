"""Liquidity estimator for position sizing and capacity analysis.

Estimates per-stock liquidity metrics: average daily volume, effective
spread, and days to liquidate.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LiquidityEstimator:
    """Estimates per-stock liquidity for capacity and execution analysis."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._adv_window = cfg.get("adv_window", 20)
        self._min_adv_usd = cfg.get("min_adv_usd", 5_000_000)

    def estimate(
        self,
        volume: pd.DataFrame,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """Estimate liquidity metrics for all tickers.

        Args:
            volume: DataFrame (dates x tickers) of daily volume (shares).
            prices: DataFrame (dates x tickers) of daily close prices.
            as_of: Point-in-time boundary.

        Returns:
            DataFrame indexed by ticker with columns:
            adv_shares, adv_usd, days_to_liquidate_1pct
        """
        vol = volume[volume.index < as_of].iloc[-self._adv_window :]
        px = prices[prices.index < as_of].iloc[-self._adv_window :]

        adv_shares = vol.mean()
        latest_price = px.iloc[-1] if len(px) > 0 else pd.Series(dtype=float)
        adv_usd = adv_shares * latest_price

        # Days to liquidate 1% of portfolio (assuming $10M portfolio)
        portfolio_value = 10_000_000
        position_value = portfolio_value * 0.01
        participation_rate = 0.05
        daily_capacity = adv_usd * participation_rate
        days_to_liquidate = position_value / daily_capacity.where(
            daily_capacity > 0, np.inf
        )

        return pd.DataFrame({
            "adv_shares": adv_shares,
            "adv_usd": adv_usd,
            "latest_price": latest_price,
            "days_to_liquidate_1pct": days_to_liquidate,
        })

    def filter_liquid(
        self, liquidity: pd.DataFrame
    ) -> pd.Index:
        """Return tickers meeting minimum liquidity threshold."""
        return liquidity[liquidity["adv_usd"] >= self._min_adv_usd].index
