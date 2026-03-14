"""Capacity simulator estimating maximum AUM without self-impact.

For a given set of signals and target turnover, estimates the maximum
AUM the strategy can trade without significant self-impact.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.portfolio.capacity_model.market_impact_model import MarketImpactModel

logger = logging.getLogger(__name__)


class CapacitySimulator:
    """Estimates maximum strategy capacity before self-impact degrades alpha.

    Uses the market impact model to simulate how transaction costs scale
    with AUM, and finds the breakeven point where costs eat into alpha.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._max_impact_pct_of_alpha = cfg.get("max_impact_pct_of_alpha", 0.20)
        self._impact_model = MarketImpactModel(cfg)

    def estimate_capacity(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
        adv_usd: pd.Series,
        volatility: pd.Series,
        expected_alpha_bps: float,
        aum_grid: Optional[list] = None,
    ) -> dict:
        """Estimate strategy capacity by simulating impact at different AUM levels.

        Args:
            target_weights: Target portfolio weights.
            current_weights: Current portfolio weights.
            adv_usd: Average daily volume in USD per ticker.
            volatility: Daily volatility per ticker.
            expected_alpha_bps: Expected alpha in basis points.
            aum_grid: List of AUM values to test (USD).

        Returns:
            Dict with max_capacity_usd and impact-vs-aum table.
        """
        if aum_grid is None:
            aum_grid = [1e6, 5e6, 10e6, 25e6, 50e6, 100e6, 250e6, 500e6, 1e9]

        # Trade weights
        trade_weights = (target_weights - current_weights).abs()
        common = trade_weights.index.intersection(adv_usd.index).intersection(
            volatility.index
        )
        trade_weights = trade_weights.reindex(common, fill_value=0.0)
        adv = adv_usd.reindex(common, fill_value=1e6)
        vol = volatility.reindex(common, fill_value=0.02)

        results = []
        max_capacity = aum_grid[-1]

        for aum in aum_grid:
            trades_usd = trade_weights * aum
            total_cost = self._impact_model.estimate_total_cost_usd(
                trades_usd, adv, vol
            )
            cost_bps = (total_cost / aum * 10000) if aum > 0 else 0.0
            cost_pct_alpha = (cost_bps / expected_alpha_bps) if expected_alpha_bps > 0 else 0.0

            results.append({
                "aum": aum,
                "total_cost_usd": total_cost,
                "cost_bps": cost_bps,
                "cost_pct_of_alpha": cost_pct_alpha,
            })

            if cost_pct_alpha > self._max_impact_pct_of_alpha:
                max_capacity = aum
                break

        return {
            "max_capacity_usd": max_capacity,
            "impact_table": pd.DataFrame(results),
        }
