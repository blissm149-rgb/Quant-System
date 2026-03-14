"""Market impact model using Almgren-Chriss framework.

Estimates expected market impact in basis points as a function of
order size / ADV and daily volatility.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketImpactModel:
    """Almgren-Chriss market impact model.

    impact_bps = sigma * (order_size / ADV) ^ power_law_exponent

    Where sigma is daily volatility and the exponent is empirically calibrated.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._power_law_exponent = cfg.get("impact_exponent", 0.6)
        self._temporary_impact_coeff = cfg.get("temporary_impact_coeff", 1.0)
        self._permanent_impact_coeff = cfg.get("permanent_impact_coeff", 0.1)

    def estimate_impact_bps(
        self,
        order_size_usd: float,
        adv_usd: float,
        daily_volatility: float,
    ) -> float:
        """Estimate market impact in basis points for a single order.

        Args:
            order_size_usd: Order notional in USD.
            adv_usd: Average daily volume in USD.
            daily_volatility: Daily return standard deviation.

        Returns:
            Estimated impact in basis points.
        """
        if adv_usd <= 0 or order_size_usd <= 0:
            return 0.0

        participation = order_size_usd / adv_usd
        temporary = (
            self._temporary_impact_coeff
            * daily_volatility
            * participation ** self._power_law_exponent
        )
        permanent = (
            self._permanent_impact_coeff
            * daily_volatility
            * participation
        )
        total_impact = temporary + permanent
        return total_impact * 10000  # Convert to bps

    def estimate_impact_portfolio(
        self,
        trades: pd.Series,
        adv_usd: pd.Series,
        volatility: pd.Series,
    ) -> pd.Series:
        """Estimate impact for a portfolio of trades.

        Args:
            trades: Absolute trade notional in USD per ticker.
            adv_usd: Average daily volume in USD per ticker.
            volatility: Daily volatility per ticker.

        Returns:
            Series of impact estimates in bps per ticker.
        """
        common = trades.index.intersection(adv_usd.index).intersection(
            volatility.index
        )
        impacts = {}
        for ticker in common:
            impacts[ticker] = self.estimate_impact_bps(
                trades[ticker], adv_usd[ticker], volatility[ticker]
            )
        return pd.Series(impacts)

    def estimate_total_cost_usd(
        self,
        trades: pd.Series,
        adv_usd: pd.Series,
        volatility: pd.Series,
    ) -> float:
        """Estimate total transaction cost in USD for a set of trades.

        Returns:
            Total estimated cost in USD.
        """
        impacts_bps = self.estimate_impact_portfolio(trades, adv_usd, volatility)
        costs = trades.reindex(impacts_bps.index, fill_value=0.0) * impacts_bps / 10000
        return float(costs.sum())
