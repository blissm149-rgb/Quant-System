"""Order book liquidity model.

Estimates depth, resilience, and available liquidity from
order book snapshots. Used by execution algorithms to size
child orders and by the fill probability model.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LiquiditySnapshot:
    """Liquidity metrics for a single ticker at a point in time."""

    ticker: str
    total_bid_depth: float  # total shares on bid side
    total_ask_depth: float  # total shares on ask side
    bid_depth_levels: int
    ask_depth_levels: int
    top_bid_size: float
    top_ask_size: float
    depth_imbalance: float  # (bid_depth - ask_depth) / (bid_depth + ask_depth)
    cost_to_trade_100k: float  # estimated cost in bps to trade $100k
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class OrderBookLevel:
    """A single price level in the order book."""

    price: float
    size: float


class OrderBookLiquidityModel:
    """Estimates available liquidity from order book data.

    Computes depth, imbalance, and cost-to-trade metrics from
    order book snapshots or summary statistics.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._default_levels = cfg.get("default_levels", 5)
        self._cost_trade_notional = cfg.get("cost_trade_notional", 100_000.0)

    def compute_depth_imbalance(
        self, bid_depth: float, ask_depth: float
    ) -> float:
        """Compute order book imbalance: (bid - ask) / (bid + ask).

        Positive means more buying interest; negative means more selling.
        """
        total = bid_depth + ask_depth
        if total <= 0:
            return 0.0
        return (bid_depth - ask_depth) / total

    def estimate_cost_to_trade(
        self,
        order_size: float,
        levels: List[OrderBookLevel],
        mid_price: float,
    ) -> float:
        """Estimate cost to trade a given size walking through book levels.

        Parameters
        ----------
        order_size : float
            Number of shares to trade.
        levels : list of OrderBookLevel
            Price levels sorted from best to worst.
        mid_price : float
            Current mid price for bps calculation.

        Returns
        -------
        float
            Estimated cost in basis points.
        """
        if mid_price <= 0 or order_size <= 0 or not levels:
            return 0.0

        remaining = order_size
        total_cost = 0.0

        for level in levels:
            fill_at_level = min(remaining, level.size)
            price_impact = abs(level.price - mid_price)
            total_cost += fill_at_level * price_impact
            remaining -= fill_at_level
            if remaining <= 0:
                break

        # If order exceeds available depth, penalize heavily
        if remaining > 0:
            last_price = levels[-1].price if levels else mid_price
            extra_impact = abs(last_price - mid_price) * 2.0
            total_cost += remaining * extra_impact

        avg_cost_per_share = total_cost / order_size
        return avg_cost_per_share / mid_price * 10000.0

    def compute_snapshot(
        self,
        ticker: str,
        bid_levels: List[OrderBookLevel],
        ask_levels: List[OrderBookLevel],
        mid_price: float,
    ) -> LiquiditySnapshot:
        """Compute a full liquidity snapshot from bid/ask levels."""
        bid_depth = sum(l.size for l in bid_levels)
        ask_depth = sum(l.size for l in ask_levels)

        shares_for_100k = self._cost_trade_notional / mid_price if mid_price > 0 else 0.0
        cost_bps = self.estimate_cost_to_trade(
            shares_for_100k, ask_levels, mid_price
        )

        return LiquiditySnapshot(
            ticker=ticker,
            total_bid_depth=bid_depth,
            total_ask_depth=ask_depth,
            bid_depth_levels=len(bid_levels),
            ask_depth_levels=len(ask_levels),
            top_bid_size=bid_levels[0].size if bid_levels else 0.0,
            top_ask_size=ask_levels[0].size if ask_levels else 0.0,
            depth_imbalance=self.compute_depth_imbalance(bid_depth, ask_depth),
            cost_to_trade_100k=cost_bps,
            timestamp=pd.Timestamp.now(),
        )

    def estimate_from_summary(
        self,
        ticker: str,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        adv: float,
    ) -> LiquiditySnapshot:
        """Estimate liquidity from top-of-book summary data.

        When full order book is unavailable, uses top-of-book
        to approximate depth and cost metrics.
        """
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
        spread = ask - bid if (bid > 0 and ask > 0) else 0.0

        # Approximate depth from ADV (assume top-of-book is ~1% of ADV)
        estimated_depth = adv * 0.01 if adv > 0 else 0.0

        depth_imbalance = self.compute_depth_imbalance(bid_size, ask_size)

        # Cost estimate: half spread + estimated market impact
        cost_bps = 0.0
        if mid > 0:
            half_spread_bps = spread / mid * 5000.0
            shares_100k = self._cost_trade_notional / mid
            impact_bps = 0.0
            if adv > 0:
                participation = shares_100k / adv
                impact_bps = participation ** 0.6 * 100.0
            cost_bps = half_spread_bps + impact_bps

        return LiquiditySnapshot(
            ticker=ticker,
            total_bid_depth=bid_size,
            total_ask_depth=ask_size,
            bid_depth_levels=1,
            ask_depth_levels=1,
            top_bid_size=bid_size,
            top_ask_size=ask_size,
            depth_imbalance=depth_imbalance,
            cost_to_trade_100k=cost_bps,
            timestamp=pd.Timestamp.now(),
        )
