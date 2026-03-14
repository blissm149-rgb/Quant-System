"""Liquidity-seeking execution algorithm.

Adapts order placement to current market conditions: posts
passive limit orders when spreads are wide and crosses when
spreads tighten. Monitors fill rates and adjusts urgency.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderType,
)

logger = logging.getLogger(__name__)


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LiquidityState:
    """Current liquidity conditions for decision-making."""

    ticker: str
    mid_price: float
    spread_bps: float
    available_depth: float
    adv: float
    urgency: Urgency = Urgency.MEDIUM
    elapsed_pct: float = 0.0  # fraction of time window elapsed
    filled_pct: float = 0.0  # fraction of order filled


class LiquiditySeekingExecution:
    """Liquidity-seeking execution algorithm.

    Adapts between passive and aggressive orders based on:
    - Current spread width relative to historical average
    - Fill rate vs. expected schedule
    - Time remaining in execution window
    - Order book depth
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._spread_threshold_bps = cfg.get("spread_threshold_bps", 5.0)
        self._urgency_ramp_pct = cfg.get("urgency_ramp_pct", 0.8)
        self._max_participation = cfg.get("max_participation", 0.10)
        self._passive_offset_bps = cfg.get("passive_offset_bps", 2.0)

    def compute_urgency(
        self,
        elapsed_pct: float,
        filled_pct: float,
    ) -> Urgency:
        """Compute execution urgency based on progress vs. time.

        Urgency increases when fill rate lags the elapsed time.
        """
        if elapsed_pct <= 0:
            return Urgency.LOW

        shortfall = elapsed_pct - filled_pct
        if shortfall > 0.3 and elapsed_pct > self._urgency_ramp_pct:
            return Urgency.CRITICAL
        if shortfall > 0.2 or elapsed_pct > self._urgency_ramp_pct:
            return Urgency.HIGH
        if shortfall > 0.1:
            return Urgency.MEDIUM
        return Urgency.LOW

    def decide_order_type(
        self,
        state: LiquidityState,
    ) -> OrderType:
        """Decide between market and limit order based on conditions."""
        if state.urgency == Urgency.CRITICAL:
            return OrderType.MARKET
        if state.urgency == Urgency.HIGH and state.spread_bps <= self._spread_threshold_bps:
            return OrderType.MARKET
        return OrderType.LIMIT

    def compute_limit_price(
        self,
        side: OrderSide,
        mid_price: float,
        spread_bps: float,
        urgency: Urgency,
    ) -> float:
        """Compute limit price based on urgency and spread.

        More urgent → price closer to or crossing the spread.
        Less urgent → passive price behind mid.
        """
        if mid_price <= 0:
            return 0.0

        urgency_offsets = {
            Urgency.LOW: self._passive_offset_bps,
            Urgency.MEDIUM: 0.0,
            Urgency.HIGH: -spread_bps / 4.0,
            Urgency.CRITICAL: -spread_bps / 2.0,
        }
        offset_bps = urgency_offsets.get(urgency, 0.0)

        if side == OrderSide.BUY:
            # Positive offset → lower price (passive)
            limit = mid_price * (1.0 - offset_bps / 10000.0)
        else:
            # Positive offset → higher price (passive)
            limit = mid_price * (1.0 + offset_bps / 10000.0)

        return round(limit, 2)

    def compute_child_size(
        self,
        remaining_shares: int,
        state: LiquidityState,
        num_remaining_slices: int = 1,
    ) -> int:
        """Compute child order size based on conditions.

        Considers available depth, participation limits, and urgency.
        """
        if remaining_shares <= 0 or num_remaining_slices <= 0:
            return 0

        # Base size: spread evenly over remaining slices
        base_size = remaining_shares // num_remaining_slices

        # Cap at participation rate of ADV
        if state.adv > 0:
            max_participation = int(state.adv * self._max_participation)
            base_size = min(base_size, max_participation)

        # Cap at available depth
        if state.available_depth > 0:
            depth_cap = int(state.available_depth * 0.5)
            base_size = min(base_size, depth_cap)

        # Urgency boost
        if state.urgency == Urgency.CRITICAL:
            base_size = int(base_size * 1.5)
        elif state.urgency == Urgency.HIGH:
            base_size = int(base_size * 1.2)

        return max(1, min(base_size, remaining_shares))

    def generate_child_order(
        self,
        ticker: str,
        side: OrderSide,
        remaining_shares: int,
        state: LiquidityState,
        num_remaining_slices: int = 1,
    ) -> Optional[Order]:
        """Generate the next child order adapting to current conditions.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        side : OrderSide
            BUY or SELL.
        remaining_shares : int
            Shares remaining to execute.
        state : LiquidityState
            Current market/liquidity conditions.
        num_remaining_slices : int
            Number of time slices remaining.

        Returns
        -------
        Order or None if nothing to do.
        """
        if remaining_shares <= 0:
            return None

        size = self.compute_child_size(
            remaining_shares, state, num_remaining_slices
        )
        order_type = self.decide_order_type(state)

        limit_price = None
        if order_type == OrderType.LIMIT:
            limit_price = self.compute_limit_price(
                side, state.mid_price, state.spread_bps, state.urgency
            )

        return Order(
            ticker=ticker,
            side=side,
            quantity=size,
            order_type=order_type,
            limit_price=limit_price,
        )
