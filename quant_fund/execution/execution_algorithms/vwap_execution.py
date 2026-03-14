"""VWAP execution algorithm.

Slices a parent order into child orders following an intraday
volume profile to achieve Volume-Weighted Average Price.
Default execution algorithm per execution_config.yaml.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderType,
)

logger = logging.getLogger(__name__)

# Default intraday volume profile (30-min buckets, 13 buckets for 9:30-16:00)
DEFAULT_VOLUME_PROFILE = np.array([
    0.12, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06,
    0.06, 0.06, 0.06, 0.07, 0.09, 0.15,
])
DEFAULT_VOLUME_PROFILE = DEFAULT_VOLUME_PROFILE / DEFAULT_VOLUME_PROFILE.sum()


@dataclass
class VWAPSlice:
    """A single child order slice in the VWAP schedule."""

    slice_index: int
    target_shares: int
    filled_shares: int = 0
    target_start: Optional[pd.Timestamp] = None
    target_end: Optional[pd.Timestamp] = None
    is_complete: bool = False


@dataclass
class VWAPPlan:
    """Full VWAP execution plan for a parent order."""

    ticker: str
    side: OrderSide
    total_shares: int
    slices: List[VWAPSlice] = field(default_factory=list)
    participation_rate: float = 0.05
    filled_shares: int = 0

    @property
    def remaining_shares(self) -> int:
        return self.total_shares - self.filled_shares

    @property
    def pct_complete(self) -> float:
        if self.total_shares <= 0:
            return 1.0
        return self.filled_shares / self.total_shares

    @property
    def is_complete(self) -> bool:
        return self.filled_shares >= self.total_shares


class VWAPExecution:
    """VWAP execution algorithm.

    Distributes a parent order across time intervals following
    a historical intraday volume profile, capped at a
    maximum participation rate of ADV per interval.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._participation_rate = cfg.get("participation_rate", 0.05)
        self._max_slice_pct = cfg.get("max_slice_pct", 0.20)
        self._num_slices = cfg.get("num_slices", 13)
        self._volume_profile = np.array(
            cfg.get("volume_profile", DEFAULT_VOLUME_PROFILE.tolist())
        )
        if len(self._volume_profile) != self._num_slices:
            self._volume_profile = DEFAULT_VOLUME_PROFILE
        self._volume_profile = self._volume_profile / self._volume_profile.sum()

    def create_plan(
        self,
        ticker: str,
        side: OrderSide,
        total_shares: int,
        adv: float = 1e6,
        start_time: Optional[pd.Timestamp] = None,
    ) -> VWAPPlan:
        """Create a VWAP execution plan.

        Parameters
        ----------
        ticker : str
            Ticker to trade.
        side : OrderSide
            BUY or SELL.
        total_shares : int
            Total shares to execute.
        adv : float
            Average daily volume for participation rate capping.
        start_time : pd.Timestamp, optional
            Start time for the plan.

        Returns
        -------
        VWAPPlan
        """
        plan = VWAPPlan(
            ticker=ticker,
            side=side,
            total_shares=total_shares,
            participation_rate=self._participation_rate,
        )

        # Cap total order at participation rate of ADV
        max_order = int(adv * self._participation_rate)
        effective_shares = min(total_shares, max_order) if max_order > 0 else total_shares

        # Distribute across slices following volume profile
        allocated = 0
        slices = []
        for i in range(self._num_slices):
            target = int(effective_shares * self._volume_profile[i])
            # Cap individual slice
            max_slice = int(total_shares * self._max_slice_pct)
            target = min(target, max_slice)
            target = max(target, 0)
            allocated += target

            slices.append(VWAPSlice(
                slice_index=i,
                target_shares=target,
            ))

        # Distribute any remainder to last slice
        remainder = effective_shares - allocated
        if remainder > 0 and slices:
            slices[-1].target_shares += remainder

        plan.slices = slices
        return plan

    def get_next_child_order(
        self,
        plan: VWAPPlan,
        current_price: float,
    ) -> Optional[Order]:
        """Get the next child order to send from the plan.

        Returns the first incomplete slice as a market order,
        or None if the plan is complete.
        """
        if plan.is_complete:
            return None

        for s in plan.slices:
            if not s.is_complete and s.target_shares > s.filled_shares:
                remaining = s.target_shares - s.filled_shares
                return Order(
                    ticker=plan.ticker,
                    side=plan.side,
                    quantity=remaining,
                    order_type=OrderType.MARKET,
                )
        return None

    def record_fill(
        self,
        plan: VWAPPlan,
        filled_shares: int,
    ) -> None:
        """Record a fill against the plan."""
        remaining = filled_shares
        for s in plan.slices:
            if s.is_complete:
                continue
            can_fill = s.target_shares - s.filled_shares
            fill_here = min(remaining, can_fill)
            s.filled_shares += fill_here
            if s.filled_shares >= s.target_shares:
                s.is_complete = True
            remaining -= fill_here
            if remaining <= 0:
                break
        plan.filled_shares += filled_shares
