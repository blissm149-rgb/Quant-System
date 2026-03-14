"""TWAP execution algorithm.

Slices a parent order into equal-sized child orders spread
evenly across a time window. Simpler than VWAP; used when
volume profile data is unavailable or for less liquid names.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderType,
)

logger = logging.getLogger(__name__)


@dataclass
class TWAPSlice:
    """A single child order in the TWAP schedule."""

    slice_index: int
    target_shares: int
    filled_shares: int = 0
    scheduled_time: Optional[pd.Timestamp] = None
    is_complete: bool = False


@dataclass
class TWAPPlan:
    """Full TWAP execution plan."""

    ticker: str
    side: OrderSide
    total_shares: int
    num_slices: int
    interval_seconds: float
    slices: List[TWAPSlice] = field(default_factory=list)
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


class TWAPExecution:
    """TWAP execution algorithm.

    Distributes a parent order into equal-sized child orders
    sent at regular intervals over a specified time window.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._default_num_slices = cfg.get("num_slices", 10)
        self._default_window_minutes = cfg.get("window_minutes", 60)
        self._participation_rate = cfg.get("participation_rate", 0.05)
        self._randomize_pct = cfg.get("randomize_pct", 0.10)

    def create_plan(
        self,
        ticker: str,
        side: OrderSide,
        total_shares: int,
        num_slices: Optional[int] = None,
        window_minutes: Optional[float] = None,
        start_time: Optional[pd.Timestamp] = None,
    ) -> TWAPPlan:
        """Create a TWAP execution plan.

        Parameters
        ----------
        ticker : str
            Ticker to trade.
        side : OrderSide
            BUY or SELL.
        total_shares : int
            Total shares to execute.
        num_slices : int, optional
            Number of slices. Defaults to configured value.
        window_minutes : float, optional
            Total execution window in minutes.
        start_time : pd.Timestamp, optional
            Start time for scheduling.

        Returns
        -------
        TWAPPlan
        """
        n_slices = num_slices or self._default_num_slices
        window_min = window_minutes or self._default_window_minutes
        interval_s = (window_min * 60.0) / n_slices

        base_per_slice = total_shares // n_slices
        remainder = total_shares - base_per_slice * n_slices

        slices = []
        for i in range(n_slices):
            target = base_per_slice + (1 if i < remainder else 0)
            scheduled = None
            if start_time is not None:
                scheduled = start_time + pd.Timedelta(seconds=interval_s * i)
            slices.append(TWAPSlice(
                slice_index=i,
                target_shares=target,
                scheduled_time=scheduled,
            ))

        return TWAPPlan(
            ticker=ticker,
            side=side,
            total_shares=total_shares,
            num_slices=n_slices,
            interval_seconds=interval_s,
            slices=slices,
        )

    def get_next_child_order(
        self,
        plan: TWAPPlan,
        current_time: Optional[pd.Timestamp] = None,
    ) -> Optional[Order]:
        """Get the next child order due for execution.

        If current_time is provided, only returns orders whose
        scheduled_time has passed. Otherwise returns the first
        incomplete slice.
        """
        if plan.is_complete:
            return None

        for s in plan.slices:
            if s.is_complete:
                continue
            # Check if it's time for this slice
            if current_time is not None and s.scheduled_time is not None:
                if current_time < s.scheduled_time:
                    return None  # Not yet time
            remaining = s.target_shares - s.filled_shares
            if remaining <= 0:
                continue
            return Order(
                ticker=plan.ticker,
                side=plan.side,
                quantity=remaining,
                order_type=OrderType.MARKET,
            )
        return None

    def record_fill(
        self,
        plan: TWAPPlan,
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

    def add_randomization(
        self,
        plan: TWAPPlan,
        rng: Optional[np.random.Generator] = None,
    ) -> TWAPPlan:
        """Add random jitter to slice sizes to reduce predictability.

        Redistributes up to randomize_pct of shares between adjacent
        slices while keeping total constant.
        """
        if not plan.slices:
            return plan
        rng = rng or np.random.default_rng()

        sizes = np.array([s.target_shares for s in plan.slices], dtype=float)
        max_jitter = self._randomize_pct * sizes.mean()

        for i in range(len(sizes) - 1):
            jitter = rng.uniform(-max_jitter, max_jitter)
            jitter = max(jitter, -sizes[i])
            jitter = min(jitter, sizes[i + 1])
            sizes[i] += jitter
            sizes[i + 1] -= jitter

        sizes = np.maximum(sizes, 0).astype(int)
        # Fix rounding
        diff = plan.total_shares - sizes.sum()
        if diff != 0:
            sizes[-1] += diff

        for i, s in enumerate(plan.slices):
            s.target_shares = int(sizes[i])

        return plan
