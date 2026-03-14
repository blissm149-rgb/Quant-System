"""Execution quality monitor.

Tracks implementation shortfall, fill rates, slippage vs model,
and adverse selection flags for post-trade analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Record of a single order's execution quality."""

    order_id: str
    ticker: str
    side: str
    target_qty: int
    filled_qty: int
    decision_price: float  # price when order was generated
    fill_price: float
    vwap_benchmark: float = 0.0
    implementation_shortfall_bps: float = 0.0
    slippage_vs_vwap_bps: float = 0.0
    fill_rate: float = 0.0
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class ExecutionSummary:
    """Aggregate execution quality metrics."""

    num_orders: int = 0
    avg_implementation_shortfall_bps: float = 0.0
    avg_slippage_vs_vwap_bps: float = 0.0
    avg_fill_rate: float = 0.0
    total_adverse_selection_flags: int = 0
    total_cost_bps: float = 0.0


class ExecutionQualityMonitor:
    """Monitors execution quality across all orders.

    Tracks:
    - Implementation shortfall vs decision price
    - Fill rate (% of target trades executed)
    - Slippage vs VWAP benchmark
    - Adverse selection flags
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._adverse_threshold_bps = cfg.get("adverse_threshold_bps", 20.0)
        self._records: List[ExecutionRecord] = []

    def record_execution(
        self,
        order_id: str,
        ticker: str,
        side: str,
        target_qty: int,
        filled_qty: int,
        decision_price: float,
        fill_price: float,
        vwap_benchmark: float = 0.0,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> ExecutionRecord:
        """Record the execution quality of a single order.

        Parameters
        ----------
        decision_price : float
            Mid price when the order decision was made.
        fill_price : float
            Actual average fill price.
        vwap_benchmark : float
            VWAP over the execution period for comparison.
        """
        # Implementation shortfall: signed difference from decision price
        if decision_price > 0:
            if side == "buy":
                is_bps = (fill_price - decision_price) / decision_price * 10000.0
            else:
                is_bps = (decision_price - fill_price) / decision_price * 10000.0
        else:
            is_bps = 0.0

        # Slippage vs VWAP
        slippage_vwap = 0.0
        if vwap_benchmark > 0:
            if side == "buy":
                slippage_vwap = (fill_price - vwap_benchmark) / vwap_benchmark * 10000.0
            else:
                slippage_vwap = (vwap_benchmark - fill_price) / vwap_benchmark * 10000.0

        fill_rate = filled_qty / target_qty if target_qty > 0 else 0.0

        record = ExecutionRecord(
            order_id=order_id,
            ticker=ticker,
            side=side,
            target_qty=target_qty,
            filled_qty=filled_qty,
            decision_price=decision_price,
            fill_price=fill_price,
            vwap_benchmark=vwap_benchmark,
            implementation_shortfall_bps=is_bps,
            slippage_vs_vwap_bps=slippage_vwap,
            fill_rate=fill_rate,
            timestamp=timestamp or pd.Timestamp.now(),
        )
        self._records.append(record)
        return record

    def get_summary(
        self, since: Optional[pd.Timestamp] = None
    ) -> ExecutionSummary:
        """Get aggregate execution quality metrics."""
        records = self._records
        if since is not None:
            records = [r for r in records if r.timestamp and r.timestamp >= since]

        if not records:
            return ExecutionSummary()

        is_vals = [r.implementation_shortfall_bps for r in records]
        vwap_vals = [r.slippage_vs_vwap_bps for r in records]
        fill_rates = [r.fill_rate for r in records]
        adverse_flags = sum(
            1 for r in records
            if r.implementation_shortfall_bps > self._adverse_threshold_bps
        )

        return ExecutionSummary(
            num_orders=len(records),
            avg_implementation_shortfall_bps=float(np.mean(is_vals)),
            avg_slippage_vs_vwap_bps=float(np.mean(vwap_vals)),
            avg_fill_rate=float(np.mean(fill_rates)),
            total_adverse_selection_flags=adverse_flags,
            total_cost_bps=float(np.sum(is_vals)),
        )

    def get_records_dataframe(self) -> pd.DataFrame:
        """Get all execution records as a DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "order_id": r.order_id,
                "ticker": r.ticker,
                "side": r.side,
                "target_qty": r.target_qty,
                "filled_qty": r.filled_qty,
                "fill_rate": r.fill_rate,
                "implementation_shortfall_bps": r.implementation_shortfall_bps,
                "slippage_vs_vwap_bps": r.slippage_vs_vwap_bps,
            }
            for r in self._records
        ])

    def detect_adverse_selection(self) -> List[ExecutionRecord]:
        """Return orders flagged for adverse selection."""
        return [
            r for r in self._records
            if r.implementation_shortfall_bps > self._adverse_threshold_bps
        ]
