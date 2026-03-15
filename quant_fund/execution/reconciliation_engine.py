"""Reconciliation engine — continuous verification of internal vs broker state.

Periodically compares internal positions, open orders, and cash balance
against the broker's reported state. Detects discrepancies, auto-corrects
internal state where safe, and alerts operators for manual review when
correction is not automatic.

Reconciliation checks:
    - Positions: internal positions vs broker positions
    - Open orders: internal order tracking vs broker open orders
    - Cash/NAV: internal PnL vs broker account value

If discrepancies are found:
    1. Log the discrepancy with full details
    2. Correct internal state (broker is source of truth)
    3. Alert operators if the correction exceeds a threshold
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    BrokerInterface,
    Fill,
)

logger = logging.getLogger(__name__)


@dataclass
class PositionDiscrepancy:
    """A discrepancy between internal and broker position."""

    ticker: str
    internal_qty: float
    broker_qty: float
    delta: float
    resolved: bool = False
    resolution: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "internal_qty": self.internal_qty,
            "broker_qty": self.broker_qty,
            "delta": self.delta,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


@dataclass
class OrderDiscrepancy:
    """A discrepancy between internal and broker order tracking."""

    order_id: str
    status: str  # "missing_internal", "missing_broker", "status_mismatch"
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "status": self.status,
            "details": self.details,
        }


@dataclass
class CashDiscrepancy:
    """A discrepancy between internal and broker cash/NAV."""

    internal_nav: float
    broker_nav: float
    delta: float
    delta_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "internal_nav": self.internal_nav,
            "broker_nav": self.broker_nav,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
        }


@dataclass
class ReconciliationResult:
    """Complete result of a reconciliation cycle."""

    timestamp: pd.Timestamp
    position_discrepancies: List[PositionDiscrepancy] = field(
        default_factory=list
    )
    order_discrepancies: List[OrderDiscrepancy] = field(default_factory=list)
    cash_discrepancy: Optional[CashDiscrepancy] = None
    positions_matched: bool = True
    orders_matched: bool = True
    cash_matched: bool = True
    auto_corrected: bool = False
    corrections_applied: List[str] = field(default_factory=list)

    @property
    def all_matched(self) -> bool:
        return (
            self.positions_matched
            and self.orders_matched
            and self.cash_matched
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": str(self.timestamp),
            "all_matched": self.all_matched,
            "positions_matched": self.positions_matched,
            "orders_matched": self.orders_matched,
            "cash_matched": self.cash_matched,
            "auto_corrected": self.auto_corrected,
            "position_discrepancies": [
                d.to_dict() for d in self.position_discrepancies
            ],
            "order_discrepancies": [
                d.to_dict() for d in self.order_discrepancies
            ],
            "cash_discrepancy": (
                self.cash_discrepancy.to_dict()
                if self.cash_discrepancy
                else None
            ),
            "corrections_applied": self.corrections_applied,
        }


class ReconciliationEngine:
    """Continuously reconciles internal state against broker state.

    The broker is always treated as the source of truth. When
    discrepancies are detected, internal state is corrected to
    match the broker.

    Usage:
        engine = ReconciliationEngine(broker, config={"nav_tolerance_pct": 0.01})
        result = engine.reconcile(internal_positions, internal_nav)
        if not result.all_matched:
            alerting.send(AlertLevel.WARNING, "reconciliation", str(result))
    """

    def __init__(
        self,
        broker: BrokerInterface,
        config: Optional[dict] = None,
    ):
        self._broker = broker
        cfg = config or {}
        # Position tolerance: ignore discrepancies smaller than this
        self._position_tolerance = cfg.get("position_tolerance_shares", 0)
        # NAV tolerance: percentage difference to flag
        self._nav_tolerance_pct = cfg.get("nav_tolerance_pct", 0.01)
        # Auto-correct positions to match broker
        self._auto_correct = cfg.get("auto_correct", True)
        # Alert threshold: notify operator if correction > this many shares
        self._alert_threshold_shares = cfg.get("alert_threshold_shares", 100)

        self._history: List[ReconciliationResult] = []
        self._correction_count = 0

    def reconcile(
        self,
        internal_positions: pd.Series,
        internal_nav: float = 0.0,
        internal_open_order_ids: Optional[Set[str]] = None,
    ) -> ReconciliationResult:
        """Run a full reconciliation cycle.

        Args:
            internal_positions: Internal position state (ticker → qty).
            internal_nav: Internal NAV calculation.
            internal_open_order_ids: Set of order IDs we think are open.

        Returns:
            ReconciliationResult with all discrepancies found.
        """
        result = ReconciliationResult(timestamp=pd.Timestamp.now())

        # 1. Position reconciliation
        self._reconcile_positions(result, internal_positions)

        # 2. Cash/NAV reconciliation
        self._reconcile_nav(result, internal_nav)

        # 3. Open order reconciliation
        if internal_open_order_ids is not None:
            self._reconcile_orders(result, internal_open_order_ids)

        self._history.append(result)
        return result

    def _reconcile_positions(
        self,
        result: ReconciliationResult,
        internal_positions: pd.Series,
    ) -> None:
        """Compare internal positions against broker positions."""
        broker_positions = self._broker.get_positions()

        all_tickers = set(internal_positions.index) | set(
            broker_positions.index
        )

        for ticker in sorted(all_tickers):
            internal_qty = float(internal_positions.get(ticker, 0.0))
            broker_qty = float(broker_positions.get(ticker, 0.0))
            delta = broker_qty - internal_qty

            if abs(delta) <= self._position_tolerance:
                continue

            disc = PositionDiscrepancy(
                ticker=ticker,
                internal_qty=internal_qty,
                broker_qty=broker_qty,
                delta=delta,
            )
            result.position_discrepancies.append(disc)
            result.positions_matched = False

            if self._auto_correct:
                disc.resolved = True
                disc.resolution = (
                    f"Auto-corrected: internal {internal_qty} → {broker_qty}"
                )
                result.auto_corrected = True
                result.corrections_applied.append(
                    f"{ticker}: {internal_qty} → {broker_qty}"
                )
                self._correction_count += 1

                if abs(delta) > self._alert_threshold_shares:
                    logger.warning(
                        "LARGE position discrepancy for %s: "
                        "internal=%.0f broker=%.0f delta=%.0f",
                        ticker,
                        internal_qty,
                        broker_qty,
                        delta,
                    )

    def _reconcile_nav(
        self, result: ReconciliationResult, internal_nav: float
    ) -> None:
        """Compare internal NAV against broker account value."""
        if internal_nav <= 0:
            return

        broker_nav = self._broker.get_account_value()
        if broker_nav <= 0:
            return

        delta = broker_nav - internal_nav
        delta_pct = abs(delta) / broker_nav

        if delta_pct > self._nav_tolerance_pct:
            result.cash_discrepancy = CashDiscrepancy(
                internal_nav=internal_nav,
                broker_nav=broker_nav,
                delta=delta,
                delta_pct=delta_pct,
            )
            result.cash_matched = False
            logger.warning(
                "NAV discrepancy: internal=%.2f broker=%.2f delta=%.2f (%.2f%%)",
                internal_nav,
                broker_nav,
                delta,
                delta_pct * 100,
            )

    def _reconcile_orders(
        self,
        result: ReconciliationResult,
        internal_open_order_ids: Set[str],
    ) -> None:
        """Compare internal open order tracking against broker.

        Note: the BrokerInterface doesn't have a get_open_orders() method,
        so we use get_fills() to detect completed orders that internal
        state still considers open.
        """
        # Check for fills on orders we think are still open
        since = pd.Timestamp.now() - pd.Timedelta(hours=24)
        recent_fills = self._broker.get_fills(since)
        filled_ids = {f.order_id for f in recent_fills}

        for order_id in internal_open_order_ids:
            if order_id in filled_ids:
                disc = OrderDiscrepancy(
                    order_id=order_id,
                    status="filled_but_tracked_as_open",
                    details=(
                        "Order has fills in broker but is still tracked "
                        "as open internally"
                    ),
                )
                result.order_discrepancies.append(disc)
                result.orders_matched = False

    def get_corrected_positions(
        self,
        internal_positions: pd.Series,
        result: ReconciliationResult,
    ) -> pd.Series:
        """Return positions corrected based on reconciliation result.

        If auto_correct is enabled, returns broker positions.
        Otherwise returns internal positions unchanged.
        """
        if not result.auto_corrected:
            return internal_positions

        broker_positions = self._broker.get_positions()
        return broker_positions

    # ------------------------------------------------------------------
    # History and metrics
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 50) -> List[ReconciliationResult]:
        """Return recent reconciliation history."""
        return self._history[-limit:]

    @property
    def latest(self) -> Optional[ReconciliationResult]:
        return self._history[-1] if self._history else None

    @property
    def correction_count(self) -> int:
        return self._correction_count

    def get_metrics(self) -> Dict[str, Any]:
        """Return reconciliation metrics."""
        total = len(self._history)
        mismatches = sum(1 for r in self._history if not r.all_matched)
        return {
            "total_reconciliations": total,
            "total_mismatches": mismatches,
            "total_corrections": self._correction_count,
            "mismatch_rate": mismatches / total if total > 0 else 0.0,
        }
