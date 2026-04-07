"""Post-market phase -- EOD P&L, trade reporting, reconciliation.

Runs daily at ~16:00 ET after market close. Settles the trading day
and generates end-of-day reports.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PostMarketResult:
    """Outcome of the post-market phase."""

    eod_nav: float = 0.0
    eod_pnl: float = 0.0
    daily_return: float = 0.0
    num_trades: int = 0
    reconciliation_passed: bool = False
    snapshot_saved: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class PostMarketPhase:
    """Orchestrates post-market settlement and reporting.

    Steps:
        1. Calculate end-of-day P&L
        2. Generate trade report
        3. Run fill quality analysis
        4. Full position reconciliation with broker
        5. Save comprehensive state snapshot
        6. Publish EOD summary alerts
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._strategy_id = cfg.get("strategy_id", "default")

        # Injected components
        self._broker = None
        self._pnl_dashboard = None
        self._reconciliation = None
        self._execution_quality_monitor = None
        self._trade_recorder = None
        self._alerting = None
        self._persistence = None
        self._state_store = None

    def inject_components(self, **components) -> None:
        for name, component in components.items():
            attr = f"_{name}"
            if hasattr(self, attr):
                setattr(self, attr, component)

    def run(self) -> PostMarketResult:
        """Execute the full post-market sequence."""
        result = PostMarketResult()

        # 1. Calculate EOD P&L
        self._calculate_eod_pnl(result)

        # 2. Execution quality summary
        self._report_execution_quality(result)

        # 3. Position reconciliation
        self._reconcile(result)

        # 4. Save state snapshot
        self._save_snapshot(result)

        # 5. Publish EOD summary
        self._publish_summary(result)

        logger.info(
            "Post-market complete: NAV=$%,.2f, PnL=$%,.2f, return=%.4f%%",
            result.eod_nav,
            result.eod_pnl,
            result.daily_return * 100,
        )
        return result

    def _calculate_eod_pnl(self, result: PostMarketResult) -> None:
        """Calculate end-of-day NAV and P&L."""
        if self._broker is None:
            result.warnings.append("No broker -- cannot calculate EOD P&L")
            return
        try:
            result.eod_nav = self._broker.get_account_value()
            if self._pnl_dashboard is not None:
                summary = self._pnl_dashboard.get_summary()
                if summary:
                    result.daily_return = summary.get("total_return", 0.0)
                    result.eod_pnl = result.eod_nav * result.daily_return
        except Exception as e:
            result.errors.append(f"EOD P&L calculation failed: {e}")

    def _report_execution_quality(self, result: PostMarketResult) -> None:
        """Generate execution quality report."""
        if self._execution_quality_monitor is None:
            return
        try:
            eq = self._execution_quality_monitor.get_summary()
            if eq:
                result.num_trades = eq.num_orders
                logger.info(
                    "Execution quality: %d orders, avg fill rate=%.2f%%",
                    eq.num_orders,
                    eq.avg_fill_rate * 100,
                )
        except Exception as e:
            result.warnings.append(f"Execution quality report failed: {e}")

    def _reconcile(self, result: PostMarketResult) -> None:
        """Full position reconciliation with broker."""
        if self._reconciliation is None:
            result.warnings.append("No reconciliation engine")
            return
        try:
            recon_result = self._reconciliation.reconcile()
            result.reconciliation_passed = True
            if hasattr(recon_result, "discrepancies") and recon_result.discrepancies:
                result.reconciliation_passed = False
                result.warnings.append(
                    f"{len(recon_result.discrepancies)} position discrepancies"
                )
            logger.info("Post-market reconciliation complete")
        except Exception as e:
            result.errors.append(f"Reconciliation failed: {e}")

    def _save_snapshot(self, result: PostMarketResult) -> None:
        """Save comprehensive state snapshot."""
        if self._persistence is None:
            return
        try:
            self._persistence.save_snapshot(
                component="post_market",
                data={"eod_nav": result.eod_nav, "daily_return": result.daily_return},
            )
            result.snapshot_saved = True
        except Exception as e:
            result.warnings.append(f"Snapshot save failed: {e}")

    def _publish_summary(self, result: PostMarketResult) -> None:
        """Publish EOD summary via alerting system."""
        if self._alerting is None:
            return
        try:
            from quant_fund.monitoring.alerting_system import AlertLevel

            msg = (
                f"EOD Summary: NAV=${result.eod_nav:,.2f}, "
                f"PnL=${result.eod_pnl:,.2f}, "
                f"Return={result.daily_return:.4%}, "
                f"Trades={result.num_trades}"
            )
            self._alerting.send(AlertLevel.INFO, "post_market", msg)
        except Exception as e:
            result.warnings.append(f"EOD summary publish failed: {e}")
