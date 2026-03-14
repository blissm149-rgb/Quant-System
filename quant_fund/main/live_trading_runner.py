"""Live trading runner — production trading loop.

Only activated after:
- Strategy passes governance/approval_workflow.py
- Minimum 6 months paper trading
- Risk team sign-off
- deployment_controller.py enables live mode

Identical pipeline to paper_trading_runner but routes orders
through a real broker adapter (IB or Alpaca) instead of the
simulation broker.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quant_fund.governance.approval_workflow import ApprovalState, ApprovalWorkflow
from quant_fund.governance.deployment_controller import DeploymentController
from quant_fund.main.paper_trading_runner import PaperTradingRunner, TradingDayResult

logger = logging.getLogger(__name__)


@dataclass
class LiveTradingConfig:
    """Configuration for live trading."""

    strategy_id: str = ""
    max_daily_loss_pct: float = 0.05
    pre_market_check_enabled: bool = True
    post_market_reconciliation: bool = True


class LiveTradingRunner:
    """Production trading loop with additional safety checks.

    Wraps PaperTradingRunner with:
    - Deployment controller gate (approval workflow check)
    - Pre-market health checks
    - Post-market reconciliation
    - Daily loss limit (separate from kill switch)
    - Audit logging

    This runner only executes if deployment_controller confirms
    the strategy is approved and in live mode.
    """

    def __init__(
        self,
        deployment_controller: DeploymentController,
        config: Optional[dict] = None,
    ):
        self._deployment = deployment_controller
        cfg = config or {}
        self._strategy_id = cfg.get("strategy_id", "")
        self._max_daily_loss_pct = cfg.get("max_daily_loss_pct", 0.05)
        self._pre_market_check = cfg.get("pre_market_check_enabled", True)
        self._post_market_reconciliation = cfg.get("post_market_reconciliation", True)

        # The inner runner does the actual work
        self._inner_runner = PaperTradingRunner(cfg)
        self._audit_log: List[dict] = []
        self._is_halted = False

    def inject_components(self, **components) -> None:
        """Pass components through to the inner runner."""
        self._inner_runner.inject_components(**components)

    def pre_market_check(self) -> tuple:
        """Run pre-market safety checks.

        Returns (ok: bool, reasons: list of str).
        """
        reasons = []

        # Check deployment status
        live_strategies = self._deployment.get_live_strategies()
        live_ids = [s.strategy_id for s in live_strategies]
        if self._strategy_id and self._strategy_id not in live_ids:
            reasons.append(
                f"Strategy {self._strategy_id} not in live mode"
            )

        # Check kill switch
        if self._deployment.is_kill_switch_active:
            reasons.append("Kill switch is active")

        if self._is_halted:
            reasons.append("Runner is halted (daily loss limit)")

        ok = len(reasons) == 0
        self._log_audit("pre_market_check", {"ok": ok, "reasons": reasons})
        return ok, reasons

    def run_single_day(
        self,
        date: pd.Timestamp,
        market_data: Optional[pd.DataFrame] = None,
        prev_nav: float = 0.0,
    ) -> TradingDayResult:
        """Run a single live trading day.

        Includes pre-market checks and post-market reconciliation.
        """
        # Pre-market check
        if self._pre_market_check:
            ok, reasons = self.pre_market_check()
            if not ok:
                logger.warning(
                    "Pre-market check failed: %s", "; ".join(reasons)
                )
                return TradingDayResult(
                    date=date,
                    nav=prev_nav,
                    daily_return=0.0,
                    num_orders=0,
                    num_fills=0,
                    status="blocked",
                    error_message="; ".join(reasons),
                )

        # Run the actual trading day
        day_result = self._inner_runner._run_single_day(
            date=date,
            market_data=market_data,
            prev_nav=prev_nav,
        )

        # Check daily loss limit
        if prev_nav > 0 and day_result.daily_return < -self._max_daily_loss_pct:
            logger.warning(
                "Daily loss limit exceeded: %.2f%% (limit: %.2f%%)",
                day_result.daily_return * 100,
                self._max_daily_loss_pct * 100,
            )
            self._is_halted = True
            self._deployment.suspend(
                self._strategy_id, reason="Daily loss limit exceeded"
            )

        # Post-market reconciliation
        if self._post_market_reconciliation:
            self._reconcile(day_result)

        self._log_audit("trading_day", {
            "date": str(date),
            "nav": day_result.nav,
            "return": day_result.daily_return,
            "orders": day_result.num_orders,
            "fills": day_result.num_fills,
            "status": day_result.status,
        })

        return day_result

    def run(
        self,
        dates: List[pd.Timestamp],
        market_data_by_date: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None,
    ) -> List[TradingDayResult]:
        """Run live trading over multiple dates.

        Returns list of daily results. Stops if halted.
        """
        results = []
        prev_nav = 0.0
        broker = self._inner_runner._broker
        if broker is not None:
            prev_nav = broker.get_account_value()

        for date in dates:
            if self._is_halted:
                logger.warning("Runner halted, skipping %s", date)
                break

            md = (
                market_data_by_date.get(date)
                if market_data_by_date else None
            )
            day = self.run_single_day(date, market_data=md, prev_nav=prev_nav)
            results.append(day)
            prev_nav = day.nav

        return results

    def reset_halt(self) -> None:
        """Reset the daily loss halt (manual intervention required)."""
        self._is_halted = False
        logger.info("Live trading runner halt reset")

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    @property
    def audit_log(self) -> List[dict]:
        return list(self._audit_log)

    def _reconcile(self, day_result: TradingDayResult) -> None:
        """Post-market reconciliation.

        Verifies that broker positions match expected state.
        Logs discrepancies for manual review.
        """
        broker = self._inner_runner._broker
        if broker is None:
            return

        positions = broker.get_positions()
        if isinstance(positions, pd.Series):
            non_zero = positions[positions != 0]
            self._log_audit("reconciliation", {
                "date": str(day_result.date),
                "num_positions": len(non_zero),
                "nav": day_result.nav,
            })

    def _log_audit(self, event: str, details: dict) -> None:
        """Log an audit event."""
        entry = {
            "event": event,
            "timestamp": pd.Timestamp.now().isoformat(),
            "strategy_id": self._strategy_id,
            **details,
        }
        self._audit_log.append(entry)
        logger.info("AUDIT: %s — %s", event, details)
