"""Risk cascade coordinator — unified risk check pipeline.

Orchestrates kill_switch, drawdown_monitor, leverage_controller, and
exposure_monitor in a defined execution order. Provides a single
entry point for the TradingEngine's convergence loop and risk checks.

Cascade order (hardest stop first):
    1. Kill switch check → HALT if triggered
    2. Drawdown monitor update → collect alerts
    3. Leverage enforcement → scale weights if needed
    4. Exposure monitor check → block orders if breached

Also tracks factor exposure snapshots over time for regime detection.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from quant_fund.risk_engine.drawdown_monitor import DrawdownAlert

logger = logging.getLogger(__name__)


@dataclass
class RiskCascadeResult:
    """Outcome of a full risk cascade evaluation."""

    kill_switch_triggered: bool = False
    drawdown_alerts: List[DrawdownAlert] = field(default_factory=list)
    leverage_enforced: bool = False
    original_gross: float = 0.0
    adjusted_gross: float = 0.0
    exposure_breaches: List[Any] = field(default_factory=list)
    orders_blocked: bool = False
    adjusted_weights: Optional[pd.Series] = None


class RiskCascadeCoordinator:
    """Unified risk check cascade.

    Accepts optional risk components via constructor injection.
    Components set to None are skipped.

    Usage:
        coordinator = RiskCascadeCoordinator(
            kill_switch=ks,
            drawdown_monitor=dm,
            leverage_controller=lc,
            exposure_monitor=em,
        )
        result = coordinator.run_cascade(nav, weights, sector_map)
        if result.orders_blocked:
            # do not generate orders
            ...
        else:
            orders = generate_orders(result.adjusted_weights, ...)
    """

    def __init__(
        self,
        kill_switch=None,
        drawdown_monitor=None,
        leverage_controller=None,
        exposure_monitor=None,
        config: Optional[dict] = None,
    ):
        cfg = config or {}
        self._kill_switch = kill_switch
        self._drawdown_monitor = drawdown_monitor
        self._leverage_controller = leverage_controller
        self._exposure_monitor = exposure_monitor

        # Factor exposure history (append-only, capped)
        max_history = cfg.get("max_factor_history", 5000)
        self._factor_history: deque = deque(maxlen=max_history)

    def run_cascade(
        self,
        nav: float,
        weights: pd.Series,
        sector_map: Optional[Dict[str, str]] = None,
        factor_exposures: Optional[pd.DataFrame] = None,
    ) -> RiskCascadeResult:
        """Run the full risk cascade. Returns result with adjusted weights.

        Parameters
        ----------
        nav : float
            Current net asset value.
        weights : pd.Series
            Target portfolio weights indexed by ticker.
        sector_map : dict, optional
            Ticker → sector mapping for exposure checks.
        factor_exposures : DataFrame, optional
            Tickers x factors for exposure checks.

        Returns
        -------
        RiskCascadeResult
            Contains adjusted weights and all check outcomes.
        """
        result = RiskCascadeResult(adjusted_weights=weights.copy())

        # Step 1: Kill switch
        if self._kill_switch is not None:
            if self._kill_switch.check(nav):
                result.kill_switch_triggered = True
                result.orders_blocked = True
                logger.warning("Risk cascade: kill switch triggered at NAV=%.2f", nav)
                return result
            self._kill_switch.update_peak(nav)

        # Step 2: Drawdown monitor
        if self._drawdown_monitor is not None:
            alerts = self._drawdown_monitor.update(nav)
            result.drawdown_alerts = alerts

        # Step 3: Leverage enforcement
        working_weights = weights
        result.original_gross = float(weights.abs().sum())

        if self._leverage_controller is not None:
            working_weights = self._leverage_controller.enforce(weights)
            result.adjusted_gross = float(working_weights.abs().sum())
            if result.adjusted_gross < result.original_gross - 1e-9:
                result.leverage_enforced = True
                logger.info(
                    "Risk cascade: leverage enforced %.3f → %.3f",
                    result.original_gross,
                    result.adjusted_gross,
                )
        else:
            result.adjusted_gross = result.original_gross

        # Step 4: Exposure check
        if self._exposure_monitor is not None:
            breaches = self._exposure_monitor.check(
                working_weights,
                sector_map=sector_map,
                factor_exposures=factor_exposures,
            )
            result.exposure_breaches = breaches
            if breaches:
                result.orders_blocked = True
                logger.warning(
                    "Risk cascade: exposure breaches — orders blocked: %s",
                    [b.message for b in breaches],
                )
                result.adjusted_weights = working_weights
                return result

        result.adjusted_weights = working_weights
        return result

    # ------------------------------------------------------------------
    # Factor exposure tracking
    # ------------------------------------------------------------------

    def track_factor_exposure(
        self,
        timestamp: pd.Timestamp,
        weights: pd.Series,
        factor_exposures: Optional[pd.DataFrame] = None,
    ) -> None:
        """Record a factor exposure snapshot for time-series tracking.

        Parameters
        ----------
        timestamp : Timestamp
            When the snapshot was taken.
        weights : pd.Series
            Portfolio weights at this time.
        factor_exposures : DataFrame, optional
            Tickers x factors exposure matrix.
        """
        if factor_exposures is None or factor_exposures.empty:
            return

        # Compute portfolio-level factor exposures: w^T @ B
        common = weights.index.intersection(factor_exposures.index)
        if len(common) == 0:
            return

        w = weights.reindex(common).fillna(0.0)
        B = factor_exposures.loc[common]
        portfolio_exposures = B.T @ w

        self._factor_history.append({
            "timestamp": timestamp,
            **{str(k): float(v) for k, v in portfolio_exposures.items()},
        })

    def get_factor_exposure_history(self) -> pd.DataFrame:
        """Return factor exposure history as DataFrame (timestamp x factor).

        Returns
        -------
        pd.DataFrame
            Index is timestamp, columns are factor names.
        """
        if not self._factor_history:
            return pd.DataFrame()

        df = pd.DataFrame(list(self._factor_history))
        df = df.set_index("timestamp")
        return df

    @property
    def factor_history_length(self) -> int:
        return len(self._factor_history)
