"""Pre-market phase -- data refresh, broker check, model warm-up.

Runs daily at ~04:00 ET before market open. Prepares the system
for the trading session by refreshing data, verifying connectivity,
and warming up ML models.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreMarketResult:
    """Outcome of the pre-market preparation phase."""

    data_refreshed: bool = False
    broker_connected: bool = False
    positions_reconciled: bool = False
    models_warmed_up: bool = False
    data_feed_healthy: bool = False
    config_changes_detected: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return (
            self.broker_connected
            and self.data_feed_healthy
            and len(self.errors) == 0
        )


class PreMarketPhase:
    """Orchestrates pre-market preparation.

    Steps:
        1. Refresh overnight data (corporate actions, dividends, splits)
        2. Verify broker connectivity and account status
        3. Reconcile positions against broker
        4. Warm up ML models (run prediction on yesterday's data)
        5. Validate data feed health
        6. Check for config changes
        7. Report readiness status
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._warmup_tickers: List[str] = cfg.get("universe", [])

        # Injected components
        self._broker = None
        self._live_data_adapter = None
        self._reconciliation = None
        self._research_runner = None
        self._alerting = None
        self._health_monitor = None
        self._data_connector = None

    def inject_components(self, **components) -> None:
        for name, component in components.items():
            attr = f"_{name}"
            if hasattr(self, attr):
                setattr(self, attr, component)

    def run(self) -> PreMarketResult:
        """Execute the full pre-market preparation sequence."""
        result = PreMarketResult()

        # 1. Refresh overnight data
        result.data_refreshed = self._refresh_data(result)

        # 2. Verify broker connectivity
        result.broker_connected = self._check_broker(result)

        # 3. Reconcile positions
        result.positions_reconciled = self._reconcile_positions(result)

        # 4. Warm up models
        result.models_warmed_up = self._warm_up_models(result)

        # 5. Validate data feed
        result.data_feed_healthy = self._check_data_feed(result)

        # 6. Report readiness
        if result.ready:
            logger.info("Pre-market phase complete: system READY")
        else:
            logger.warning(
                "Pre-market phase complete with issues: %s",
                result.errors,
            )

        return result

    def _refresh_data(self, result: PreMarketResult) -> bool:
        """Download overnight corporate actions and pre-market quotes."""
        if self._data_connector is None:
            result.warnings.append("No data connector -- skipping data refresh")
            return False
        try:
            if self._warmup_tickers:
                bars = self._data_connector.fetch_latest(self._warmup_tickers)
                if not bars.empty:
                    logger.info("Pre-market data refreshed: %d tickers", len(bars))
                    return True
                result.warnings.append("Data refresh returned empty bars")
            return False
        except Exception as e:
            result.errors.append(f"Data refresh failed: {e}")
            logger.error("Pre-market data refresh failed", exc_info=True)
            return False

    def _check_broker(self, result: PreMarketResult) -> bool:
        """Verify broker connectivity and account status."""
        if self._broker is None:
            result.warnings.append("No broker configured")
            return False
        try:
            nav = self._broker.get_account_value()
            if nav > 0:
                logger.info("Broker connected: NAV=$%,.2f", nav)
                return True
            result.warnings.append(f"Broker returned zero NAV")
            return False
        except Exception as e:
            result.errors.append(f"Broker connectivity check failed: {e}")
            return False

    def _reconcile_positions(self, result: PreMarketResult) -> bool:
        """Reconcile internal positions against broker."""
        if self._reconciliation is None:
            result.warnings.append("No reconciliation engine -- skipping")
            return False
        try:
            recon_result = self._reconciliation.reconcile()
            if hasattr(recon_result, "discrepancies"):
                if recon_result.discrepancies:
                    result.warnings.append(
                        f"{len(recon_result.discrepancies)} position discrepancies"
                    )
            logger.info("Position reconciliation complete")
            return True
        except Exception as e:
            result.errors.append(f"Reconciliation failed: {e}")
            return False

    def _warm_up_models(self, result: PreMarketResult) -> bool:
        """Run ML models on yesterday's data to pre-load and validate."""
        if self._research_runner is None:
            result.warnings.append("No research runner -- skipping model warm-up")
            return False
        try:
            yesterday = pd.Timestamp.now() - pd.Timedelta(days=1)
            self._research_runner.run_cycle(as_of=yesterday)
            logger.info("Model warm-up complete")
            return True
        except Exception as e:
            result.warnings.append(f"Model warm-up failed (non-critical): {e}")
            return False

    def _check_data_feed(self, result: PreMarketResult) -> bool:
        """Validate that the live data feed is connectable."""
        if self._live_data_adapter is None:
            result.warnings.append("No live data adapter -- skipping feed check")
            return True  # not a blocker if no adapter configured
        try:
            self._live_data_adapter.connect()
            logger.info("Data feed health check passed")
            return True
        except Exception as e:
            result.errors.append(f"Data feed health check failed: {e}")
            return False
