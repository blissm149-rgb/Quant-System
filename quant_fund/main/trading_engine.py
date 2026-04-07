"""Trading engine — always-on event-driven trading loop.

The central orchestrator that runs continuously, processing events
and coordinating all system components. Replaces the batch-oriented
PaperTradingRunner/LiveTradingRunner for always-on operation.

Lifecycle:
    1. INITIALIZING: Load state snapshots, connect data feeds
    2. DATA_READY: Data feed validated, run readiness checks
    3. TRADING_ENABLED: Process market data → signals → risk → orders
    4. RISK_HALT: Suspend trading, wait for manual reset
    5. SHUTDOWN: Persist state, disconnect, exit

The engine uses the EventBus for component communication and the
SystemStateMachine for lifecycle control. All state is periodically
persisted via StatePersistenceManager.

Heavy lifting is delegated to focused modules:
- convergence_loop: order convergence toward target weights
- research_optimizer: research → factor model → optimization pipeline
- engine_health: reconciliation, health checks, state persistence
"""

import logging
import signal
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    BrokerInterface,
    OrderStatus,
)
from quant_fund.execution.order_management.order_generator import (
    OrderGenerator,
)
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.execution.reconciliation_engine import ReconciliationEngine
from quant_fund.infrastructure.event_bus import (
    Event,
    EventBus,
    EventPriority,
    EventType,
)
from quant_fund.infrastructure.state_persistence_manager import (
    StatePersistenceManager,
)
from quant_fund.infrastructure.state_store import StateStore
from quant_fund.infrastructure.system_state_machine import (
    ReadinessCheck,
    SystemState,
    SystemStateMachine,
)
from quant_fund.main.convergence_loop import run_convergence_tick
from quant_fund.main.engine_health import (
    register_readiness_checks,
    restore_state,
    run_health_check,
    run_reconciliation,
    take_snapshot,
)
from quant_fund.main.research_optimizer import run_research_and_optimize
from quant_fund.monitoring.alerting_system import AlertingSystem, AlertLevel
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.monitoring.system_health_monitor import (
    HealthCheck,
    SystemHealthMonitor,
)
from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

logger = logging.getLogger(__name__)


class TradingEngine:
    """Always-on event-driven trading engine.

    This is the top-level orchestrator. It:
    - Manages the system lifecycle via SystemStateMachine
    - Routes events via EventBus
    - Persists state via StatePersistenceManager
    - Reconciles with broker via ReconciliationEngine
    - Runs the portfolio convergence loop
    - Handles graceful shutdown via signal handlers

    Usage:
        engine = TradingEngine(config={...})
        engine.inject_components(
            broker=broker,
            order_generator=og,
            order_router=router,
            kill_switch=ks,
            ...
        )
        engine.run()  # blocks until shutdown
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}

        # Core infrastructure
        self._event_bus = EventBus(
            config=cfg.get("event_bus", {}),
            synchronous=cfg.get("synchronous", False),
        )
        self._state_machine = SystemStateMachine()

        db_path = cfg.get("state_db_path", ":memory:")
        self._state_store = StateStore(db_path)
        self._persistence = StatePersistenceManager(self._state_store)

        # Timing configuration
        self._tick_interval_s: float = cfg.get("tick_interval_s", 1.0)
        self._snapshot_interval_s: float = cfg.get("snapshot_interval_s", 60.0)
        self._reconciliation_interval_s: float = cfg.get(
            "reconciliation_interval_s", 300.0
        )
        self._health_check_interval_s: float = cfg.get(
            "health_check_interval_s", 60.0
        )
        self._convergence_interval_s: float = cfg.get(
            "convergence_interval_s", 5.0
        )
        self._research_interval_s: float = cfg.get(
            "research_interval_s", 3600.0  # default: rebalance hourly
        )

        # Components (injected)
        self._broker: Optional[BrokerInterface] = None
        self._research_runner = None
        self._portfolio_optimizer = None
        self._constraint_engine = None
        self._kill_switch: Optional[KillSwitch] = None
        self._drawdown_monitor: Optional[DrawdownMonitor] = None
        self._exposure_monitor = None
        self._leverage_controller = None
        self._order_generator: Optional[OrderGenerator] = None
        self._order_router: Optional[OrderRouter] = None
        self._pnl_dashboard: Optional[PnLDashboard] = None
        self._alerting: Optional[AlertingSystem] = None
        self._health_monitor: Optional[SystemHealthMonitor] = None
        self._reconciliation: Optional[ReconciliationEngine] = None
        self._live_data_adapter = None
        self._risk_cascade = None
        self._execution_quality_monitor = None
        self._trade_recorder = None
        self._strategy_allocator = None
        self._strategy_nav_allocations: Dict[str, float] = {}
        self._capacity_model = None
        self._max_impact_bps: float = cfg.get("max_impact_bps", 50.0)
        self._volatility_estimates: Optional[pd.Series] = None
        self._factor_exposure_estimator = None
        self._factor_covariance_estimator = None
        self._market_hours_enforcer = None
        self._stock_returns: Optional[pd.DataFrame] = None
        self._factor_returns: Optional[pd.DataFrame] = None
        self._historical_data: Optional[pd.DataFrame] = None
        self._model_registry = None
        self._model_health_monitor = None

        # Context data (injected or updated)
        self._sector_map: Optional[Dict[str, str]] = None
        self._factor_exposures: Optional[pd.DataFrame] = None

        # Runtime state
        self._target_weights: Optional[pd.Series] = None
        self._latest_alpha_scores: Optional[pd.Series] = None
        self._last_snapshot_time: float = 0.0
        self._last_reconciliation_time: float = 0.0
        self._last_health_check_time: float = 0.0
        self._last_convergence_time: float = 0.0
        self._last_signal_time: float = 0.0
        self._last_research_time: float = 0.0
        self._strategy_id: str = cfg.get("strategy_id", "default")

        # Shutdown coordination
        self._shutdown_requested = False
        self._lock = threading.Lock()

        # Data lock protects shared mutable state accessed from multiple
        # threads (e.g. signal handler thread, dashboard thread, main loop).
        self._data_lock = threading.RLock()

        # Register event handlers
        self._register_event_handlers()

    # ------------------------------------------------------------------
    # Component injection
    # ------------------------------------------------------------------

    def inject_components(self, **components) -> None:
        """Inject system components by name.

        Accepted keys: broker, research_runner, portfolio_optimizer,
        constraint_engine, kill_switch, drawdown_monitor, exposure_monitor,
        leverage_controller, order_generator, order_router, pnl_dashboard,
        alerting, health_monitor, live_data_adapter, risk_cascade,
        execution_quality_monitor, strategy_allocator, capacity_model,
        factor_exposure_estimator, factor_covariance_estimator,
        market_hours_enforcer, stock_returns, factor_returns,
        sector_map, factor_exposures, trade_recorder,
        model_registry, model_health_monitor.
        """
        for name, component in components.items():
            attr = f"_{name}"
            if hasattr(self, attr):
                setattr(self, attr, component)
                logger.debug("Injected component: %s", name)
            else:
                logger.warning("Unknown component: %s", name)

        # Auto-create reconciliation engine if broker is available
        if self._broker is not None and self._reconciliation is None:
            self._reconciliation = ReconciliationEngine(self._broker)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _register_event_handlers(self) -> None:
        """Wire up internal event handlers."""
        self._event_bus.subscribe(
            "engine_market_data",
            self._on_market_data,
            {EventType.MARKET_DATA},
            priority=EventPriority.NORMAL,
        )
        self._event_bus.subscribe(
            "engine_signal",
            self._on_signal_generated,
            {EventType.SIGNAL_GENERATED},
            priority=EventPriority.NORMAL,
        )
        self._event_bus.subscribe(
            "engine_fill",
            self._on_order_fill,
            {EventType.ORDER_FILL},
            priority=EventPriority.HIGH,
        )
        self._event_bus.subscribe(
            "engine_risk",
            self._on_risk_check,
            {EventType.RISK_CHECK},
            priority=EventPriority.HIGH,
        )

    def _on_market_data(self, event: Event) -> None:
        """Handle new market data event."""
        if not self._state_machine.can_trade:
            return
        if self._health_monitor is not None:
            feed = event.payload.get("feed", "default")
            self._health_monitor.record_data_timestamp(
                feed, pd.Timestamp.now()
            )

    def _on_signal_generated(self, event: Event) -> None:
        """Handle new alpha signals."""
        if not self._state_machine.can_trade:
            return
        scores_dict = event.payload.get("alpha_scores", {})
        if scores_dict:
            with self._data_lock:
                self._latest_alpha_scores = pd.Series(scores_dict, dtype=float)
            logger.info("Received alpha scores for %d tickers", len(scores_dict))

    def _on_order_fill(self, event: Event) -> None:
        """Handle fill event — update PnL and positions."""
        if self._pnl_dashboard is not None and self._broker is not None:
            nav = self._broker.get_account_value()
            self._pnl_dashboard.update(nav)

    def _on_risk_check(self, event: Event) -> None:
        """Handle risk check event — may trigger RISK_HALT."""
        action = event.payload.get("action")
        if action == "halt":
            self._state_machine.try_transition(
                SystemState.RISK_HALT,
                reason=event.payload.get("reason", "Risk check triggered"),
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the always-on trading loop.

        Blocks until shutdown is requested (via signal or explicit call).
        """
        logger.info("TradingEngine starting...")
        self._install_signal_handlers()

        try:
            self._initialize()
            self._main_loop()
        except Exception:
            logger.exception("TradingEngine crashed")
        finally:
            self._shutdown()

    def _initialize(self) -> None:
        """Initialize the system: restore state, connect feeds, check readiness."""
        logger.info("Phase 1: Initialization")

        # Restore persisted state
        results, cached_scores, open_ids = restore_state(
            persistence=self._persistence,
            kill_switch=self._kill_switch,
            drawdown_monitor=self._drawdown_monitor,
            pnl_dashboard=self._pnl_dashboard,
            state_machine=self._state_machine,
            research_runner=self._research_runner,
        )
        if cached_scores is not None:
            self._latest_alpha_scores = cached_scores

        # Connect data feed
        if self._live_data_adapter is not None:
            try:
                self._live_data_adapter.connect()
                logger.info("Live data feed connected")
            except Exception:
                logger.exception("Failed to connect data feed")

        # Register readiness checks
        register_readiness_checks(
            state_machine=self._state_machine,
            broker=self._broker,
            order_generator=self._order_generator,
            order_router=self._order_router,
            kill_switch=self._kill_switch,
        )

        # Transition to DATA_READY
        self._state_machine.transition_to(
            SystemState.DATA_READY,
            reason="Initialization complete",
        )

        # Run readiness checks and transition to TRADING_ENABLED
        ready, checks = self._state_machine.transition_if_ready(
            SystemState.TRADING_ENABLED,
            reason="All readiness checks passed",
        )
        if not ready:
            failed = [c for c in checks if not c.passed]
            logger.warning(
                "System not ready for trading: %s",
                [(c.name, c.message) for c in failed],
            )

        # Start event bus worker
        self._event_bus.start()

        # Reconcile with broker on startup
        run_reconciliation(
            reconciliation=self._reconciliation,
            broker=self._broker,
            pnl_dashboard=self._pnl_dashboard,
            order_router=self._order_router,
            event_bus=self._event_bus,
        )

    def _main_loop(self) -> None:
        """The core always-on loop."""
        logger.info("Phase 2: Main loop started")
        consecutive_errors = 0
        max_consecutive_errors = 50

        while not self._shutdown_requested:
            now = time.monotonic()

            try:
                # Periodic state snapshot
                if now - self._last_snapshot_time >= self._snapshot_interval_s:
                    self._take_snapshot()
                    self._last_snapshot_time = now

                # Periodic reconciliation
                if (
                    now - self._last_reconciliation_time
                    >= self._reconciliation_interval_s
                ):
                    self._run_reconciliation()
                    self._last_reconciliation_time = now

                # Periodic health check
                if (
                    now - self._last_health_check_time
                    >= self._health_check_interval_s
                ):
                    self._run_health_check()
                    self._last_health_check_time = now

                # Portfolio convergence (only when trading)
                if self._state_machine.can_trade:
                    # Market hours gate
                    if self._market_hours_enforcer is not None:
                        can_trade, reason = self._market_hours_enforcer.can_submit_order(
                            pd.Timestamp.now(tz="America/New_York")
                        )
                        if not can_trade:
                            self._event_bus.drain()
                            time.sleep(self._tick_interval_s)
                            continue

                    # Periodic research & optimization
                    if (
                        now - self._last_research_time
                        >= self._research_interval_s
                    ):
                        self._run_research_and_optimize()
                        self._last_research_time = now

                    if (
                        now - self._last_convergence_time
                        >= self._convergence_interval_s
                    ):
                        self._convergence_tick()
                        self._last_convergence_time = now

                    # Risk checks
                    self._check_risk()

                # Process queued events
                self._event_bus.drain()

                # Reset error counter on successful tick
                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                logger.exception(
                    "Error in main loop tick (%d/%d consecutive)",
                    consecutive_errors, max_consecutive_errors,
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        "Error budget exhausted (%d consecutive errors) "
                        "— transitioning to RISK_HALT",
                        consecutive_errors,
                    )
                    self._state_machine.try_transition(
                        SystemState.RISK_HALT,
                        reason=f"Error budget exhausted: {consecutive_errors} consecutive errors",
                    )
                    if self._alerting is not None:
                        self._alerting.send(
                            AlertLevel.CRITICAL,
                            "error_budget",
                            f"Main loop error budget exhausted after {consecutive_errors} consecutive errors",
                        )
                    consecutive_errors = 0

            time.sleep(self._tick_interval_s)

    def _shutdown(self) -> None:
        """Graceful shutdown: persist state, disconnect feeds, stop bus."""
        logger.info("Phase 3: Shutdown")

        self._state_machine.try_transition(
            SystemState.SHUTDOWN, reason="Engine shutdown"
        )

        # Final state snapshot
        self._take_snapshot()

        # Stop event bus
        self._event_bus.stop()

        # Disconnect data feed
        if self._live_data_adapter is not None:
            try:
                self._live_data_adapter.disconnect()
            except Exception:
                logger.exception("Error disconnecting data feed")

        # Close state store
        self._state_store.close()

        logger.info("TradingEngine shutdown complete")

    # ------------------------------------------------------------------
    # Delegated operations
    # ------------------------------------------------------------------

    def _convergence_tick(self) -> None:
        """Delegate to convergence_loop module."""
        with self._data_lock:
            target_weights = self._target_weights
        run_convergence_tick(
            target_weights=target_weights,
            broker=self._broker,
            order_generator=self._order_generator,
            order_router=self._order_router,
            event_bus=self._event_bus,
            state_machine=self._state_machine,
            risk_cascade=self._risk_cascade,
            leverage_controller=self._leverage_controller,
            exposure_monitor=self._exposure_monitor,
            capacity_model=self._capacity_model,
            execution_quality_monitor=self._execution_quality_monitor,
            trade_recorder=self._trade_recorder,
            strategy_allocator=self._strategy_allocator,
            strategy_nav_allocations=self._strategy_nav_allocations,
            strategy_id=self._strategy_id,
            sector_map=self._sector_map,
            factor_exposures=self._factor_exposures,
            volatility_estimates=self._volatility_estimates,
            max_impact_bps=self._max_impact_bps,
        )

    def _run_research_and_optimize(self) -> None:
        """Delegate to research_optimizer module."""
        target_weights, alpha_scores, factor_exposures = run_research_and_optimize(
            research_runner=self._research_runner,
            portfolio_optimizer=self._portfolio_optimizer,
            constraint_engine=self._constraint_engine,
            broker=self._broker,
            event_bus=self._event_bus,
            factor_exposure_estimator=self._factor_exposure_estimator,
            factor_covariance_estimator=self._factor_covariance_estimator,
            stock_returns=self._stock_returns,
            factor_returns=self._factor_returns,
            factor_exposures=self._factor_exposures,
            sector_map=self._sector_map,
            historical_data=self._historical_data,
            live_data_adapter=self._live_data_adapter,
        )
        if alpha_scores is not None:
            self._latest_alpha_scores = alpha_scores
        if factor_exposures is not None:
            self._factor_exposures = factor_exposures
        if target_weights is not None:
            with self._data_lock:
                self._target_weights = target_weights

    def _take_snapshot(self) -> None:
        """Delegate to engine_health module."""
        take_snapshot(
            persistence=self._persistence,
            state_store=self._state_store,
            broker=self._broker,
            order_router=self._order_router,
            kill_switch=self._kill_switch,
            drawdown_monitor=self._drawdown_monitor,
            pnl_dashboard=self._pnl_dashboard,
            state_machine=self._state_machine,
            latest_alpha_scores=self._latest_alpha_scores,
            research_runner=self._research_runner,
        )

    def _run_reconciliation(self) -> None:
        """Delegate to engine_health module."""
        run_reconciliation(
            reconciliation=self._reconciliation,
            broker=self._broker,
            pnl_dashboard=self._pnl_dashboard,
            order_router=self._order_router,
            event_bus=self._event_bus,
        )

    def _run_health_check(self) -> None:
        """Delegate to engine_health module."""
        run_health_check(
            health_monitor=self._health_monitor,
            broker=self._broker,
            state_machine=self._state_machine,
            event_bus=self._event_bus,
        )

    def _check_risk(self) -> None:
        """Run risk checks: kill switch and drawdown."""
        if self._broker is None:
            return

        nav = self._broker.get_account_value()

        # Kill switch
        if self._kill_switch is not None:
            if self._kill_switch.check(nav):
                self._state_machine.try_transition(
                    SystemState.RISK_HALT,
                    reason=f"Kill switch triggered at NAV={nav:.2f}",
                )
                if self._alerting is not None:
                    self._alerting.send(
                        AlertLevel.CRITICAL,
                        "kill_switch",
                        f"Kill switch triggered. NAV={nav:.2f}",
                    )
                return
            self._kill_switch.update_peak(nav)

        # Drawdown monitor
        if self._drawdown_monitor is not None:
            alerts = self._drawdown_monitor.update(nav)
            for alert in alerts:
                if self._alerting is not None:
                    level = {
                        "WARNING": AlertLevel.WARNING,
                        "ALERT": AlertLevel.WARNING,
                        "CRITICAL": AlertLevel.CRITICAL,
                    }.get(alert.level.value, AlertLevel.INFO)
                    self._alerting.send(
                        level, "drawdown_monitor", alert.message
                    )

    def get_execution_quality_summary(self):
        """Get aggregate execution quality metrics."""
        if self._execution_quality_monitor is None:
            return None
        return self._execution_quality_monitor.get_summary()

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def update_target_weights(self, weights: pd.Series) -> None:
        """Set the target portfolio weights for convergence."""
        with self._data_lock:
            self._target_weights = weights
        logger.info("Target weights updated: %d tickers", len(weights))

    def update_sector_map(self, sector_map: Dict[str, str]) -> None:
        """Set the ticker → sector mapping for exposure monitoring."""
        self._sector_map = sector_map

    def update_factor_exposures(self, factor_exposures: pd.DataFrame) -> None:
        """Set the current factor exposures (tickers x factors) for risk checks."""
        self._factor_exposures = factor_exposures

    def update_volatility_estimates(self, volatility: pd.Series) -> None:
        """Set per-ticker daily volatility estimates for capacity model."""
        self._volatility_estimates = volatility

    def update_strategy_allocations(
        self, strategy_sharpes: Dict[str, float], **kwargs
    ) -> Dict[str, float]:
        """Compute and cache strategy allocations via the injected allocator."""
        if self._strategy_allocator is None:
            return {}
        allocations = self._strategy_allocator.allocate(
            strategy_sharpes, **kwargs
        )
        self._strategy_nav_allocations = allocations
        logger.info("Strategy allocations updated: %s", allocations)
        return allocations

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except (ValueError, OSError):
            logger.debug("Cannot install signal handlers (not main thread)")

    def _handle_signal(self, signum, frame) -> None:
        """Handle shutdown signal."""
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, initiating shutdown", sig_name)
        self.request_shutdown()

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the engine."""
        with self._lock:
            self._shutdown_requested = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> SystemState:
        return self._state_machine.state

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def state_machine(self) -> SystemStateMachine:
        return self._state_machine

    @property
    def is_running(self) -> bool:
        return not self._shutdown_requested
