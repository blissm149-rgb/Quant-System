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
        sector_map, factor_exposures.
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
        # Update health monitor
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
            # Phase 1: Initialize
            self._initialize()

            # Phase 2: Main loop
            self._main_loop()

        except Exception:
            logger.exception("TradingEngine crashed")
        finally:
            # Phase 3: Shutdown
            self._shutdown()

    def _initialize(self) -> None:
        """Initialize the system: restore state, connect feeds, check readiness."""
        logger.info("Phase 1: Initialization")

        # Restore persisted state
        self._restore_state()

        # Connect data feed
        if self._live_data_adapter is not None:
            try:
                self._live_data_adapter.connect()
                logger.info("Live data feed connected")
            except Exception:
                logger.exception("Failed to connect data feed")

        # Register readiness checks
        self._register_readiness_checks()

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
        self._run_reconciliation()

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
                            # Outside market hours — skip convergence
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
                    # Reset counter to allow recovery if system resumes
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
    # Portfolio convergence loop
    # ------------------------------------------------------------------

    def _convergence_tick(self) -> None:
        """One iteration of the portfolio convergence loop.

        1. Get current target weights
        2. Get current positions from broker
        3. Enforce leverage constraints
        4. Check exposure limits (may block orders)
        5. Generate orders for delta
        6. Route orders
        """
        with self._data_lock:
            target_weights = self._target_weights
        if target_weights is None:
            return
        if self._broker is None or self._order_generator is None:
            return
        if self._order_router is None:
            return

        nav = self._broker.get_account_value()
        if nav <= 0:
            return

        # Scale NAV by strategy allocation if allocator is active
        if (
            self._strategy_allocator is not None
            and self._strategy_id in self._strategy_nav_allocations
        ):
            alloc_fraction = self._strategy_nav_allocations[self._strategy_id]
            if alloc_fraction <= 0:
                return  # zero allocation — no orders
            nav = nav * alloc_fraction

        current_positions = self._broker.get_positions()
        prices = pd.Series(dtype=float)
        tickers = list(target_weights.index)
        if tickers:
            md = self._broker.get_market_data(tickers)
            if not md.empty and "mid" in md.columns:
                prices = md["mid"]

        # --- Risk gate ---
        if self._risk_cascade is not None:
            # Unified cascade: kill_switch → drawdown → leverage → exposure
            cascade_result = self._risk_cascade.run_cascade(
                nav, target_weights, self._sector_map,
                self._factor_exposures,
            )
            if cascade_result.orders_blocked:
                action = "halt" if cascade_result.kill_switch_triggered else "exposure_breach"
                self._event_bus.publish(
                    Event(
                        event_type=EventType.RISK_CHECK,
                        payload={
                            "action": action,
                            "kill_switch": cascade_result.kill_switch_triggered,
                            "breaches": [
                                {
                                    "type": b.exposure_type,
                                    "current": b.current_value,
                                    "limit": b.limit,
                                }
                                for b in cascade_result.exposure_breaches
                            ],
                        },
                        source="convergence_loop",
                        priority=EventPriority.HIGH,
                    )
                )
                if cascade_result.kill_switch_triggered:
                    from quant_fund.infrastructure.system_state_machine import SystemState
                    self._state_machine.try_transition(
                        SystemState.RISK_HALT,
                        reason=f"Kill switch triggered at NAV={nav:.2f}",
                    )
                return
            working_weights = cascade_result.adjusted_weights
        else:
            # Fallback: inline risk checks (Module A)
            working_weights = target_weights

            if self._leverage_controller is not None:
                working_weights = self._leverage_controller.enforce(working_weights)

            if self._exposure_monitor is not None:
                breaches = self._exposure_monitor.check(
                    working_weights,
                    sector_map=self._sector_map,
                    factor_exposures=self._factor_exposures,
                )
                if breaches:
                    logger.warning(
                        "Exposure breach in convergence — orders blocked: %s",
                        [b.message for b in breaches],
                    )
                    self._event_bus.publish(
                        Event(
                            event_type=EventType.RISK_CHECK,
                            payload={
                                "action": "exposure_breach",
                                "breaches": [
                                    {
                                        "type": b.exposure_type,
                                        "current": b.current_value,
                                        "limit": b.limit,
                                    }
                                    for b in breaches
                                ],
                            },
                            source="convergence_loop",
                            priority=EventPriority.HIGH,
                        )
                    )
                    return

        # --- Capacity constraint: scale high-impact trades ---
        if self._capacity_model is not None and not prices.empty:
            # Compute trade notional per ticker
            current_weight = pd.Series(dtype=float)
            if nav > 0 and not current_positions.empty:
                current_value = current_positions * prices.reindex(
                    current_positions.index
                ).fillna(0)
                current_weight = current_value / nav

            weight_delta = working_weights.subtract(
                current_weight, fill_value=0.0
            ).abs()
            trade_notional = weight_delta * nav

            # Get ADV from market data
            adv_series = pd.Series(dtype=float)
            if tickers:
                md = self._broker.get_market_data(tickers)
                if not md.empty and "adv" in md.columns:
                    adv_series = md["adv"]

            vol = self._volatility_estimates
            if vol is None:
                vol = pd.Series(0.02, index=trade_notional.index)

            if not adv_series.empty:
                impact = self._capacity_model.estimate_impact_portfolio(
                    trade_notional, adv_series, vol,
                )
                # Scale down high-impact tickers by 50%
                for ticker in impact.index:
                    if impact[ticker] > self._max_impact_bps and ticker in working_weights.index:
                        midpoint = (
                            working_weights[ticker]
                            + current_weight.get(ticker, 0.0)
                        ) / 2.0
                        working_weights = working_weights.copy()
                        working_weights[ticker] = midpoint
                        logger.info(
                            "Capacity constraint: %s impact %.1f bps > %.1f, "
                            "scaling toward current weight",
                            ticker, impact[ticker], self._max_impact_bps,
                        )

        orders = self._order_generator.generate_orders(
            target_weights=working_weights,
            current_positions=current_positions,
            prices=prices,
            nav=nav,
            strategy_id=self._strategy_id,
        )

        if orders:
            acks = self._order_router.route_orders(orders)
            fills = sum(
                1
                for a in acks
                if a.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL)
            )
            logger.info(
                "Convergence: %d orders, %d fills", len(orders), fills
            )

            # Build order lookup for execution quality
            order_by_id = {o.order_id: o for o in orders}

            # Publish fill events and record execution quality
            for ack in acks:
                if ack.status in (
                    OrderStatus.FILLED,
                    OrderStatus.PARTIAL_FILL,
                ):
                    self._event_bus.publish(
                        Event(
                            event_type=EventType.ORDER_FILL,
                            payload={
                                "order_id": ack.order_id,
                                "status": ack.status.value,
                            },
                            source="convergence_loop",
                            priority=EventPriority.HIGH,
                        )
                    )

                    # Record execution quality
                    if self._execution_quality_monitor is not None:
                        order = order_by_id.get(ack.order_id)
                        if order is not None:
                            decision_price = prices.get(order.ticker, 0.0)
                            # Use mid price as proxy for fill price when
                            # actual fill data isn't directly on the ack
                            fill_price = decision_price
                            if self._broker is not None:
                                recent_fills = getattr(
                                    self._broker, "_fills", []
                                )
                                for f in reversed(recent_fills):
                                    if f.order_id == ack.order_id:
                                        fill_price = f.fill_price
                                        break
                            self._execution_quality_monitor.record_execution(
                                order_id=ack.order_id,
                                ticker=order.ticker,
                                side=order.side.value,
                                target_qty=order.quantity,
                                filled_qty=order.quantity,
                                decision_price=decision_price,
                                fill_price=fill_price,
                            )

    def get_execution_quality_summary(self):
        """Get aggregate execution quality metrics."""
        if self._execution_quality_monitor is None:
            return None
        return self._execution_quality_monitor.get_summary()

    # ------------------------------------------------------------------
    # Research → optimization pipeline
    # ------------------------------------------------------------------

    def _run_research_and_optimize(self) -> None:
        """Run the research → factor model → optimization → target weights chain.

        Mirrors PaperTradingRunner steps 3-6:
        1. Run research cycle to produce alpha scores
        2. Estimate factor exposures and covariance
        3. Build constraints
        4. Optimize portfolio
        5. Set target weights for convergence loop
        """
        if self._research_runner is None:
            return

        as_of = pd.Timestamp.now(tz="America/New_York")

        # Get market data for research — prefer historical cache (needed for
        # feature computation lookbacks) over single latest bar
        market_data = self._historical_data
        if market_data is None and self._live_data_adapter is not None:
            try:
                market_data = self._live_data_adapter.get_latest_bar()
            except Exception:
                logger.warning("Failed to get latest market data for research")

        # Step 1: Research cycle → alpha scores
        try:
            research_result = self._research_runner.run_cycle(
                as_of=as_of, market_data=market_data,
            )
            alpha_scores = research_result.alpha_scores
        except Exception:
            logger.exception("Research cycle failed")
            return

        if alpha_scores is None or alpha_scores.empty:
            logger.info("Research produced no alpha scores")
            return

        self._latest_alpha_scores = alpha_scores
        logger.info("Research produced alpha scores for %d tickers", len(alpha_scores))

        # Publish signal event
        self._event_bus.publish(
            Event(
                event_type=EventType.SIGNAL_GENERATED,
                payload={
                    "alpha_scores": alpha_scores.to_dict(),
                    "as_of": str(as_of),
                },
                source="research_runner",
                priority=EventPriority.NORMAL,
            )
        )

        # Step 2: Factor model estimation
        factor_exposures = self._factor_exposures
        factor_covariance = None

        if (
            self._factor_exposure_estimator is not None
            and self._factor_covariance_estimator is not None
            and self._stock_returns is not None
            and self._factor_returns is not None
        ):
            try:
                factor_exposures = self._factor_exposure_estimator.estimate(
                    returns=self._stock_returns,
                    factor_returns=self._factor_returns,
                    as_of=as_of,
                    sector_map=self._sector_map,
                )
                self._factor_exposures = factor_exposures

                factor_covariance = self._factor_covariance_estimator.estimate(
                    factor_returns=self._factor_returns,
                    as_of=as_of,
                )
            except Exception:
                logger.exception("Factor model estimation failed")

        # Step 3: Portfolio optimization
        if self._portfolio_optimizer is not None:
            try:
                constraints = None
                if self._constraint_engine is not None:
                    constraints = self._constraint_engine.build_constraints(
                        sector_map=self._sector_map,
                    )

                current_positions = pd.Series(dtype=float)
                if self._broker is not None:
                    current_positions = self._broker.get_positions()

                target_weights = self._portfolio_optimizer.optimize(
                    alpha_scores=alpha_scores,
                    factor_covariance=factor_covariance,
                    factor_exposures=factor_exposures,
                    constraints=constraints,
                    current_positions=current_positions,
                )

                with self._data_lock:
                    self._target_weights = target_weights
                logger.info(
                    "Optimization produced weights for %d tickers (gross=%.3f)",
                    len(target_weights),
                    target_weights.abs().sum(),
                )
            except Exception:
                logger.exception("Portfolio optimization failed")
        else:
            # No optimizer — use alpha scores as simple weights
            # Normalize to reasonable gross exposure
            total = alpha_scores.abs().sum()
            if total > 0:
                with self._data_lock:
                    self._target_weights = alpha_scores / total * 0.5
                logger.info(
                    "No optimizer — using normalized alpha scores as weights"
                )

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
        """Compute and cache strategy allocations via the injected allocator.

        Parameters
        ----------
        strategy_sharpes : dict
            Mapping of strategy_id → trailing Sharpe ratio.
        **kwargs
            Passed through to DynamicStrategyAllocator.allocate()
            (e.g. correlation_penalties, regime_adjustments).

        Returns
        -------
        dict
            Mapping of strategy_id → allocation weight.
        """
        if self._strategy_allocator is None:
            return {}
        allocations = self._strategy_allocator.allocate(
            strategy_sharpes, **kwargs
        )
        self._strategy_nav_allocations = allocations
        logger.info("Strategy allocations updated: %s", allocations)
        return allocations

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def _run_reconciliation(self) -> None:
        """Run a reconciliation cycle."""
        if self._reconciliation is None or self._broker is None:
            return

        internal_positions = pd.Series(dtype=float)
        if self._broker is not None:
            # For reconciliation, we compare our last-known positions
            # against what the broker reports
            internal_positions = self._broker.get_positions()

        nav = (
            self._pnl_dashboard.latest.nav
            if self._pnl_dashboard is not None
            and self._pnl_dashboard.latest is not None
            else 0.0
        )

        # Get open order IDs from router
        open_ids: Set[str] = set()
        if self._order_router is not None:
            open_ids = {
                r.acknowledgement.order_id
                for r in self._order_router.get_active_orders()
                if r.acknowledgement is not None
            }

        try:
            result = self._reconciliation.reconcile(
                internal_positions=internal_positions,
                internal_nav=nav,
                internal_open_order_ids=open_ids,
            )
        except Exception:
            logger.exception("Reconciliation failed")
            return

        if not result.all_matched:
            logger.warning("Reconciliation found discrepancies: %s", result.to_dict())
            self._event_bus.publish(
                Event(
                    event_type=EventType.RECONCILIATION,
                    payload=result.to_dict(),
                    source="reconciliation_engine",
                    priority=EventPriority.HIGH,
                )
            )

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    def _run_health_check(self) -> None:
        """Run system health check."""
        if self._health_monitor is None:
            return

        custom_checks = []

        # Broker connectivity
        if self._broker is not None:
            try:
                self._broker.get_account_value()
                custom_checks.append(
                    HealthCheck(
                        component="broker",
                        status="healthy",
                        message="Broker responding",
                    )
                )
            except Exception as e:
                custom_checks.append(
                    HealthCheck(
                        component="broker",
                        status="down",
                        message=f"Broker error: {e}",
                    )
                )

        # State machine
        custom_checks.append(
            HealthCheck(
                component="state_machine",
                status="healthy",
                message=f"State: {self._state_machine.state.value}",
            )
        )

        snapshot = self._health_monitor.run_health_check(
            custom_checks=custom_checks
        )

        self._event_bus.publish(
            Event(
                event_type=EventType.SYSTEM_HEALTH,
                payload={
                    "overall_status": snapshot.overall_status,
                    "uptime_s": snapshot.uptime_s,
                    "data_feed_lag_s": snapshot.data_feed_lag_s,
                },
                source="health_monitor",
                priority=EventPriority.LOW,
            )
        )

        # If system is down, transition to DATA_READY (suspend trading)
        if (
            snapshot.overall_status == "down"
            and self._state_machine.can_trade
        ):
            self._state_machine.try_transition(
                SystemState.DATA_READY,
                reason="Health check failed: system down",
            )

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> None:
        """Persist current state of all components."""
        positions = None
        if self._broker is not None:
            try:
                positions = self._broker.get_positions()
            except Exception:
                pass

        open_ids = None
        if self._order_router is not None:
            open_ids = [
                r.acknowledgement.order_id
                for r in self._order_router.get_active_orders()
                if r.acknowledgement is not None
            ]

        self._persistence.snapshot_all(
            kill_switch=self._kill_switch,
            drawdown_monitor=self._drawdown_monitor,
            pnl_dashboard=self._pnl_dashboard,
            state_machine=self._state_machine,
            positions=positions,
            alpha_scores=self._latest_alpha_scores,
            open_order_ids=open_ids,
        )

        # Periodic WAL checkpoint to prevent unbounded WAL file growth
        self._state_store.checkpoint()

    def _restore_state(self) -> None:
        """Restore state from the most recent snapshot."""
        results = self._persistence.restore_all(
            kill_switch=self._kill_switch,
            drawdown_monitor=self._drawdown_monitor,
            pnl_dashboard=self._pnl_dashboard,
            state_machine=self._state_machine,
        )
        logger.info("State restore results: %s", results)

        # Restore signal cache
        cached_scores = self._persistence.restore_signal_cache()
        if cached_scores is not None:
            self._latest_alpha_scores = cached_scores

        # Restore open order IDs for dedup
        open_ids = self._persistence.restore_open_orders()
        if open_ids:
            logger.info(
                "Restored %d open order IDs from previous session",
                len(open_ids),
            )

    # ------------------------------------------------------------------
    # Readiness checks
    # ------------------------------------------------------------------

    def _register_readiness_checks(self) -> None:
        """Register readiness checks for the trading-enabled transition."""

        def check_broker() -> ReadinessCheck:
            if self._broker is None:
                return ReadinessCheck(
                    name="broker", passed=False, message="No broker configured"
                )
            try:
                nav = self._broker.get_account_value()
                return ReadinessCheck(
                    name="broker",
                    passed=nav > 0,
                    message=f"NAV={nav:.2f}",
                )
            except Exception as e:
                return ReadinessCheck(
                    name="broker", passed=False, message=str(e)
                )

        def check_order_pipeline() -> ReadinessCheck:
            has_og = self._order_generator is not None
            has_or = self._order_router is not None
            passed = has_og and has_or
            return ReadinessCheck(
                name="order_pipeline",
                passed=passed,
                message=(
                    "OK"
                    if passed
                    else f"Missing: OG={has_og} OR={has_or}"
                ),
            )

        def check_risk() -> ReadinessCheck:
            has_ks = self._kill_switch is not None
            return ReadinessCheck(
                name="risk",
                passed=has_ks,
                message="OK" if has_ks else "No kill switch configured",
            )

        self._state_machine.register_readiness_check("broker", check_broker)
        self._state_machine.register_readiness_check(
            "order_pipeline", check_order_pipeline
        )
        self._state_machine.register_readiness_check("risk", check_risk)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except (ValueError, OSError):
            # Can't install signal handlers from non-main thread
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
