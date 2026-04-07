"""Engine health — reconciliation, health checks, state persistence.

Extracted from TradingEngine to reduce the monolithic orchestrator.
Contains operational health functions: reconciliation with broker,
periodic health checks, state snapshot/restore, and readiness checks.
"""

import logging
from typing import Optional, Set

import pandas as pd

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
from quant_fund.monitoring.system_health_monitor import (
    HealthCheck,
    SystemHealthMonitor,
)

logger = logging.getLogger(__name__)


def run_reconciliation(
    *,
    reconciliation,
    broker,
    pnl_dashboard=None,
    order_router=None,
    event_bus: EventBus,
) -> None:
    """Run a reconciliation cycle against the broker."""
    if reconciliation is None or broker is None:
        return

    internal_positions = broker.get_positions()

    nav = (
        pnl_dashboard.latest.nav
        if pnl_dashboard is not None
        and pnl_dashboard.latest is not None
        else 0.0
    )

    open_ids: Set[str] = set()
    if order_router is not None:
        open_ids = {
            r.acknowledgement.order_id
            for r in order_router.get_active_orders()
            if r.acknowledgement is not None
        }

    try:
        result = reconciliation.reconcile(
            internal_positions=internal_positions,
            internal_nav=nav,
            internal_open_order_ids=open_ids,
        )
    except Exception:
        logger.exception("Reconciliation failed")
        return

    if not result.all_matched:
        logger.warning("Reconciliation found discrepancies: %s", result.to_dict())
        event_bus.publish(
            Event(
                event_type=EventType.RECONCILIATION,
                payload=result.to_dict(),
                source="reconciliation_engine",
                priority=EventPriority.HIGH,
            )
        )


def run_health_check(
    *,
    health_monitor: Optional[SystemHealthMonitor],
    broker,
    state_machine: SystemStateMachine,
    event_bus: EventBus,
) -> None:
    """Run system health check and publish results."""
    if health_monitor is None:
        return

    custom_checks = []

    # Broker connectivity
    if broker is not None:
        try:
            broker.get_account_value()
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
            message=f"State: {state_machine.state.value}",
        )
    )

    snapshot = health_monitor.run_health_check(custom_checks=custom_checks)

    event_bus.publish(
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
    if snapshot.overall_status == "down" and state_machine.can_trade:
        state_machine.try_transition(
            SystemState.DATA_READY,
            reason="Health check failed: system down",
        )


def take_snapshot(
    *,
    persistence: StatePersistenceManager,
    state_store: StateStore,
    broker=None,
    order_router=None,
    kill_switch=None,
    drawdown_monitor=None,
    pnl_dashboard=None,
    state_machine=None,
    latest_alpha_scores=None,
    research_runner=None,
) -> None:
    """Persist current state of all components."""
    positions = None
    if broker is not None:
        try:
            positions = broker.get_positions()
        except Exception:
            pass

    open_ids = None
    if order_router is not None:
        open_ids = [
            r.acknowledgement.order_id
            for r in order_router.get_active_orders()
            if r.acknowledgement is not None
        ]

    persistence.snapshot_all(
        kill_switch=kill_switch,
        drawdown_monitor=drawdown_monitor,
        pnl_dashboard=pnl_dashboard,
        state_machine=state_machine,
        positions=positions,
        alpha_scores=latest_alpha_scores,
        open_order_ids=open_ids,
    )

    # Persist research runner retrain counter
    if research_runner is not None:
        try:
            persistence.save_research_state(research_runner)
        except Exception:
            logger.debug("Failed to save research state", exc_info=True)

    # Periodic WAL checkpoint to prevent unbounded WAL file growth
    state_store.checkpoint()


def restore_state(
    *,
    persistence: StatePersistenceManager,
    kill_switch=None,
    drawdown_monitor=None,
    pnl_dashboard=None,
    state_machine=None,
    research_runner=None,
) -> tuple:
    """Restore state from the most recent snapshot.

    Returns (results, cached_scores, open_ids).
    """
    results = persistence.restore_all(
        kill_switch=kill_switch,
        drawdown_monitor=drawdown_monitor,
        pnl_dashboard=pnl_dashboard,
        state_machine=state_machine,
    )
    logger.info("State restore results: %s", results)

    cached_scores = persistence.restore_signal_cache()
    open_ids = persistence.restore_open_orders()
    if open_ids:
        logger.info(
            "Restored %d open order IDs from previous session", len(open_ids),
        )

    if research_runner is not None:
        persistence.restore_research_state(research_runner)

    return results, cached_scores, open_ids


def register_readiness_checks(
    *,
    state_machine: SystemStateMachine,
    broker,
    order_generator,
    order_router,
    kill_switch,
) -> None:
    """Register readiness checks for the trading-enabled transition."""

    def check_broker() -> ReadinessCheck:
        if broker is None:
            return ReadinessCheck(
                name="broker", passed=False, message="No broker configured"
            )
        try:
            nav = broker.get_account_value()
            return ReadinessCheck(
                name="broker", passed=nav > 0, message=f"NAV={nav:.2f}",
            )
        except Exception as e:
            return ReadinessCheck(name="broker", passed=False, message=str(e))

    def check_order_pipeline() -> ReadinessCheck:
        has_og = order_generator is not None
        has_or = order_router is not None
        passed = has_og and has_or
        return ReadinessCheck(
            name="order_pipeline",
            passed=passed,
            message="OK" if passed else f"Missing: OG={has_og} OR={has_or}",
        )

    def check_risk() -> ReadinessCheck:
        has_ks = kill_switch is not None
        return ReadinessCheck(
            name="risk",
            passed=has_ks,
            message="OK" if has_ks else "No kill switch configured",
        )

    state_machine.register_readiness_check("broker", check_broker)
    state_machine.register_readiness_check("order_pipeline", check_order_pipeline)
    state_machine.register_readiness_check("risk", check_risk)
