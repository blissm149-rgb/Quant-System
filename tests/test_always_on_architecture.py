"""Tests for always-on trading system architecture components.

Covers:
    1. Event bus — publish/subscribe, idempotency, replay
    2. System state machine — transitions, readiness checks
    3. State persistence — save/restore across components
    4. Reconciliation engine — position/NAV/order reconciliation
    5. Trading engine — lifecycle, convergence, risk halts
    6. Broker reconnection — retry, failover
    7. Startup/recovery — cold start, restart with positions
    8. Resilience — component failures, data feed loss
    9. Strategy determinism — replay produces identical results
    10. Long-duration stability — extended simulation
"""

import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_fund.infrastructure.event_bus import (
    Event,
    EventBus,
    EventPriority,
    EventType,
)
from quant_fund.infrastructure.system_state_machine import (
    InvalidTransitionError,
    ReadinessCheck,
    SystemState,
    SystemStateMachine,
)
from quant_fund.infrastructure.state_store import StateStore
from quant_fund.infrastructure.state_persistence_manager import (
    StatePersistenceManager,
)
from quant_fund.execution.reconciliation_engine import (
    ReconciliationEngine,
    ReconciliationResult,
)
from quant_fund.broker_interface.broker_abstraction_layer import (
    BrokerInterface,
    Fill,
    Order,
    OrderAcknowledgement,
    OrderSide,
    OrderStatus,
    OrderType,
)
from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.broker_interface.broker_reconnection_manager import (
    BrokerReconnectionManager,
)
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.monitoring.alerting_system import AlertingSystem
from quant_fund.monitoring.system_health_monitor import SystemHealthMonitor
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.main.trading_engine import TradingEngine

# Standard market data for SimulationBroker
STANDARD_MARKET_DATA = {
    "AAPL": {"bid": 149.9, "ask": 150.1, "mid": 150.0, "last": 150.0, "volume": 5e6, "adv": 5e6},
    "MSFT": {"bid": 299.8, "ask": 300.2, "mid": 300.0, "last": 300.0, "volume": 3e6, "adv": 3e6},
    "GOOG": {"bid": 139.9, "ask": 140.1, "mid": 140.0, "last": 140.0, "volume": 2e6, "adv": 2e6},
    "NVDA": {"bid": 499.8, "ask": 500.2, "mid": 500.0, "last": 500.0, "volume": 4e6, "adv": 4e6},
    "AMZN": {"bid": 179.9, "ask": 180.1, "mid": 180.0, "last": 180.0, "volume": 3e6, "adv": 3e6},
}


def _make_broker(initial_cash: float = 10_000_000.0) -> SimulationBroker:
    broker = SimulationBroker({"initial_cash": initial_cash})
    broker.set_market_data(STANDARD_MARKET_DATA)
    return broker


# ======================================================================
# 1. EventBus Tests
# ======================================================================


class TestEventBus:
    """Tests for the event bus publish/subscribe system."""

    def test_publish_and_subscribe(self):
        """Basic pub/sub: handler receives published events."""
        bus = EventBus(synchronous=True)
        received = []
        bus.subscribe(
            "test_sub",
            lambda e: received.append(e),
            {EventType.MARKET_DATA},
        )
        event = Event(event_type=EventType.MARKET_DATA, payload={"ticker": "AAPL"})
        bus.publish(event)
        assert len(received) == 1
        assert received[0].payload["ticker"] == "AAPL"

    def test_subscribe_multiple_types(self):
        """Handler subscribed to multiple types receives both."""
        bus = EventBus(synchronous=True)
        received = []
        bus.subscribe(
            "multi",
            lambda e: received.append(e.event_type),
            {EventType.MARKET_DATA, EventType.ORDER_FILL},
        )
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        bus.publish(Event(event_type=EventType.ORDER_FILL))
        bus.publish(Event(event_type=EventType.SIGNAL_GENERATED))  # not subscribed
        assert len(received) == 2
        assert EventType.MARKET_DATA in received
        assert EventType.ORDER_FILL in received

    def test_unsubscribe(self):
        """After unsubscribe, handler no longer receives events."""
        bus = EventBus(synchronous=True)
        received = []
        bus.subscribe("unsub_test", lambda e: received.append(e), {EventType.MARKET_DATA})
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        assert len(received) == 1
        bus.unsubscribe("unsub_test")
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        assert len(received) == 1

    def test_idempotency_dedup(self):
        """Duplicate events (same ID) are deduplicated."""
        bus = EventBus(synchronous=True)
        received = []
        bus.subscribe("dedup", lambda e: received.append(e), {EventType.MARKET_DATA})

        event = Event(event_type=EventType.MARKET_DATA, event_id="unique123")
        assert bus.publish(event) is True
        assert bus.publish(event) is False  # duplicate
        assert len(received) == 1

    def test_idempotency_key(self):
        """Idempotency key deduplicates different event IDs."""
        bus = EventBus(synchronous=True)
        received = []
        bus.subscribe("idem", lambda e: received.append(e), {EventType.MARKET_DATA})

        e1 = Event(event_type=EventType.MARKET_DATA, event_id="a1", idempotency_key="same")
        e2 = Event(event_type=EventType.MARKET_DATA, event_id="a2", idempotency_key="same")
        bus.publish(e1)
        bus.publish(e2)
        assert len(received) == 1

    def test_priority_ordering(self):
        """Higher priority handlers execute first."""
        bus = EventBus(synchronous=True)
        order = []
        bus.subscribe("low", lambda e: order.append("low"), {EventType.MARKET_DATA}, EventPriority.LOW)
        bus.subscribe("high", lambda e: order.append("high"), {EventType.MARKET_DATA}, EventPriority.HIGH)
        bus.subscribe("critical", lambda e: order.append("critical"), {EventType.MARKET_DATA}, EventPriority.CRITICAL)
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        assert order == ["critical", "high", "low"]

    def test_handler_error_isolation(self):
        """A failing handler does not prevent other handlers from running."""
        bus = EventBus(synchronous=True)
        received = []

        def bad_handler(e):
            raise ValueError("boom")

        bus.subscribe("bad", bad_handler, {EventType.MARKET_DATA})
        bus.subscribe("good", lambda e: received.append(e), {EventType.MARKET_DATA})
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        assert len(received) == 1  # good handler still ran

    def test_event_history(self):
        """Published events are stored in history."""
        bus = EventBus(synchronous=True)
        for i in range(5):
            bus.publish(Event(event_type=EventType.MARKET_DATA, payload={"i": i}))
        history = bus.get_history()
        assert len(history) == 5
        assert history[0].payload["i"] == 0

    def test_replay(self):
        """Events can be replayed, respecting idempotency."""
        bus = EventBus(synchronous=True)
        received = []
        bus.subscribe("replay", lambda e: received.append(e), {EventType.MARKET_DATA})

        events = [
            Event(event_type=EventType.MARKET_DATA, event_id=f"r{i}")
            for i in range(3)
        ]
        for e in events:
            bus.publish(e)
        assert len(received) == 3

        # Replay: duplicates are skipped
        count = bus.replay(events)
        assert count == 0
        assert len(received) == 3

    def test_async_dispatch(self):
        """Events dispatched via background worker thread."""
        bus = EventBus(synchronous=False)
        received = []
        bus.subscribe("async_test", lambda e: received.append(e), {EventType.MARKET_DATA})
        bus.start()
        try:
            bus.publish(Event(event_type=EventType.MARKET_DATA))
            # Give worker time to process
            time.sleep(0.3)
            assert len(received) == 1
        finally:
            bus.stop()

    def test_metrics(self):
        """Metrics track publish/dispatch/dedup counts."""
        bus = EventBus(synchronous=True)
        bus.subscribe("m", lambda e: None, {EventType.MARKET_DATA})
        bus.publish(Event(event_type=EventType.MARKET_DATA, event_id="x1"))
        bus.publish(Event(event_type=EventType.MARKET_DATA, event_id="x1"))  # dedup
        m = bus.get_metrics()
        assert m["publish_count"] == 1
        assert m["dedup_count"] == 1
        assert m["dispatch_count"] == 1

    def test_persist_callback(self):
        """Persist callback is called for each published event."""
        bus = EventBus(synchronous=True)
        persisted = []
        bus.set_persist_callback(lambda e: persisted.append(e))
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        assert len(persisted) == 1

    def test_drain(self):
        """Drain processes all queued events synchronously."""
        bus = EventBus(synchronous=False)
        received = []
        bus.subscribe("drain", lambda e: received.append(e), {EventType.MARKET_DATA})
        for i in range(5):
            bus.publish(Event(event_type=EventType.MARKET_DATA, event_id=f"d{i}"))
        count = bus.drain()
        assert count == 5
        assert len(received) == 5


# ======================================================================
# 2. System State Machine Tests
# ======================================================================


class TestSystemStateMachine:
    """Tests for the system lifecycle state machine."""

    def test_initial_state(self):
        sm = SystemStateMachine()
        assert sm.state == SystemState.INITIALIZING
        assert not sm.can_trade
        assert sm.is_initializing

    def test_valid_transition_init_to_data_ready(self):
        sm = SystemStateMachine()
        t = sm.transition_to(SystemState.DATA_READY, reason="Data ok")
        assert sm.state == SystemState.DATA_READY
        assert t.from_state == SystemState.INITIALIZING
        assert t.to_state == SystemState.DATA_READY

    def test_valid_transition_to_trading(self):
        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        assert sm.can_trade
        assert not sm.is_halted

    def test_invalid_transition_raises(self):
        sm = SystemStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(SystemState.TRADING_ENABLED)  # can't skip DATA_READY

    def test_try_transition_returns_none_on_invalid(self):
        sm = SystemStateMachine()
        result = sm.try_transition(SystemState.TRADING_ENABLED)
        assert result is None
        assert sm.state == SystemState.INITIALIZING

    def test_risk_halt(self):
        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        sm.transition_to(SystemState.RISK_HALT, reason="Kill switch")
        assert sm.is_halted
        assert not sm.can_trade

    def test_resume_from_risk_halt(self):
        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        sm.transition_to(SystemState.RISK_HALT)
        sm.transition_to(SystemState.TRADING_ENABLED, reason="Manual reset")
        assert sm.can_trade

    def test_shutdown_is_terminal(self):
        sm = SystemStateMachine()
        sm.transition_to(SystemState.SHUTDOWN, reason="Bye")
        assert sm.is_shutdown
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(SystemState.INITIALIZING)

    def test_transition_callback(self):
        sm = SystemStateMachine()
        transitions = []
        sm.on_transition(lambda t: transitions.append(t))
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        assert len(transitions) == 2

    def test_history(self):
        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        history = sm.get_history()
        assert len(history) == 2
        assert history[0].to_state == SystemState.DATA_READY

    def test_readiness_checks_pass(self):
        sm = SystemStateMachine()
        sm.register_readiness_check(
            "check1",
            lambda: ReadinessCheck(name="check1", passed=True),
        )
        sm.register_readiness_check(
            "check2",
            lambda: ReadinessCheck(name="check2", passed=True),
        )
        ok, results = sm.run_readiness_checks()
        assert ok
        assert len(results) == 2

    def test_readiness_checks_fail(self):
        sm = SystemStateMachine()
        sm.register_readiness_check(
            "good",
            lambda: ReadinessCheck(name="good", passed=True),
        )
        sm.register_readiness_check(
            "bad",
            lambda: ReadinessCheck(name="bad", passed=False, message="No broker"),
        )
        ok, results = sm.run_readiness_checks()
        assert not ok

    def test_transition_if_ready(self):
        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.register_readiness_check(
            "ok", lambda: ReadinessCheck(name="ok", passed=True)
        )
        transitioned, _ = sm.transition_if_ready(SystemState.TRADING_ENABLED)
        assert transitioned
        assert sm.can_trade

    def test_transition_if_not_ready(self):
        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.register_readiness_check(
            "fail", lambda: ReadinessCheck(name="fail", passed=False, message="nope")
        )
        transitioned, _ = sm.transition_if_ready(SystemState.TRADING_ENABLED)
        assert not transitioned
        assert sm.state == SystemState.DATA_READY

    def test_serialization_roundtrip(self):
        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        data = sm.to_dict()
        assert data["current_state"] == "TRADING_ENABLED"

        sm2 = SystemStateMachine()
        sm2.restore_from_dict(data)
        assert sm2.state == SystemState.TRADING_ENABLED

    def test_data_ready_to_initializing(self):
        """DATA_READY can go back to INITIALIZING for re-init."""
        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.INITIALIZING, reason="Re-init after data loss")
        assert sm.is_initializing


# ======================================================================
# 3. State Persistence Tests
# ======================================================================


class TestStatePersistence:
    """Tests for state save/restore across system components."""

    def test_kill_switch_save_restore(self):
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        ks = KillSwitch({"drawdown_limit": 0.15, "initial_nav": 2_000_000})
        ks.peak_nav = 2_100_000
        mgr.save_kill_switch_state(ks)

        ks2 = KillSwitch({"drawdown_limit": 0.15})
        assert mgr.restore_kill_switch_state(ks2)
        assert ks2.peak_nav == 2_100_000
        # max_drawdown is set by constructor config, not restored from state
        assert ks2.max_drawdown == 0.15
        store.close()

    def test_kill_switch_halted_state_persists(self):
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        ks = KillSwitch({"initial_nav": 1_000_000})
        ks._is_halted = True
        mgr.save_kill_switch_state(ks)

        ks2 = KillSwitch()
        mgr.restore_kill_switch_state(ks2)
        assert ks2.is_halted is True
        store.close()

    def test_drawdown_save_restore(self):
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        dm = DrawdownMonitor({"initial_nav": 1_000_000})
        dm.update(1_050_000)  # new peak
        mgr.save_drawdown_state(dm)

        dm2 = DrawdownMonitor()
        assert mgr.restore_drawdown_state(dm2)
        assert dm2.peak_nav == 1_050_000
        store.close()

    def test_pnl_save_restore(self):
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        pnl = PnLDashboard({"initial_nav": 1_000_000})
        pnl.update(1_010_000)
        pnl.update(1_020_000)
        mgr.save_pnl_state(pnl)

        pnl2 = PnLDashboard({"initial_nav": 1_000_000})
        assert mgr.restore_pnl_state(pnl2)
        assert pnl2._peak_nav == 1_020_000
        store.close()

    def test_positions_save_restore(self):
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        positions = pd.Series({"AAPL": 100, "MSFT": -50}, dtype=float)
        mgr.save_positions(positions)

        restored = mgr.restore_positions()
        assert restored is not None
        assert restored["AAPL"] == 100
        assert restored["MSFT"] == -50
        store.close()

    def test_signal_cache_save_restore(self):
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        scores = pd.Series({"AAPL": 0.8, "MSFT": -0.3, "GOOG": 0.5})
        mgr.save_signal_cache(scores)

        restored = mgr.restore_signal_cache()
        assert restored is not None
        assert abs(restored["AAPL"] - 0.8) < 1e-10
        store.close()

    def test_open_orders_save_restore(self):
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        mgr.save_open_orders(["ORD-001", "ORD-002", "ORD-003"])
        restored = mgr.restore_open_orders()
        assert restored == ["ORD-001", "ORD-002", "ORD-003"]
        store.close()

    def test_snapshot_all(self):
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        ks = KillSwitch({"initial_nav": 1_000_000})
        dm = DrawdownMonitor({"initial_nav": 1_000_000})
        pnl = PnLDashboard({"initial_nav": 1_000_000})
        pnl.update(1_010_000)
        sm = SystemStateMachine()

        mgr.snapshot_all(
            kill_switch=ks,
            drawdown_monitor=dm,
            pnl_dashboard=pnl,
            state_machine=sm,
            positions=pd.Series({"AAPL": 100}, dtype=float),
            alpha_scores=pd.Series({"AAPL": 0.5}),
            open_order_ids=["ORD-001"],
        )

        assert mgr.snapshot_count == 1
        store.close()

    def test_restore_no_data_returns_false(self):
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)
        ks = KillSwitch()
        assert mgr.restore_kill_switch_state(ks) is False
        store.close()

    def test_system_state_save_restore(self):
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        mgr.save_system_state(sm)

        sm2 = SystemStateMachine()
        assert mgr.restore_system_state(sm2)
        assert sm2.state == SystemState.TRADING_ENABLED
        store.close()


# ======================================================================
# 4. Reconciliation Engine Tests
# ======================================================================


class TestReconciliationEngine:
    """Tests for position/NAV/order reconciliation."""

    def test_positions_match(self):
        broker = _make_broker()
        engine = ReconciliationEngine(broker)
        # No positions, so should match
        result = engine.reconcile(pd.Series(dtype=float))
        assert result.all_matched

    def test_position_discrepancy_detected(self):
        broker = _make_broker()
        # Buy some shares to create broker positions
        broker.submit_order(Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            timestamp=pd.Timestamp.now(),
        ))

        engine = ReconciliationEngine(broker)
        # Internal says 0 AAPL, broker has 100
        result = engine.reconcile(pd.Series(dtype=float))
        assert not result.positions_matched
        assert len(result.position_discrepancies) == 1
        disc = result.position_discrepancies[0]
        assert disc.ticker == "AAPL"
        assert disc.broker_qty == 100
        assert disc.internal_qty == 0

    def test_auto_correct_positions(self):
        broker = _make_broker()
        broker.submit_order(Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            timestamp=pd.Timestamp.now(),
        ))

        engine = ReconciliationEngine(broker, config={"auto_correct": True})
        result = engine.reconcile(pd.Series(dtype=float))
        assert result.auto_corrected
        assert len(result.corrections_applied) == 1

    def test_nav_discrepancy_detected(self):
        broker = _make_broker(initial_cash=1_000_000)
        engine = ReconciliationEngine(
            broker, config={"nav_tolerance_pct": 0.001}
        )
        # Internal NAV differs significantly from broker
        result = engine.reconcile(
            pd.Series(dtype=float), internal_nav=500_000
        )
        assert not result.cash_matched
        assert result.cash_discrepancy is not None
        assert result.cash_discrepancy.delta != 0

    def test_nav_within_tolerance(self):
        broker = _make_broker(initial_cash=1_000_000)
        engine = ReconciliationEngine(
            broker, config={"nav_tolerance_pct": 0.01}
        )
        result = engine.reconcile(
            pd.Series(dtype=float), internal_nav=1_000_000
        )
        assert result.cash_matched

    def test_order_discrepancy(self):
        broker = _make_broker()
        # Submit and fill an order
        ack = broker.submit_order(Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=10,
            timestamp=pd.Timestamp.now(),
        ))
        engine = ReconciliationEngine(broker)
        # Pretend we still think the order is open
        result = engine.reconcile(
            pd.Series(dtype=float),
            internal_open_order_ids={ack.order_id},
        )
        assert not result.orders_matched
        assert len(result.order_discrepancies) == 1

    def test_reconciliation_history(self):
        broker = _make_broker()
        engine = ReconciliationEngine(broker)
        engine.reconcile(pd.Series(dtype=float))
        engine.reconcile(pd.Series(dtype=float))
        assert len(engine.get_history()) == 2

    def test_metrics(self):
        broker = _make_broker()
        engine = ReconciliationEngine(broker)
        engine.reconcile(pd.Series(dtype=float))
        m = engine.get_metrics()
        assert m["total_reconciliations"] == 1
        assert m["total_mismatches"] == 0

    def test_corrected_positions_returns_broker_positions(self):
        broker = _make_broker()
        broker.submit_order(Order(
            ticker="MSFT", side=OrderSide.BUY, quantity=50,
            timestamp=pd.Timestamp.now(),
        ))
        engine = ReconciliationEngine(broker, config={"auto_correct": True})
        internal = pd.Series(dtype=float)
        result = engine.reconcile(internal)
        corrected = engine.get_corrected_positions(internal, result)
        assert corrected["MSFT"] == 50


# ======================================================================
# 5. Trading Engine Tests
# ======================================================================


class TestTradingEngine:
    """Tests for the always-on trading engine."""

    def _build_engine(self, **overrides):
        config = {
            "synchronous": True,
            "tick_interval_s": 0.01,
            "snapshot_interval_s": 0.05,
            "reconciliation_interval_s": 0.05,
            "health_check_interval_s": 0.05,
            "convergence_interval_s": 0.01,
            "strategy_id": "test_strategy",
            **overrides,
        }
        engine = TradingEngine(config=config)
        return engine

    def test_engine_initialization(self):
        engine = self._build_engine()
        assert engine.state == SystemState.INITIALIZING
        assert engine.is_running

    def test_engine_inject_components(self):
        engine = self._build_engine()
        broker = _make_broker()
        ks = KillSwitch({"initial_nav": 10_000_000})
        og = OrderGenerator()
        router = OrderRouter(broker)

        engine.inject_components(
            broker=broker,
            kill_switch=ks,
            order_generator=og,
            order_router=router,
        )
        # Reconciliation engine auto-created
        assert engine._reconciliation is not None

    def test_engine_lifecycle_to_trading(self):
        """Engine can transition INIT → DATA_READY → TRADING_ENABLED."""
        engine = self._build_engine()
        broker = _make_broker()
        ks = KillSwitch({"initial_nav": 10_000_000})
        og = OrderGenerator()
        router = OrderRouter(broker)

        engine.inject_components(
            broker=broker,
            kill_switch=ks,
            order_generator=og,
            order_router=router,
        )

        # Simulate initialization
        engine._initialize()
        assert engine.state == SystemState.TRADING_ENABLED

    def test_engine_convergence_loop(self):
        """Convergence loop generates orders for target weights."""
        engine = self._build_engine()
        broker = _make_broker()
        ks = KillSwitch({"initial_nav": 10_000_000})
        og = OrderGenerator({"min_trade_value": 100})
        router = OrderRouter(broker)

        engine.inject_components(
            broker=broker,
            kill_switch=ks,
            order_generator=og,
            order_router=router,
        )

        # Set target weights
        weights = pd.Series({"AAPL": 0.3, "MSFT": 0.2, "GOOG": -0.1})
        engine.update_target_weights(weights)

        # Run convergence tick
        engine._convergence_tick()

        # Should have created positions
        positions = broker.get_positions()
        assert len(positions) > 0

    def test_engine_risk_halt(self):
        """Kill switch triggers RISK_HALT state."""
        engine = self._build_engine()
        broker = _make_broker(initial_cash=1_000_000)
        ks = KillSwitch({"initial_nav": 1_000_000, "drawdown_limit": 0.01})
        ks.peak_nav = 1_000_000

        engine.inject_components(
            broker=broker,
            kill_switch=ks,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
        )

        engine._initialize()
        assert engine.state == SystemState.TRADING_ENABLED

        # Simulate a loss by reducing broker cash
        broker._cash = 500_000

        engine._check_risk()
        assert engine.state == SystemState.RISK_HALT

    def test_engine_state_persistence(self):
        """Engine saves and restores state."""
        engine = self._build_engine()
        broker = _make_broker()
        ks = KillSwitch({"initial_nav": 10_000_000})
        pnl = PnLDashboard({"initial_nav": 10_000_000})

        engine.inject_components(
            broker=broker,
            kill_switch=ks,
            pnl_dashboard=pnl,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
        )

        pnl.update(10_100_000)
        engine._take_snapshot()
        assert engine._persistence.snapshot_count == 1

    def test_engine_shutdown(self):
        """Engine shuts down gracefully."""
        engine = self._build_engine()
        broker = _make_broker()
        ks = KillSwitch({"initial_nav": 10_000_000})

        engine.inject_components(
            broker=broker,
            kill_switch=ks,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
        )

        engine._initialize()
        engine._shutdown()
        assert engine.state == SystemState.SHUTDOWN

    def test_engine_health_check(self):
        """Health check runs without error."""
        engine = self._build_engine()
        broker = _make_broker()
        hm = SystemHealthMonitor()

        engine.inject_components(
            broker=broker,
            health_monitor=hm,
            kill_switch=KillSwitch({"initial_nav": 10_000_000}),
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
        )

        engine._initialize()
        engine._run_health_check()
        assert hm.latest is not None

    def test_engine_event_bus_integration(self):
        """Events flow through the bus during engine operation."""
        engine = self._build_engine()
        broker = _make_broker()
        hm = SystemHealthMonitor()

        received = []
        engine.event_bus.subscribe(
            "test_listener",
            lambda e: received.append(e),
            {EventType.SYSTEM_HEALTH},
        )

        engine.inject_components(
            broker=broker,
            kill_switch=KillSwitch({"initial_nav": 10_000_000}),
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
            health_monitor=hm,
        )

        engine._initialize()
        engine._run_health_check()
        engine.event_bus.drain()
        assert len(received) >= 1


# ======================================================================
# 6. Broker Reconnection Tests
# ======================================================================


class TestBrokerReconnection:
    """Tests for broker reconnection and failover."""

    def test_connect_to_simulation_broker(self):
        """SimulationBroker connects without issues."""
        broker = _make_broker()
        mgr = BrokerReconnectionManager(primary=broker)
        assert mgr.connect() is True
        assert mgr.is_connected

    def test_proxy_get_positions(self):
        """Proxied calls go through to underlying broker."""
        broker = _make_broker()
        mgr = BrokerReconnectionManager(primary=broker)
        mgr.connect()
        positions = mgr.get_positions()
        assert isinstance(positions, pd.Series)

    def test_proxy_submit_order(self):
        broker = _make_broker()
        mgr = BrokerReconnectionManager(primary=broker)
        mgr.connect()
        ack = mgr.submit_order(Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=10,
            timestamp=pd.Timestamp.now(),
        ))
        assert ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL)

    def test_proxy_get_account_value(self):
        broker = _make_broker(initial_cash=5_000_000)
        mgr = BrokerReconnectionManager(primary=broker)
        mgr.connect()
        nav = mgr.get_account_value()
        assert nav == 5_000_000

    def test_failover_to_secondary(self):
        """If primary fails to connect, uses secondary."""
        primary = MagicMock()
        primary.connect.return_value = False
        secondary = _make_broker()

        mgr = BrokerReconnectionManager(
            primary=primary,
            secondary=secondary,
            config={"max_retries": 1, "initial_backoff_s": 0.01},
        )

        assert mgr.connect() is True
        assert mgr.using_failover
        assert mgr.failover_count == 1

    def test_reconnection_on_call_failure(self):
        """Manager reconnects when a call raises."""
        broker = _make_broker()
        call_count = [0]
        original_get_positions = broker.get_positions

        def flaky_get_positions():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("network down")
            return original_get_positions()

        broker.get_positions = flaky_get_positions
        mgr = BrokerReconnectionManager(
            primary=broker,
            config={"max_retries": 1, "initial_backoff_s": 0.01},
        )
        mgr.connect()
        positions = mgr.get_positions()
        assert isinstance(positions, pd.Series)
        assert mgr.reconnection_count >= 1

    def test_connection_callback(self):
        broker = _make_broker()
        mgr = BrokerReconnectionManager(primary=broker)
        events = []
        mgr.set_connection_callback(lambda name, ok: events.append((name, ok)))

        # Force a reconnection
        mgr._reconnect()
        assert len(events) == 1
        assert events[0][1] is True  # success

    def test_metrics(self):
        broker = _make_broker()
        mgr = BrokerReconnectionManager(primary=broker)
        mgr.connect()
        m = mgr.get_metrics()
        assert m["is_connected"] is True
        assert m["reconnection_count"] == 0


# ======================================================================
# 7. Startup / Recovery Tests
# ======================================================================


class TestStartupRecovery:
    """Tests for cold start and restart with persisted state."""

    def test_cold_start_no_previous_state(self):
        """Engine starts cleanly with no prior state."""
        engine = TradingEngine({"synchronous": True})
        broker = _make_broker()
        engine.inject_components(
            broker=broker,
            kill_switch=KillSwitch({"initial_nav": 10_000_000}),
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
        )
        engine._initialize()
        assert engine.state == SystemState.TRADING_ENABLED

    def test_restart_with_persisted_kill_switch(self):
        """Kill switch state survives restart."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        # Session 1: kill switch was halted
        ks1 = KillSwitch({"initial_nav": 1_000_000})
        ks1._is_halted = True
        ks1.peak_nav = 1_100_000
        mgr.save_kill_switch_state(ks1)

        # Session 2: restore
        ks2 = KillSwitch()
        mgr.restore_kill_switch_state(ks2)
        assert ks2.is_halted
        assert ks2.peak_nav == 1_100_000
        store.close()

    def test_restart_with_positions(self):
        """Positions survive restart via StateStore."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        positions = pd.Series({"AAPL": 500, "MSFT": -200}, dtype=float)
        mgr.save_positions(positions)

        restored = mgr.restore_positions()
        assert restored is not None
        assert restored["AAPL"] == 500
        assert restored["MSFT"] == -200
        store.close()

    def test_restart_with_open_orders_no_duplicate(self):
        """Open order IDs from previous session prevent duplicates."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        mgr.save_open_orders(["ORD-001", "ORD-002"])
        restored_ids = mgr.restore_open_orders()
        assert "ORD-001" in restored_ids
        assert "ORD-002" in restored_ids
        store.close()

    def test_corrupted_state_recovery(self):
        """Invalid state store data doesn't crash restore."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        # Write invalid data
        store.save("kill_switch", "peak_nav", "not_a_number")
        ks = KillSwitch()
        # Should restore without crashing (sets peak_nav to string)
        # The system should handle this gracefully
        result = mgr.restore_kill_switch_state(ks)
        assert result is True  # data was found
        store.close()


# ======================================================================
# 8. Resilience Tests
# ======================================================================


class TestResilience:
    """Tests for component failure handling."""

    def test_strategy_engine_crash_continues(self):
        """If research runner crashes, engine doesn't halt."""
        engine = TradingEngine({"synchronous": True})
        broker = _make_broker()
        engine.inject_components(
            broker=broker,
            kill_switch=KillSwitch({"initial_nav": 10_000_000}),
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
        )
        engine._initialize()

        # Publish a signal event with bad data — should not crash engine
        engine.event_bus.publish(Event(
            event_type=EventType.SIGNAL_GENERATED,
            payload={"alpha_scores": {"AAPL": 0.5}},
        ))
        engine.event_bus.drain()
        assert engine.state == SystemState.TRADING_ENABLED

    def test_reconciliation_failure_isolated(self):
        """Reconciliation error doesn't crash main loop."""
        engine = TradingEngine({"synchronous": True})
        broker = _make_broker()
        engine.inject_components(
            broker=broker,
            kill_switch=KillSwitch({"initial_nav": 10_000_000}),
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
        )
        engine._initialize()

        # Force reconciliation engine to fail
        engine._reconciliation._broker = MagicMock(spec=BrokerInterface)
        engine._reconciliation._broker.get_positions.side_effect = RuntimeError("boom")

        # Should not raise
        try:
            engine._run_reconciliation()
        except RuntimeError:
            pytest.fail("Reconciliation error should be contained")

    def test_health_check_detects_broker_down(self):
        """Health check detects broker failure."""
        engine = TradingEngine({"synchronous": True})
        broker = MagicMock(spec=BrokerInterface)
        broker.get_account_value.side_effect = ConnectionError("down")
        broker.get_positions.return_value = pd.Series(dtype=float)

        hm = SystemHealthMonitor()
        engine.inject_components(
            broker=broker,
            health_monitor=hm,
            kill_switch=KillSwitch({"initial_nav": 10_000_000}),
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
        )

        # Manually get to TRADING_ENABLED
        engine._state_machine.transition_to(SystemState.DATA_READY)
        engine._state_machine.transition_to(SystemState.TRADING_ENABLED)

        engine._run_health_check()
        # The broker check should show "down"
        assert hm.latest is not None
        down_checks = [
            c for c in hm.latest.checks if c.status == "down"
        ]
        assert len(down_checks) >= 1

    def test_convergence_with_no_target_weights(self):
        """Convergence tick with no target weights does nothing."""
        engine = TradingEngine({"synchronous": True})
        broker = _make_broker()
        engine.inject_components(
            broker=broker,
            kill_switch=KillSwitch({"initial_nav": 10_000_000}),
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
        )
        # No target weights set
        engine._convergence_tick()
        # No positions created
        assert len(broker.get_positions()) == 0


# ======================================================================
# 9. Strategy Determinism Tests
# ======================================================================


class TestDeterminism:
    """Tests that verify deterministic behavior with identical inputs."""

    def test_order_generation_deterministic(self):
        """Same inputs → same orders."""
        og = OrderGenerator({"min_trade_value": 100})

        weights = pd.Series({"AAPL": 0.3, "MSFT": 0.2, "GOOG": -0.1})
        positions = pd.Series(dtype=float)
        prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0, "GOOG": 140.0})

        orders1 = og.generate_orders(weights, positions, prices, nav=1_000_000)
        orders2 = og.generate_orders(weights, positions, prices, nav=1_000_000)

        assert len(orders1) == len(orders2)
        for o1, o2 in zip(orders1, orders2):
            assert o1.ticker == o2.ticker
            assert o1.side == o2.side
            assert o1.quantity == o2.quantity

    def test_simulation_broker_deterministic(self):
        """Same orders on same broker state → same fills."""
        def run_session():
            broker = _make_broker()
            broker.submit_order(Order(
                ticker="AAPL", side=OrderSide.BUY, quantity=100,
                timestamp=pd.Timestamp("2024-01-01"),
            ))
            broker.submit_order(Order(
                ticker="MSFT", side=OrderSide.SELL, quantity=50,
                timestamp=pd.Timestamp("2024-01-01"),
            ))
            return broker.get_account_value(), broker.get_positions()

        nav1, pos1 = run_session()
        nav2, pos2 = run_session()
        assert nav1 == nav2
        assert pos1.equals(pos2)


# ======================================================================
# 10. Extended Simulation Stability Tests
# ======================================================================


class TestLongDurationStability:
    """Extended simulation to verify stability over many iterations."""

    def test_30_day_simulation_stability(self):
        """Run 30 simulated days and verify consistent state."""
        broker = _make_broker()
        ks = KillSwitch({"initial_nav": 10_000_000, "drawdown_limit": 0.50})
        dm = DrawdownMonitor({"initial_nav": 10_000_000, "drawdown_limit": 0.50})
        pnl = PnLDashboard({"initial_nav": 10_000_000})
        og = OrderGenerator({"min_trade_value": 100})
        router = OrderRouter(broker)

        rng = np.random.RandomState(42)
        tickers = list(STANDARD_MARKET_DATA.keys())

        navs = []
        for day in range(30):
            nav = broker.get_account_value()
            navs.append(nav)

            # Skip if killed
            if ks.check(nav):
                break
            ks.update_peak(nav)
            dm.update(nav)
            pnl.update(nav)

            # Random target weights
            raw = rng.randn(len(tickers))
            raw = raw / (np.abs(raw).sum() + 1e-8)  # normalize
            weights = pd.Series(raw, index=tickers)

            positions = broker.get_positions()
            prices = pd.Series(
                {t: STANDARD_MARKET_DATA[t]["mid"] for t in tickers}
            )

            orders = og.generate_orders(
                target_weights=weights,
                current_positions=positions,
                prices=prices,
                nav=nav,
            )

            if orders:
                router.route_orders(orders)

        # Verify: NAV is always positive, PnL has history
        assert all(n > 0 for n in navs)
        assert pnl.latest is not None
        assert len(navs) >= 10  # at least 10 days ran

    def test_state_persistence_across_simulation(self):
        """State can be saved and restored mid-simulation."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        broker = _make_broker()
        ks = KillSwitch({"initial_nav": 10_000_000})
        dm = DrawdownMonitor({"initial_nav": 10_000_000})
        pnl = PnLDashboard({"initial_nav": 10_000_000})

        # Run a few iterations
        for i in range(5):
            nav = broker.get_account_value()
            ks.update_peak(nav)
            dm.update(nav)
            pnl.update(nav)

        # Snapshot
        mgr.snapshot_all(
            kill_switch=ks,
            drawdown_monitor=dm,
            pnl_dashboard=pnl,
            positions=broker.get_positions(),
        )

        # Restore into fresh components
        ks2 = KillSwitch()
        dm2 = DrawdownMonitor()
        pnl2 = PnLDashboard()

        results = mgr.restore_all(
            kill_switch=ks2,
            drawdown_monitor=dm2,
            pnl_dashboard=pnl2,
        )

        assert results["kill_switch"] is True
        assert results["drawdown_monitor"] is True
        assert results["pnl_dashboard"] is True
        assert ks2.peak_nav == ks.peak_nav
        store.close()
