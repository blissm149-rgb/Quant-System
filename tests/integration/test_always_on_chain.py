"""Integration Chain 5: Always-On Architecture

Tests: EventBus → SystemStateMachine → StateStore →
StatePersistenceManager → recovery from crash
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
@pytest.mark.tier3
class TestEventBusToStateMachine:
    """Events on bus drive state machine transitions."""

    def test_events_reach_all_subscribers(self):
        """Events published on the bus reach all registered subscribers."""
        from quant_fund.infrastructure.event_bus import EventBus, Event, EventType

        bus = EventBus(synchronous=True)
        received_a = []
        received_b = []

        bus.subscribe("sub_a", lambda e: received_a.append(e), event_types={EventType.MARKET_DATA})
        bus.subscribe("sub_b", lambda e: received_b.append(e), event_types={EventType.MARKET_DATA})

        for i in range(10):
            bus.publish(Event(event_type=EventType.MARKET_DATA, payload={"seq": i}))

        assert len(received_a) == 10
        assert len(received_b) == 10

    def test_state_machine_transitions_via_events(self):
        """State machine transitions driven by event handling."""
        from quant_fund.infrastructure.event_bus import EventBus, Event, EventType
        from quant_fund.infrastructure.system_state_machine import SystemStateMachine, SystemState

        bus = EventBus(synchronous=True)
        sm = SystemStateMachine()

        def on_data_ready(event):
            if sm.state == SystemState.INITIALIZING:
                sm.transition_to(SystemState.DATA_READY)

        bus.subscribe("data_handler", on_data_ready, event_types={EventType.MARKET_DATA})

        # Publish data event → should trigger transition
        bus.publish(Event(event_type=EventType.MARKET_DATA, payload={"status": "ready"}))

        assert sm.state == SystemState.DATA_READY

    def test_state_changes_published_as_events(self):
        """State machine transitions can publish events back to bus."""
        from quant_fund.infrastructure.event_bus import EventBus, Event, EventType
        from quant_fund.infrastructure.system_state_machine import SystemStateMachine, SystemState

        bus = EventBus(synchronous=True)
        sm = SystemStateMachine()
        state_events = []

        bus.subscribe("state_listener", lambda e: state_events.append(e),
                      event_types={EventType.STATE_CHANGE})

        # Register callback to publish state change events
        def on_transition(transition):
            bus.publish(Event(
                event_type=EventType.STATE_CHANGE,
                payload={"from": transition.from_state.value, "to": transition.to_state.value},
            ))

        sm.on_transition(on_transition)

        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)

        assert len(state_events) == 2


@pytest.mark.integration
@pytest.mark.tier3
class TestStatePersistenceIntegration:
    """State snapshots saved and restored correctly."""

    def test_save_and_restore_kill_switch(self):
        """Kill switch state persists across save/restore."""
        from quant_fund.infrastructure.state_store import StateStore
        from quant_fund.infrastructure.state_persistence_manager import StatePersistenceManager
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        store = StateStore(":memory:")
        persistence = StatePersistenceManager(store)

        # Save state with updated peak
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000.0})
        ks.update_peak(1_200_000.0)
        persistence.save_kill_switch_state(ks)

        # Restore into a fresh instance
        ks2 = KillSwitch({"drawdown_limit": 0.20})
        restored = persistence.restore_kill_switch_state(ks2)

        assert restored is True
        store.close()

    def test_save_and_restore_state_machine(self):
        """System state machine persists across save/restore."""
        from quant_fund.infrastructure.state_store import StateStore
        from quant_fund.infrastructure.state_persistence_manager import StatePersistenceManager
        from quant_fund.infrastructure.system_state_machine import SystemStateMachine, SystemState

        store = StateStore(":memory:")
        persistence = StatePersistenceManager(store)

        # Advance state machine
        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        persistence.save_system_state(sm)

        # Restore
        sm2 = SystemStateMachine()
        restored = persistence.restore_system_state(sm2)

        assert restored is True
        assert sm2.state == SystemState.TRADING_ENABLED
        store.close()

    def test_snapshot_all_and_restore_all(self):
        """Full system snapshot and restore cycle."""
        from quant_fund.infrastructure.state_store import StateStore
        from quant_fund.infrastructure.state_persistence_manager import StatePersistenceManager
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.infrastructure.system_state_machine import SystemStateMachine, SystemState

        store = StateStore(":memory:")
        persistence = StatePersistenceManager(store)

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)
        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)

        persistence.snapshot_all(kill_switch=ks, state_machine=sm)

        # Restore
        ks2 = KillSwitch({"drawdown_limit": 0.20})
        sm2 = SystemStateMachine()
        results = persistence.restore_all(kill_switch=ks2, state_machine=sm2)

        assert sm2.state == SystemState.DATA_READY
        store.close()

    def test_positions_round_trip(self):
        """Portfolio positions saved and restored correctly."""
        from quant_fund.infrastructure.state_store import StateStore
        from quant_fund.infrastructure.state_persistence_manager import StatePersistenceManager

        store = StateStore(":memory:")
        persistence = StatePersistenceManager(store)

        positions = pd.Series(
            [100.0, -50.0, 200.0, -75.0, 150.0],
            index=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
        )
        persistence.save_positions(positions)

        restored = persistence.restore_positions()
        assert restored is not None
        pd.testing.assert_series_equal(positions, restored, check_dtype=False)
        store.close()


@pytest.mark.integration
@pytest.mark.tier3
class TestReconciliationIntegration:
    """Reconciliation detects discrepancies between internal and broker state."""

    def test_reconciliation_detects_mismatch(self):
        """Position mismatch between internal and broker is detected."""
        from quant_fund.execution.reconciliation_engine import ReconciliationEngine
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_abstraction_layer import Order, OrderSide, OrderType

        broker = SimulationBroker({"initial_cash": 1_000_000.0})
        from tests.conftest import STANDARD_MARKET_DATA
        broker.set_market_data(STANDARD_MARKET_DATA)

        # Execute a trade so broker has positions
        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, order_id="REC-001",
        )
        broker.submit_order(order)

        # Internal position says 0 AAPL (mismatch)
        internal = pd.Series({"AAPL": 0.0})

        recon = ReconciliationEngine(broker)
        result = recon.reconcile(internal)

        # Should detect the discrepancy
        assert not result.all_matched or result.positions_matched is False

    def test_reconciliation_no_mismatch_when_aligned(self):
        """No discrepancy when internal matches broker."""
        from quant_fund.execution.reconciliation_engine import ReconciliationEngine
        from quant_fund.broker_interface.simulation_broker import SimulationBroker

        broker = SimulationBroker({"initial_cash": 1_000_000.0})
        from tests.conftest import STANDARD_MARKET_DATA
        broker.set_market_data(STANDARD_MARKET_DATA)

        # No trades → both internal and broker have empty positions
        internal = pd.Series(dtype=float)

        recon = ReconciliationEngine(broker)
        result = recon.reconcile(internal)

        assert result.all_matched or result.positions_matched
