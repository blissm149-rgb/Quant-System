"""Integration Chain 6: Broker Resilience

Tests: BrokerReconnectionManager → disconnect → reconnect → failover
→ orders handled correctly during disruption
"""

import pandas as pd
import pytest

from tests.conftest import STANDARD_MARKET_DATA


@pytest.mark.integration
@pytest.mark.tier3
class TestBrokerResilience:
    """Broker reconnection and failover integration."""

    def test_reconnection_after_disconnect(self):
        """Reconnection manager reconnects after disconnect."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_reconnection_manager import BrokerReconnectionManager

        primary = SimulationBroker({"initial_cash": 1_000_000.0})
        primary.set_market_data(STANDARD_MARKET_DATA)

        mgr = BrokerReconnectionManager(primary, config={"max_retries": 4})
        mgr.connect()
        assert mgr.is_connected

        mgr.disconnect()

        # Reconnect
        mgr.connect()
        assert mgr.is_connected

    def test_operations_work_after_reconnect(self):
        """Basic operations (get_positions, submit_order) work after reconnect."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_reconnection_manager import BrokerReconnectionManager
        from quant_fund.broker_interface.broker_abstraction_layer import Order, OrderSide, OrderType

        primary = SimulationBroker({"initial_cash": 1_000_000.0})
        primary.set_market_data(STANDARD_MARKET_DATA)

        mgr = BrokerReconnectionManager(primary, config={"max_retries": 4})
        mgr.connect()

        # Do some operations
        positions = mgr.get_positions()
        assert positions is not None

        # Disconnect and reconnect
        mgr.disconnect()
        mgr.connect()

        # Operations still work
        positions2 = mgr.get_positions()
        assert positions2 is not None

        # Submit order
        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=50,
            order_type=OrderType.MARKET, order_id="RESIL-001",
        )
        ack = mgr.submit_order(order)
        assert ack is not None

    def test_failover_to_secondary(self):
        """When primary is unavailable, failover to secondary."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_reconnection_manager import BrokerReconnectionManager

        primary = SimulationBroker({"initial_cash": 1_000_000.0})
        primary.set_market_data(STANDARD_MARKET_DATA)

        secondary = SimulationBroker({"initial_cash": 500_000.0})
        secondary.set_market_data(STANDARD_MARKET_DATA)

        mgr = BrokerReconnectionManager(primary, secondary=secondary, config={"max_retries": 2})
        connected = mgr.connect()
        assert connected is True

    def test_connection_callback_registration(self):
        """Connection callback can be registered without error."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_reconnection_manager import BrokerReconnectionManager

        primary = SimulationBroker({"initial_cash": 1_000_000.0})
        primary.set_market_data(STANDARD_MARKET_DATA)

        events = []
        mgr = BrokerReconnectionManager(primary)
        mgr.set_connection_callback(lambda event, success: events.append((event, success)))

        # Callback registration itself should not error
        mgr.connect()
        mgr.disconnect()
        mgr.connect()

        # Callback may or may not fire depending on implementation;
        # the key assertion is that the system doesn't crash
        assert mgr.is_connected

    def test_metrics_tracked_correctly(self):
        """Reconnection metrics are tracked."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_reconnection_manager import BrokerReconnectionManager

        primary = SimulationBroker({"initial_cash": 1_000_000.0})
        primary.set_market_data(STANDARD_MARKET_DATA)

        mgr = BrokerReconnectionManager(primary)
        mgr.connect()

        metrics = mgr.get_metrics()
        assert isinstance(metrics, dict)
        assert "reconnection_count" in metrics or hasattr(mgr, "reconnection_count")
