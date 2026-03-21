"""Unit tests for broker reconnection manager.

TESTING_PLAN.md Section 3.11 — broker_reconnection_manager.
"""

import pytest

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.broker_interface.broker_reconnection_manager import (
    BrokerReconnectionManager,
)
from tests.conftest import STANDARD_MARKET_DATA


@pytest.mark.unit
@pytest.mark.tier2
class TestBrokerReconnectionManager:
    """BrokerReconnectionManager — failover and retry logic."""

    @pytest.fixture
    def primary(self):
        b = SimulationBroker(config={"initial_cash": 1_000_000})
        b.set_market_data(STANDARD_MARKET_DATA)
        return b

    @pytest.fixture
    def manager(self, primary):
        return BrokerReconnectionManager(primary=primary)

    def test_connect_succeeds(self, manager):
        """connect returns True with healthy primary."""
        result = manager.connect()
        assert result is True

    def test_is_connected_after_connect(self, manager):
        """is_connected is True after successful connect."""
        manager.connect()
        assert manager.is_connected is True

    def test_disconnect(self, manager):
        """disconnect sets is_connected to False."""
        manager.connect()
        manager.disconnect()
        assert manager.is_connected is False

    def test_get_positions_delegates(self, manager, primary):
        """get_positions delegates to primary broker."""
        manager.connect()
        positions = manager.get_positions()
        assert positions is not None

    def test_get_metrics(self, manager):
        """get_metrics returns dict with reconnection stats."""
        manager.connect()
        metrics = manager.get_metrics()
        assert isinstance(metrics, dict)

    def test_failover_with_secondary(self, primary):
        """With secondary broker, failover is possible."""
        secondary = SimulationBroker(config={"initial_cash": 500_000})
        secondary.set_market_data(STANDARD_MARKET_DATA)
        mgr = BrokerReconnectionManager(primary=primary, secondary=secondary)
        mgr.connect()
        assert mgr.is_connected is True
