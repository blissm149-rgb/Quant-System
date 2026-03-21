"""Unit tests for TradingEngine.

TESTING_PLAN.md Section 3.19 — always-on event-driven trading loop.
"""

import pandas as pd
import pytest

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.infrastructure.system_state_machine import SystemState
from quant_fund.main.trading_engine import TradingEngine
from quant_fund.monitoring.alerting_system import AlertingSystem
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.monitoring.system_health_monitor import SystemHealthMonitor
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from tests.conftest import STANDARD_MARKET_DATA


@pytest.mark.unit
@pytest.mark.tier2
class TestTradingEngine:
    """TradingEngine — lifecycle, component injection, convergence."""

    @pytest.fixture
    def broker(self):
        b = SimulationBroker(config={"initial_cash": 1_000_000})
        b.set_market_data(STANDARD_MARKET_DATA)
        return b

    @pytest.fixture
    def engine(self, broker):
        e = TradingEngine(config={"synchronous": True})
        e.inject_components(
            broker=broker,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
            kill_switch=KillSwitch(config={"drawdown_threshold": 0.20}),
            pnl_dashboard=PnLDashboard(),
            alerting=AlertingSystem(),
            health_monitor=SystemHealthMonitor(),
        )
        return e

    def test_initial_state(self, engine):
        assert engine.state == SystemState.INITIALIZING

    def test_inject_components(self, engine):
        """Components are injected successfully."""
        assert engine._broker is not None
        assert engine._order_generator is not None
        assert engine._kill_switch is not None

    def test_state_machine_accessible(self, engine):
        assert engine.state_machine is not None

    def test_event_bus_accessible(self, engine):
        assert engine.event_bus is not None

    def test_update_target_weights(self, engine):
        weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        engine.update_target_weights(weights)
        assert engine._target_weights is not None
        assert len(engine._target_weights) == 2

    def test_request_shutdown(self, engine):
        engine.request_shutdown()
        assert engine.is_running is False

    def test_reconciliation_auto_created(self, engine):
        """Reconciliation engine auto-created when broker is injected."""
        assert engine._reconciliation is not None
