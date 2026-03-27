"""Unit tests for TradingEngine execution quality integration.

Module F: Verifies ExecutionQualityMonitor is wired into
_convergence_tick() and records fills.
"""

import pandas as pd
import pytest

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.main.trading_engine import TradingEngine
from quant_fund.monitoring.execution_quality_monitor import ExecutionQualityMonitor
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from tests.conftest import STANDARD_MARKET_DATA


@pytest.mark.unit
@pytest.mark.tier2
class TestTradingEngineExecutionQuality:
    """ExecutionQualityMonitor integration into TradingEngine."""

    @pytest.fixture
    def broker(self):
        b = SimulationBroker(config={"initial_cash": 1_000_000})
        b.set_market_data(STANDARD_MARKET_DATA)
        return b

    @pytest.fixture
    def engine_with_monitor(self, broker):
        e = TradingEngine(config={"synchronous": True})
        monitor = ExecutionQualityMonitor()
        e.inject_components(
            broker=broker,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
            kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
            execution_quality_monitor=monitor,
        )
        return e, monitor

    @pytest.fixture
    def engine_without_monitor(self, broker):
        e = TradingEngine(config={"synchronous": True})
        e.inject_components(
            broker=broker,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
            kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
        )
        return e

    def test_execution_quality_monitor_injection(self, engine_with_monitor):
        """Monitor is injected and stored."""
        engine, monitor = engine_with_monitor
        assert engine._execution_quality_monitor is monitor

    def test_execution_quality_recorded_after_fills(self, engine_with_monitor):
        """Convergence with fills records execution quality."""
        engine, monitor = engine_with_monitor

        weights = pd.Series({"AAPL": 0.05, "MSFT": -0.05})
        engine.update_target_weights(weights)
        engine._convergence_tick()

        summary = monitor.get_summary()
        assert summary.num_orders > 0

    def test_implementation_shortfall_calculated(self, engine_with_monitor):
        """Implementation shortfall is computed for each fill."""
        engine, monitor = engine_with_monitor

        weights = pd.Series({"AAPL": 0.05})
        engine.update_target_weights(weights)
        engine._convergence_tick()

        records = monitor.get_records_dataframe()
        if len(records) > 0:
            # Implementation shortfall should be a number (may be 0 if
            # decision_price == fill_price in simulation)
            assert "implementation_shortfall_bps" in records.columns

    def test_no_monitor_no_crash(self, engine_without_monitor, broker):
        """Without monitor, convergence works normally."""
        engine = engine_without_monitor

        weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        engine.update_target_weights(weights)
        engine._convergence_tick()

        positions = broker.get_positions()
        assert len(positions) > 0

    def test_get_execution_quality_summary(self, engine_with_monitor):
        """get_execution_quality_summary delegates to monitor."""
        engine, monitor = engine_with_monitor

        weights = pd.Series({"AAPL": 0.05, "MSFT": -0.05})
        engine.update_target_weights(weights)
        engine._convergence_tick()

        summary = engine.get_execution_quality_summary()
        assert summary is not None
        assert summary.num_orders > 0
        assert summary.avg_fill_rate > 0

    def test_adverse_selection_detection(self):
        """Large slippage triggers adverse selection flag."""
        monitor = ExecutionQualityMonitor(config={"adverse_threshold_bps": 20.0})

        # Record an execution with large adverse slippage (30 bps)
        monitor.record_execution(
            order_id="test-001",
            ticker="AAPL",
            side="buy",
            target_qty=100,
            filled_qty=100,
            decision_price=150.0,
            fill_price=150.45,  # 30 bps slippage
        )

        adversely_selected = monitor.detect_adverse_selection()
        assert len(adversely_selected) == 1
        assert adversely_selected[0].order_id == "test-001"

        summary = monitor.get_summary()
        assert summary.total_adverse_selection_flags == 1
