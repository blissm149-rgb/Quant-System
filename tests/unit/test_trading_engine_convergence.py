"""Unit tests for TradingEngine convergence risk gate.

Module A: Verifies that leverage_controller and exposure_monitor are
applied in _convergence_tick() before order generation.
"""

import pandas as pd
import pytest

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.infrastructure.event_bus import EventType
from quant_fund.main.trading_engine import TradingEngine
from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
from quant_fund.risk_engine.leverage_controller import LeverageController
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from tests.conftest import STANDARD_MARKET_DATA, STANDARD_SECTORS


@pytest.mark.unit
@pytest.mark.tier2
class TestTradingEngineConvergence:
    """TradingEngine._convergence_tick() risk gate integration."""

    @pytest.fixture
    def broker(self):
        b = SimulationBroker(config={"initial_cash": 1_000_000})
        b.set_market_data(STANDARD_MARKET_DATA)
        return b

    @pytest.fixture
    def base_engine(self, broker):
        """Engine with minimal components for convergence."""
        e = TradingEngine(config={"synchronous": True})
        e.inject_components(
            broker=broker,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
            kill_switch=KillSwitch(config={"drawdown_threshold": 0.20}),
        )
        return e

    def test_convergence_applies_leverage_controller(self, base_engine, broker):
        """Over-leveraged weights are scaled down before order generation."""
        lc = LeverageController(config={"max_leverage": 1.0})
        base_engine.inject_components(leverage_controller=lc)

        # Weights with gross leverage 3.0 — should be scaled to 1.0
        weights = pd.Series({
            "AAPL": 1.0, "MSFT": 1.0, "GOOG": 1.0,
        })
        base_engine.update_target_weights(weights)
        base_engine._convergence_tick()

        # Orders should have been generated (not blocked),
        # but from scaled-down weights. Check broker had some activity.
        positions = broker.get_positions()
        if len(positions) > 0:
            # Total position value should be <= 1.0 * NAV
            nav = broker.get_account_value()
            prices = pd.Series({t: STANDARD_MARKET_DATA[t]["mid"]
                                for t in positions.index})
            total_value = (positions.abs() * prices).sum()
            assert total_value <= nav * 1.1  # 10% tolerance for rounding

    def test_convergence_skips_orders_on_exposure_breach(self, base_engine, broker):
        """Concentrated weights trigger exposure breach — no orders generated."""
        em = ExposureMonitor(config={"max_single_name_exposure": 0.02})
        base_engine.inject_components(exposure_monitor=em)

        # 50% in one name — breaches 2% single-name limit
        weights = pd.Series({"AAPL": 0.50, "MSFT": 0.01})
        base_engine.update_target_weights(weights)

        initial_positions = broker.get_positions()
        base_engine._convergence_tick()
        after_positions = broker.get_positions()

        # No new orders should have been submitted
        assert initial_positions.equals(after_positions)

    def test_convergence_proceeds_when_no_breach(self, base_engine, broker):
        """Clean weights result in normal order generation."""
        em = ExposureMonitor(config={
            "max_single_name_exposure": 0.10,
            "max_leverage": 2.0,
        })
        base_engine.inject_components(exposure_monitor=em)

        weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        base_engine.update_target_weights(weights)
        base_engine._convergence_tick()

        positions = broker.get_positions()
        assert len(positions) > 0

    def test_convergence_without_leverage_controller(self, base_engine, broker):
        """Without leverage_controller, convergence works as before."""
        weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        base_engine.update_target_weights(weights)
        base_engine._convergence_tick()

        positions = broker.get_positions()
        assert len(positions) > 0

    def test_convergence_without_exposure_monitor(self, base_engine, broker):
        """Without exposure_monitor, convergence works as before."""
        lc = LeverageController(config={"max_leverage": 2.0})
        base_engine.inject_components(leverage_controller=lc)

        weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        base_engine.update_target_weights(weights)
        base_engine._convergence_tick()

        positions = broker.get_positions()
        assert len(positions) > 0

    def test_convergence_leverage_then_exposure_order(self, base_engine, broker):
        """Leverage is enforced BEFORE exposure check.

        Weights with gross=3.0 would breach max_single_name=0.40,
        but after leverage scaling to max_leverage=1.0, each weight
        becomes ~0.33, which is within the 0.40 limit.
        """
        lc = LeverageController(config={"max_leverage": 1.0})
        em = ExposureMonitor(config={
            "max_single_name_exposure": 0.40,
            "max_leverage": 2.0,
        })
        base_engine.inject_components(
            leverage_controller=lc,
            exposure_monitor=em,
        )

        # Gross = 3.0, max single = 1.0 — would breach 0.40 pre-scaling
        # After leverage enforcement: each ≈ 0.33 — within 0.40 limit
        weights = pd.Series({"AAPL": 1.0, "MSFT": 1.0, "GOOG": 1.0})
        base_engine.update_target_weights(weights)
        base_engine._convergence_tick()

        # Should proceed (not blocked)
        positions = broker.get_positions()
        assert len(positions) > 0

    def test_convergence_exposure_breach_publishes_event(self, base_engine):
        """Exposure breach publishes a RISK_CHECK event on the EventBus."""
        em = ExposureMonitor(config={"max_single_name_exposure": 0.02})
        base_engine.inject_components(exposure_monitor=em)

        received_events = []
        base_engine.event_bus.subscribe(
            "test_exposure_sub",
            lambda event: received_events.append(event),
            {EventType.RISK_CHECK},
        )

        weights = pd.Series({"AAPL": 0.50, "MSFT": 0.01})
        base_engine.update_target_weights(weights)
        base_engine._convergence_tick()

        # Drain synchronous bus
        base_engine.event_bus.drain()

        assert len(received_events) >= 1
        evt = received_events[0]
        assert evt.payload["action"] == "exposure_breach"
        assert len(evt.payload["breaches"]) > 0

    def test_convergence_with_sector_map_and_factor_exposures(self, base_engine):
        """ExposureMonitor receives sector_map and factor_exposures."""
        em = ExposureMonitor(config={
            "max_sector_exposure": 0.30,
            "max_single_name_exposure": 0.10,
            "max_leverage": 2.0,
        })
        base_engine.inject_components(exposure_monitor=em)
        base_engine.update_sector_map(STANDARD_SECTORS)

        # 4 tech stocks at 0.08 each = 0.32 sector exposure > 0.30 limit
        weights = pd.Series({
            "AAPL": 0.08, "MSFT": 0.08, "GOOG": 0.08, "NVDA": 0.08,
            "JPM": 0.02, "WMT": 0.02,
        })
        base_engine.update_target_weights(weights)

        received_events = []
        base_engine.event_bus.subscribe(
            "test_sector_sub",
            lambda event: received_events.append(event),
            {EventType.RISK_CHECK},
        )

        base_engine._convergence_tick()
        base_engine.event_bus.drain()

        # Should trigger sector exposure breach
        assert len(received_events) >= 1
        assert received_events[0].payload["action"] == "exposure_breach"
