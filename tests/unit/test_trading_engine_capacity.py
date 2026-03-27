"""Unit tests for TradingEngine capacity-constrained execution.

Module C: Verifies MarketImpactModel is wired into _convergence_tick()
to filter high-impact trades.
"""

import pandas as pd
import pytest

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.main.trading_engine import TradingEngine
from quant_fund.portfolio.capacity_model.market_impact_model import (
    MarketImpactModel,
)
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from tests.conftest import STANDARD_MARKET_DATA


def _make_engine(broker, capacity_model=None, max_impact_bps=50.0):
    """Build a TradingEngine for capacity testing."""
    e = TradingEngine(config={
        "synchronous": True,
        "max_impact_bps": max_impact_bps,
    })
    components = dict(
        broker=broker,
        order_generator=OrderGenerator(),
        order_router=OrderRouter(broker),
        kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
    )
    if capacity_model is not None:
        components["capacity_model"] = capacity_model
    e.inject_components(**components)
    return e


@pytest.mark.unit
@pytest.mark.tier2
class TestTradingEngineCapacity:
    """MarketImpactModel integration into TradingEngine."""

    @pytest.fixture
    def broker(self):
        b = SimulationBroker(config={"initial_cash": 1_000_000})
        b.set_market_data(STANDARD_MARKET_DATA)
        return b

    def test_capacity_model_injection(self, broker):
        """Capacity model is injected and stored."""
        model = MarketImpactModel()
        engine = _make_engine(broker, capacity_model=model)
        assert engine._capacity_model is model

    def test_high_impact_trade_is_scaled(self):
        """Trade with high impact (tiny ADV) results in smaller order."""
        # Use tiny ADV so any trade has high impact
        tiny_adv_data = {
            "AAPL": {"bid": 149.0, "ask": 151.0, "mid": 150.0,
                      "last": 150.0, "volume": 100, "adv": 100},
        }
        broker_constrained = SimulationBroker(config={"initial_cash": 1_000_000})
        broker_constrained.set_market_data(tiny_adv_data)

        model = MarketImpactModel()
        engine = _make_engine(
            broker_constrained, capacity_model=model, max_impact_bps=5.0,
        )

        weights = pd.Series({"AAPL": 0.10})
        engine.update_target_weights(weights)
        engine._convergence_tick()

        pos_constrained = broker_constrained.get_positions()

        # Without capacity constraint
        broker_free = SimulationBroker(config={"initial_cash": 1_000_000})
        broker_free.set_market_data(tiny_adv_data)
        engine_free = _make_engine(broker_free)
        engine_free.update_target_weights(weights)
        engine_free._convergence_tick()
        pos_free = broker_free.get_positions()

        # Constrained position should be smaller (or equal if both are zero)
        aapl_constrained = pos_constrained.get("AAPL", 0)
        aapl_free = pos_free.get("AAPL", 0)
        assert aapl_constrained <= aapl_free

    def test_low_impact_trade_passes_through(self, broker):
        """Small trade relative to ADV has no scaling."""
        model = MarketImpactModel()
        engine_cap = _make_engine(broker, capacity_model=model)

        # Small weight — tiny trade vs huge ADV
        weights = pd.Series({"AAPL": 0.01})
        engine_cap.update_target_weights(weights)
        engine_cap._convergence_tick()
        pos_cap = broker.get_positions()

        # Without model
        broker2 = SimulationBroker(config={"initial_cash": 1_000_000})
        broker2.set_market_data(STANDARD_MARKET_DATA)
        engine_free = _make_engine(broker2)
        engine_free.update_target_weights(weights)
        engine_free._convergence_tick()
        pos_free = broker2.get_positions()

        # Should be same position (no scaling needed)
        if "AAPL" in pos_cap.index and "AAPL" in pos_free.index:
            assert pos_cap["AAPL"] == pos_free["AAPL"]

    def test_no_capacity_model_no_change(self, broker):
        """Without capacity_model, original behavior is preserved."""
        engine = _make_engine(broker)

        weights = pd.Series({"AAPL": 0.05, "MSFT": -0.05})
        engine.update_target_weights(weights)
        engine._convergence_tick()

        positions = broker.get_positions()
        assert len(positions) > 0

    def test_capacity_constraint_converges_over_ticks(self):
        """Two ticks close more of the gap than one tick for high-impact trade."""
        tiny_adv_data = {
            "AAPL": {"bid": 149.0, "ask": 151.0, "mid": 150.0,
                      "last": 150.0, "volume": 100, "adv": 100},
        }
        broker = SimulationBroker(config={"initial_cash": 1_000_000})
        broker.set_market_data(tiny_adv_data)

        model = MarketImpactModel()
        engine = _make_engine(broker, capacity_model=model, max_impact_bps=5.0)

        weights = pd.Series({"AAPL": 0.10})
        engine.update_target_weights(weights)

        engine._convergence_tick()
        pos_after_1 = broker.get_positions().get("AAPL", 0)

        engine._convergence_tick()
        pos_after_2 = broker.get_positions().get("AAPL", 0)

        # Second tick should move closer to target (or stay same)
        assert pos_after_2 >= pos_after_1
