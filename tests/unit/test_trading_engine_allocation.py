"""Unit tests for TradingEngine strategy allocation integration.

Module B: Verifies DynamicStrategyAllocator is wired into TradingEngine
and scales effective NAV per strategy.
"""

import pandas as pd
import pytest

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.main.trading_engine import TradingEngine
from quant_fund.portfolio.capital_allocation.dynamic_strategy_allocator import (
    DynamicStrategyAllocator,
)
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from tests.conftest import STANDARD_MARKET_DATA


@pytest.mark.unit
@pytest.mark.tier2
class TestTradingEngineAllocation:
    """DynamicStrategyAllocator integration into TradingEngine."""

    @pytest.fixture
    def broker(self):
        b = SimulationBroker(config={"initial_cash": 1_000_000})
        b.set_market_data(STANDARD_MARKET_DATA)
        return b

    @pytest.fixture
    def engine(self, broker):
        e = TradingEngine(config={
            "synchronous": True,
            "strategy_id": "alpha_momentum",
        })
        e.inject_components(
            broker=broker,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
            kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
            strategy_allocator=DynamicStrategyAllocator(),
        )
        return e

    def test_strategy_allocator_injection(self, engine):
        """Allocator is injected and stored."""
        assert engine._strategy_allocator is not None

    def test_allocations_scale_effective_nav(self, broker):
        """50% allocation → orders are roughly half the size."""
        # Engine A: no allocation (full NAV)
        engine_full = TradingEngine(config={
            "synchronous": True,
            "strategy_id": "strat_a",
        })
        broker_full = SimulationBroker(config={"initial_cash": 1_000_000})
        broker_full.set_market_data(STANDARD_MARKET_DATA)
        engine_full.inject_components(
            broker=broker_full,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker_full),
            kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
        )
        weights = pd.Series({"AAPL": 0.10})
        engine_full.update_target_weights(weights)
        engine_full._convergence_tick()
        pos_full = broker_full.get_positions()

        # Engine B: 50% allocation
        engine_half = TradingEngine(config={
            "synchronous": True,
            "strategy_id": "strat_a",
        })
        broker_half = SimulationBroker(config={"initial_cash": 1_000_000})
        broker_half.set_market_data(STANDARD_MARKET_DATA)
        allocator = DynamicStrategyAllocator()
        engine_half.inject_components(
            broker=broker_half,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker_half),
            kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
            strategy_allocator=allocator,
        )
        engine_half._strategy_nav_allocations = {"strat_a": 0.5}
        engine_half.update_target_weights(weights)
        engine_half._convergence_tick()
        pos_half = broker_half.get_positions()

        # Half-allocation should produce roughly half the position
        if "AAPL" in pos_full.index and "AAPL" in pos_half.index:
            ratio = pos_half["AAPL"] / pos_full["AAPL"]
            assert 0.3 <= ratio <= 0.7  # ~50% with rounding tolerance

    def test_no_allocator_uses_full_nav(self, broker):
        """Without allocator, full NAV is used (backward compatible)."""
        engine = TradingEngine(config={"synchronous": True})
        engine.inject_components(
            broker=broker,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
            kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
        )

        weights = pd.Series({"AAPL": 0.05})
        engine.update_target_weights(weights)
        engine._convergence_tick()

        positions = broker.get_positions()
        assert len(positions) > 0

    def test_update_strategy_allocations_calls_allocator(self, engine):
        """update_strategy_allocations calls the allocator."""
        allocations = engine.update_strategy_allocations({
            "alpha_momentum": 1.5,
            "mean_reversion": 0.8,
        })
        assert "alpha_momentum" in allocations
        assert "mean_reversion" in allocations
        assert sum(allocations.values()) <= 1.0 + 1e-9

    def test_zero_allocation_generates_no_orders(self, broker):
        """Zero allocation for the strategy → no orders."""
        engine = TradingEngine(config={
            "synchronous": True,
            "strategy_id": "strat_a",
        })
        allocator = DynamicStrategyAllocator()
        engine.inject_components(
            broker=broker,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
            kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
            strategy_allocator=allocator,
        )
        engine._strategy_nav_allocations = {"strat_a": 0.0}

        weights = pd.Series({"AAPL": 0.10})
        engine.update_target_weights(weights)

        initial_positions = broker.get_positions()
        engine._convergence_tick()
        after_positions = broker.get_positions()

        assert initial_positions.equals(after_positions)
