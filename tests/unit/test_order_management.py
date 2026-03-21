"""Unit tests for order management modules.

TESTING_PLAN.md Section 3.10 — order_generator, order_safety_validator.
"""

import pandas as pd
import pytest

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderType,
)
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_safety_validator import (
    OrderSafetyValidator,
    SafetyResult,
)


@pytest.mark.unit
@pytest.mark.tier1
class TestOrderGenerator:
    """OrderGenerator — converts target weights to orders."""

    @pytest.fixture
    def generator(self):
        return OrderGenerator()

    def test_generates_orders_from_weights(self, generator):
        """Non-zero weight deltas produce orders."""
        target = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        current = pd.Series({"AAPL": 0, "MSFT": 0}, dtype=float)
        prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})
        orders = generator.generate_orders(target, current, prices, nav=1_000_000)
        assert len(orders) > 0
        assert all(isinstance(o, Order) for o in orders)

    def test_buy_and_sell_sides(self, generator):
        """Positive target → BUY, negative target → SELL."""
        target = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        current = pd.Series({"AAPL": 0, "MSFT": 0}, dtype=float)
        prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})
        orders = generator.generate_orders(target, current, prices, nav=1_000_000)
        sides = {o.ticker: o.side for o in orders}
        assert sides["AAPL"] == OrderSide.BUY
        assert sides["MSFT"] == OrderSide.SELL

    def test_no_orders_when_on_target(self, generator):
        """No orders generated when positions match targets."""
        target = pd.Series({"AAPL": 0.0})
        current = pd.Series({"AAPL": 0.0})
        prices = pd.Series({"AAPL": 150.0})
        orders = generator.generate_orders(target, current, prices, nav=1_000_000)
        assert len(orders) == 0

    def test_liquidation_orders(self, generator):
        """Liquidation orders flatten all positions."""
        positions = pd.Series({"AAPL": 100, "MSFT": -50})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})
        orders = generator.generate_liquidation_orders(positions, prices)
        assert len(orders) == 2
        sides = {o.ticker: o.side for o in orders}
        assert sides["AAPL"] == OrderSide.SELL  # long → sell
        assert sides["MSFT"] == OrderSide.BUY   # short → buy

    def test_strategy_id_propagated(self, generator):
        """Strategy ID is set on generated orders."""
        target = pd.Series({"AAPL": 0.02})
        current = pd.Series({"AAPL": 0.0})
        prices = pd.Series({"AAPL": 150.0})
        orders = generator.generate_orders(
            target, current, prices, nav=1_000_000, strategy_id="strat_1",
        )
        assert all(o.strategy_id == "strat_1" for o in orders)


@pytest.mark.unit
@pytest.mark.tier1
class TestOrderSafetyValidator:
    """OrderSafetyValidator — pre-trade safety checks."""

    @pytest.fixture
    def validator(self):
        return OrderSafetyValidator()

    def test_valid_order_passes(self, validator):
        """Normal order passes all checks."""
        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET,
        )
        result = validator.validate(
            order, nav=1_000_000, adv=50_000_000, mid_price=150.0,
        )
        assert isinstance(result, SafetyResult)
        assert result.passed is True

    def test_zero_quantity_rejected(self, validator):
        """Zero quantity order is rejected."""
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=0)
        result = validator.validate(order, nav=1_000_000)
        assert result.passed is False

    def test_exceeds_max_shares(self, validator):
        """Order exceeding max shares per order is rejected."""
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=2_000_000)
        result = validator.validate(order, nav=1_000_000_000, mid_price=150.0)
        assert result.passed is False

    def test_fat_finger_check(self, validator):
        """Fat finger: limit price too far from mid is rejected."""
        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, limit_price=1500.0,
        )
        result = validator.validate(order, nav=1_000_000, mid_price=150.0)
        assert result.passed is False

    def test_clear_state_resets(self, validator):
        """clear_state resets internal tracking."""
        validator.clear_state()  # should not raise
