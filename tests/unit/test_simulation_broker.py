"""Unit tests for simulation broker.

TESTING_PLAN.md Section 3.11 — simulation_broker (CRITICAL).
"""

import pandas as pd
import pytest

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from quant_fund.broker_interface.simulation_broker import SimulationBroker
from tests.conftest import STANDARD_MARKET_DATA


@pytest.mark.unit
@pytest.mark.tier1
class TestSimulationBroker:
    """SimulationBroker — CRITICAL: fills, slippage, cash floor."""

    @pytest.fixture
    def broker(self):
        b = SimulationBroker(config={"initial_cash": 1_000_000})
        b.set_market_data(STANDARD_MARKET_DATA)
        return b

    def test_submit_market_order_fills(self, broker):
        """Market order gets filled."""
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=10)
        ack = broker.submit_order(order)
        assert ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL)

    def test_position_updates_after_fill(self, broker):
        """Position reflects filled order."""
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=10)
        broker.submit_order(order)
        positions = broker.get_positions()
        assert positions.get("AAPL", 0) > 0

    def test_cash_decreases_on_buy(self, broker):
        """Cash decreases after buying."""
        initial_cash = broker.cash
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=10)
        broker.submit_order(order)
        assert broker.cash < initial_cash

    def test_cash_increases_on_sell(self, broker):
        """Cash increases after selling (short sale)."""
        initial_cash = broker.cash
        order = Order(ticker="AAPL", side=OrderSide.SELL, quantity=10)
        broker.submit_order(order)
        assert broker.cash > initial_cash

    def test_slippage_applied(self, broker):
        """Fill price differs from mid price (slippage)."""
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(order)
        fills = broker.all_fills
        assert len(fills) > 0
        mid = STANDARD_MARKET_DATA["AAPL"]["mid"]
        # Buy fill should be >= mid (slippage costs more)
        assert fills[-1].fill_price >= mid

    def test_commission_charged(self, broker):
        """Commission is applied on fills."""
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(order)
        fills = broker.all_fills
        assert fills[-1].commission > 0

    def test_cancel_order(self, broker):
        """cancel_order returns bool."""
        result = broker.cancel_order("nonexistent_id")
        assert isinstance(result, bool)

    def test_get_account_value(self, broker):
        """Account value equals initial cash when no positions."""
        assert broker.get_account_value() == pytest.approx(1_000_000, rel=0.01)

    def test_get_fills_since(self, broker):
        """get_fills returns fills since timestamp."""
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=10)
        broker.submit_order(order)
        fills = broker.get_fills(since=pd.Timestamp("2020-01-01"))
        assert len(fills) > 0

    def test_enforce_cash_floor(self):
        """Cash floor prevents negative cash."""
        b = SimulationBroker(config={
            "initial_cash": 100, "enforce_cash_floor": True,
        })
        b.set_market_data(STANDARD_MARKET_DATA)
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=10000)
        ack = b.submit_order(order)
        # Should either reject or partially fill to stay above zero
        assert b.cash >= 0 or ack.status == OrderStatus.REJECTED
