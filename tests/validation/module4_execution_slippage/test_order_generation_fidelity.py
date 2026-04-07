"""Test order generation fidelity.

Validates that OrderGenerator correctly converts portfolio weights
to orders, filters sub-minimum trades, and handles edge cases.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.broker_interface.broker_abstraction_layer import OrderSide


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestOrderGenerationFidelity:
    """Verify OrderGenerator produces correct, filtered orders."""

    @pytest.fixture(autouse=True)
    def setup_gen(self):
        self.gen = OrderGenerator()
        self.tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        self.prices = pd.Series(
            [150.0, 300.0, 140.0, 180.0, 350.0], index=self.tickers
        )
        self.nav = 10_000_000.0  # $10M NAV

    def test_long_short_orders_generated_correctly(self):
        """Positive target weights should produce BUY orders, negative
        weights should produce SELL orders (from flat)."""
        target_weights = pd.Series(
            [0.10, -0.10, 0.05, -0.05, 0.0], index=self.tickers
        )
        current_positions = pd.Series(
            [0, 0, 0, 0, 0], index=self.tickers, dtype=float
        )

        orders = self.gen.generate_orders(
            target_weights, current_positions, self.prices, self.nav
        )

        buy_tickers = {o.ticker for o in orders if o.side == OrderSide.BUY}
        sell_tickers = {o.ticker for o in orders if o.side == OrderSide.SELL}

        assert "AAPL" in buy_tickers, "AAPL (positive weight) should be a BUY"
        assert "MSFT" in sell_tickers, "MSFT (negative weight) should be a SELL"
        assert "GOOG" in buy_tickers, "GOOG (positive weight) should be a BUY"
        assert "AMZN" in sell_tickers, "AMZN (negative weight) should be a SELL"

    def test_sub_minimum_trades_filtered_out(self):
        """Trades below $1,000 notional should be filtered out."""
        # Very small weight → trade < $1,000
        tiny_weight = 0.00005  # 0.005% of $10M = $500
        target_weights = pd.Series(
            [tiny_weight, 0.0, 0.0, 0.0, 0.0], index=self.tickers
        )
        current_positions = pd.Series(
            [0, 0, 0, 0, 0], index=self.tickers, dtype=float
        )

        orders = self.gen.generate_orders(
            target_weights, current_positions, self.prices, self.nav
        )

        aapl_orders = [o for o in orders if o.ticker == "AAPL"]
        assert len(aapl_orders) == 0, (
            f"Trade of ~${tiny_weight * self.nav:.0f} should be filtered "
            f"(below $1,000 min), but got {len(aapl_orders)} orders"
        )

    def test_liquidation_orders_close_all_positions(self):
        """Liquidation should generate one order per non-zero position."""
        positions = pd.Series(
            [1000, -500, 200, 0, -300], index=self.tickers, dtype=float
        )

        orders = self.gen.generate_liquidation_orders(
            positions, self.prices, strategy_id="test"
        )

        # Should have 4 orders (skip META with 0 position)
        assert len(orders) == 4, (
            f"Expected 4 liquidation orders (skip zero position), got {len(orders)}"
        )

        # Long positions → SELL, Short positions → BUY
        for o in orders:
            pos = positions[o.ticker]
            if pos > 0:
                assert o.side == OrderSide.SELL, (
                    f"Long position in {o.ticker} should produce SELL"
                )
            else:
                assert o.side == OrderSide.BUY, (
                    f"Short position in {o.ticker} should produce BUY"
                )

    def test_no_orders_on_zero_nav(self):
        """OrderGenerator should return empty list when NAV is zero."""
        target_weights = pd.Series([0.10], index=["AAPL"])
        current = pd.Series([0], index=["AAPL"], dtype=float)

        orders = self.gen.generate_orders(
            target_weights, current, self.prices, nav=0.0
        )
        assert len(orders) == 0, "Zero NAV should produce no orders"
