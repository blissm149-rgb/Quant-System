"""Test fill realism assumptions in backtests.

Validates that the execution simulation does not assume zero-cost fills
and that market impact scales appropriately with order size.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.portfolio.capacity_model.market_impact_model import MarketImpactModel
from quant_fund.execution.execution_algorithms.vwap_execution import (
    VWAPExecution,
    VWAPPlan,
)
from tests.conftest import STANDARD_MARKET_DATA


pytestmark = [pytest.mark.validation]


class TestFillRealism:
    """Verify that fill assumptions are realistic and not overly optimistic."""

    def test_no_zero_slippage_assumption(self):
        """Market impact model must return nonzero cost for any nonzero trade."""
        model = MarketImpactModel()

        # Test a range of order sizes
        order_sizes = [10_000, 100_000, 1_000_000, 10_000_000]
        adv = 50_000_000  # $50M ADV
        vol = 0.02  # 2% daily vol

        for size in order_sizes:
            impact = model.estimate_impact_bps(
                order_size_usd=float(size),
                adv_usd=float(adv),
                daily_volatility=vol,
            )
            assert impact > 0, (
                f"Impact must be > 0 for order_size={size}, "
                f"ADV={adv}, vol={vol}. Got impact={impact}"
            )

    def test_fill_prices_differ_from_close(self):
        """When using VWAP execution with market impact, simulated fill prices
        must differ from close prices."""
        model = MarketImpactModel()

        # Simulate fills for a medium-sized order
        order_size = 500_000  # $500K
        adv = 50_000_000
        vol = 0.02
        close_price = 150.0

        impact_bps = model.estimate_impact_bps(
            order_size_usd=order_size,
            adv_usd=adv,
            daily_volatility=vol,
        )

        # Apply impact to close price (BUY side: fill worse than close)
        fill_price = close_price * (1 + impact_bps / 10_000.0)

        assert fill_price != close_price, (
            "Fill price should differ from close when market impact is applied"
        )
        assert fill_price > close_price, (
            "BUY fill price should be higher than close due to market impact"
        )

    def test_large_orders_have_higher_impact(self):
        """Orders > 1% ADV must have higher impact_bps than orders < 0.1% ADV."""
        model = MarketImpactModel()
        adv = 50_000_000
        vol = 0.02

        # Small order: 0.05% of ADV = $25K
        small_order = adv * 0.0005
        # Large order: 2% of ADV = $1M
        large_order = adv * 0.02

        impact_small = model.estimate_impact_bps(
            order_size_usd=small_order, adv_usd=adv, daily_volatility=vol
        )
        impact_large = model.estimate_impact_bps(
            order_size_usd=large_order, adv_usd=adv, daily_volatility=vol
        )

        assert impact_large > impact_small, (
            f"Large order impact ({impact_large:.2f} bps) must exceed "
            f"small order impact ({impact_small:.2f} bps)"
        )

        # The ratio should be significant (power law with exponent 0.6)
        ratio = impact_large / impact_small
        assert ratio > 2.0, (
            f"Impact ratio (large/small) = {ratio:.2f}, expected > 2.0 "
            f"given 40x size difference and power law exponent"
        )

    def test_participation_rate_enforced(self):
        """No single-day fill in a VWAP plan should exceed the participation
        rate times ADV."""
        vwap = VWAPExecution()
        adv = 1_000_000  # 1M shares ADV

        # Order for 200K shares (20% of ADV -- intentionally large)
        plan = vwap.create_plan(
            ticker="AAPL",
            side="BUY",
            total_shares=200_000,
            adv=float(adv),
        )

        # Check each slice
        max_participation_rate = 0.05  # default 5%
        for s in plan.slices:
            pct_of_adv = s.target_shares / adv
            assert pct_of_adv <= max_participation_rate + 0.01, (
                f"Slice {s.slice_index}: {s.target_shares} shares = "
                f"{pct_of_adv:.4f} of ADV, exceeds participation rate "
                f"{max_participation_rate}"
            )

        # Total plan should also not exceed participation rate * ADV per day
        total_planned = sum(s.target_shares for s in plan.slices)
        daily_cap = adv * max_participation_rate
        assert total_planned <= daily_cap + 1, (
            f"Total planned shares ({total_planned}) exceeds daily cap "
            f"({daily_cap}) = {max_participation_rate * 100}% of ADV"
        )
