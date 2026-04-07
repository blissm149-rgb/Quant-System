"""Test spread estimation and capacity analysis.

Validates bid-ask spread model estimators and capacity simulator
breakeven logic.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.execution.microstructure_models.bid_ask_spread_model import (
    BidAskSpreadModel,
)
from quant_fund.portfolio.capacity_model.capacity_simulator import CapacitySimulator


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestSpreadAndCapacity:
    """Verify spread estimation accuracy and capacity scaling."""

    def test_quoted_spread_correct_for_known_values(self):
        """Quoted spread should equal (ask - bid) / mid * 10000."""
        model = BidAskSpreadModel()
        bid, ask = 99.95, 100.05
        expected = (ask - bid) / ((bid + ask) / 2) * 10000

        result = model.estimate_quoted_spread(bid, ask)
        assert abs(result - expected) < 0.01, (
            f"Quoted spread {result:.4f} should equal {expected:.4f} bps"
        )

    def test_roll_estimator_detects_bid_ask_bounce(self):
        """Roll estimator should return positive spread when returns
        exhibit negative autocorrelation (bid-ask bounce)."""
        rng = np.random.default_rng(42)
        # Simulate bid-ask bounce: alternating positive/negative returns
        n = 500
        bounce = np.where(np.arange(n) % 2 == 0, 0.001, -0.001)
        noise = rng.normal(0, 0.0005, n)
        returns = pd.Series(bounce + noise)

        model = BidAskSpreadModel()
        spread = model.estimate_effective_spread_roll(returns)

        assert spread > 0, (
            f"Roll estimator should detect positive spread from bid-ask "
            f"bounce, got {spread:.4f} bps"
        )

    def test_capacity_decreases_with_lower_adv(self):
        """Strategy trading illiquid names should have lower capacity
        than the same strategy trading liquid names."""
        tickers = ["A", "B", "C"]
        target = pd.Series([0.10, 0.10, 0.10], index=tickers)
        current = pd.Series([0.0, 0.0, 0.0], index=tickers)
        vol = pd.Series([0.02, 0.02, 0.02], index=tickers)
        alpha = 50.0  # 50 bps expected alpha

        # Liquid universe
        sim_liquid = CapacitySimulator()
        adv_liquid = pd.Series([500e6, 300e6, 200e6], index=tickers)
        cap_liquid = sim_liquid.estimate_capacity(
            target, current, adv_liquid, vol, alpha
        )

        # Illiquid universe
        sim_illiquid = CapacitySimulator()
        adv_illiquid = pd.Series([5e6, 3e6, 2e6], index=tickers)
        cap_illiquid = sim_illiquid.estimate_capacity(
            target, current, adv_illiquid, vol, alpha
        )

        assert cap_illiquid["max_capacity_usd"] <= cap_liquid["max_capacity_usd"], (
            f"Illiquid capacity (${cap_illiquid['max_capacity_usd']:,.0f}) "
            f"should be <= liquid capacity "
            f"(${cap_liquid['max_capacity_usd']:,.0f})"
        )
