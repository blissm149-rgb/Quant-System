"""Test market impact model scaling behavior.

Validates that the Almgren-Chriss impact model exhibits correct
power-law scaling and that portfolio-level costs are self-consistent.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.portfolio.capacity_model.market_impact_model import MarketImpactModel


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestMarketImpactScaling:
    """Verify market impact scales correctly with order size and volatility."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        self.model = MarketImpactModel()

    def test_impact_increases_with_order_size(self):
        """Impact should increase monotonically with order size (concave)."""
        adv = 50_000_000.0  # $50M ADV
        vol = 0.02  # 2% daily vol

        sizes = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
        impacts = [self.model.estimate_impact_bps(s, adv, vol) for s in sizes]

        for i in range(1, len(impacts)):
            assert impacts[i] > impacts[i - 1], (
                f"Impact should increase: size ${sizes[i]:,.0f} → "
                f"{impacts[i]:.2f} bps <= size ${sizes[i-1]:,.0f} → "
                f"{impacts[i-1]:.2f} bps"
            )

    def test_impact_concave_in_order_size(self):
        """Power-law exponent < 1 means marginal impact decreases with size
        (concavity): doubling order size should less-than-double the impact."""
        adv = 50_000_000.0
        vol = 0.02

        impact_1m = self.model.estimate_impact_bps(1_000_000, adv, vol)
        impact_2m = self.model.estimate_impact_bps(2_000_000, adv, vol)

        # The temporary component has exponent 0.6, so impact shouldn't
        # double. Allow some tolerance for the linear permanent component.
        ratio = impact_2m / impact_1m
        assert ratio < 2.0, (
            f"Doubling order size should less-than-double impact: "
            f"ratio = {ratio:.3f}"
        )

    def test_impact_increases_with_volatility(self):
        """Higher daily volatility should increase market impact."""
        order_size = 1_000_000.0
        adv = 50_000_000.0

        impact_low = self.model.estimate_impact_bps(order_size, adv, 0.01)
        impact_high = self.model.estimate_impact_bps(order_size, adv, 0.04)

        assert impact_high > impact_low, (
            f"Impact with 4% vol ({impact_high:.2f} bps) should exceed "
            f"impact with 1% vol ({impact_low:.2f} bps)"
        )

    def test_portfolio_impact_consistent_with_single_name(self):
        """Portfolio-level impact should equal sum of per-name impacts."""
        tickers = ["AAPL", "MSFT", "GOOG"]
        trades = pd.Series([1_000_000, 500_000, 750_000], index=tickers)
        adv = pd.Series([100_000_000, 80_000_000, 60_000_000], index=tickers)
        vol = pd.Series([0.02, 0.018, 0.025], index=tickers)

        portfolio_impacts = self.model.estimate_impact_portfolio(trades, adv, vol)

        for ticker in tickers:
            single = self.model.estimate_impact_bps(
                trades[ticker], adv[ticker], vol[ticker]
            )
            assert abs(portfolio_impacts[ticker] - single) < 1e-10, (
                f"Portfolio impact for {ticker} ({portfolio_impacts[ticker]:.4f}) "
                f"should match single-name ({single:.4f})"
            )
