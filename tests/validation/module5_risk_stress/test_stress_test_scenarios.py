"""Test stress test engine scenario behavior.

Validates that the stress test engine produces negative returns under
crisis scenarios, correct factor P&L attribution, and consistent results.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.risk_engine.stress_test_engine import (
    StressTestEngine,
    DEFAULT_SCENARIOS,
)


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


def _make_portfolio():
    """Create a test portfolio with known factor exposures."""
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    weights = pd.Series([0.10, 0.08, -0.05, 0.07, -0.06], index=tickers)
    factors = ["market", "momentum", "value", "quality", "low_vol", "size"]
    rng = np.random.default_rng(42)
    exposures = pd.DataFrame(
        rng.normal(0, 0.5, (len(tickers), len(factors))),
        index=tickers,
        columns=factors,
    )
    # Set market beta close to 1.0 for long positions
    exposures["market"] = [1.1, 0.9, 1.2, 1.0, 0.8]
    return weights, exposures


class TestStressTestScenarios:
    """Verify stress test engine produces correct outputs."""

    def test_2008_scenario_produces_negative_return(self):
        """A net-long portfolio should have negative return under 2008 GFC."""
        weights, exposures = _make_portfolio()
        engine = StressTestEngine()
        results = engine.run_all(weights, exposures)

        gfc = [r for r in results if r.scenario_name == "2008_financial_crisis"][0]
        assert gfc.portfolio_return < 0, (
            f"Net-long portfolio should lose money under 2008 GFC, "
            f"got return = {gfc.portfolio_return:.4f}"
        )

    def test_all_scenarios_produce_results(self):
        """All default scenarios should produce StressTestResult objects."""
        weights, exposures = _make_portfolio()
        engine = StressTestEngine()
        results = engine.run_all(weights, exposures)

        assert len(results) == len(DEFAULT_SCENARIOS), (
            f"Expected {len(DEFAULT_SCENARIOS)} results, got {len(results)}"
        )
        for r in results:
            assert r.scenario_name in DEFAULT_SCENARIOS
            assert isinstance(r.factor_pnl, dict)
            assert len(r.factor_pnl) > 0

    def test_factor_pnl_sums_to_portfolio_return(self):
        """Sum of factor P&L contributions should equal the total portfolio
        return (when no idiosyncratic term is added)."""
        weights, exposures = _make_portfolio()
        engine = StressTestEngine()
        results = engine.run_all(weights, exposures)

        for r in results:
            factor_sum = sum(r.factor_pnl.values())
            assert abs(factor_sum - r.portfolio_return) < 1e-10, (
                f"Factor P&L sum ({factor_sum:.6f}) should equal portfolio "
                f"return ({r.portfolio_return:.6f}) in scenario {r.scenario_name}"
            )

    def test_drawdown_is_positive_for_negative_return(self):
        """Drawdown should be positive (absolute value) when return is
        negative, and zero when return is positive."""
        weights, exposures = _make_portfolio()
        engine = StressTestEngine()
        results = engine.run_all(weights, exposures)

        for r in results:
            if r.portfolio_return < 0:
                assert r.portfolio_drawdown > 0, (
                    f"Drawdown should be positive when return is negative "
                    f"in {r.scenario_name}"
                )
                assert abs(r.portfolio_drawdown - abs(r.portfolio_return)) < 1e-10
            else:
                assert r.portfolio_drawdown == 0.0
