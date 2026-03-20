"""Unit tests for capital allocation modules.

TESTING_PLAN.md Section 3.8 — dynamic_strategy_allocator, strategy_correlation_matrix, strategy_performance_tracker.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.portfolio.capital_allocation.dynamic_strategy_allocator import (
    DynamicStrategyAllocator,
)
from quant_fund.portfolio.capital_allocation.strategy_correlation_matrix import (
    StrategyCorrelationMatrix,
)
from quant_fund.portfolio.capital_allocation.strategy_performance_tracker import (
    StrategyMetrics,
    StrategyPerformanceTracker,
)


@pytest.mark.unit
@pytest.mark.tier2
class TestDynamicStrategyAllocator:
    """DynamicStrategyAllocator — Sharpe-weighted allocation."""

    @pytest.fixture
    def allocator(self):
        return DynamicStrategyAllocator()

    def test_allocations_sum_to_one(self, allocator):
        """Allocations always sum to 1.0."""
        sharpes = {"strat_a": 1.5, "strat_b": 0.8, "strat_c": 2.0}
        alloc = allocator.allocate(sharpes)
        np.testing.assert_allclose(sum(alloc.values()), 1.0, atol=1e-6)

    def test_higher_sharpe_gets_more_allocation(self, allocator):
        """Strategy with higher Sharpe gets larger allocation."""
        sharpes = {"high": 3.0, "low": 0.5}
        alloc = allocator.allocate(sharpes)
        assert alloc["high"] >= alloc["low"]

    def test_correlation_penalty_reduces_allocation(self, allocator):
        """Correlation penalty reduces crowded strategy allocation."""
        sharpes = {"a": 1.5, "b": 1.5}
        penalties = pd.Series({"a": 0.8, "b": 0.0})
        alloc = allocator.allocate(sharpes, correlation_penalties=penalties)
        # Strategy 'a' is crowded, should get less
        assert alloc["a"] <= alloc["b"]


@pytest.mark.unit
@pytest.mark.tier2
class TestStrategyCorrelationMatrix:
    """StrategyCorrelationMatrix — correlation between strategy returns."""

    @pytest.fixture
    def corr_calc(self):
        return StrategyCorrelationMatrix()

    def test_correlation_matrix_symmetric(self, corr_calc):
        """Correlation matrix is symmetric."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-02", periods=100)
        strat_returns = {
            "a": pd.Series(rng.standard_normal(100), index=dates),
            "b": pd.Series(rng.standard_normal(100), index=dates),
        }
        corr = corr_calc.compute(strat_returns)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_diagonal_is_one(self, corr_calc):
        """Diagonal of correlation matrix is 1.0."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-02", periods=100)
        strat_returns = {
            "a": pd.Series(rng.standard_normal(100), index=dates),
            "b": pd.Series(rng.standard_normal(100), index=dates),
        }
        corr = corr_calc.compute(strat_returns)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-6)

    def test_bounded_minus_one_to_one(self, corr_calc):
        """All correlation values in [-1, 1]."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-02", periods=100)
        strat_returns = {
            "a": pd.Series(rng.standard_normal(100), index=dates),
            "b": pd.Series(rng.standard_normal(100), index=dates),
        }
        corr = corr_calc.compute(strat_returns)
        assert corr.values.min() >= -1.0 - 1e-6
        assert corr.values.max() <= 1.0 + 1e-6


@pytest.mark.unit
@pytest.mark.tier2
class TestStrategyPerformanceTracker:
    """StrategyPerformanceTracker — trailing Sharpe and Sortino."""

    @pytest.fixture
    def tracker(self):
        return StrategyPerformanceTracker()

    def test_update_and_get_metrics(self, tracker):
        """Update with daily returns and retrieve metrics."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-02", periods=100)
        for dt in dates:
            tracker.update("strat_1", rng.normal(0.001, 0.01), dt)
        metrics = tracker.get_metrics("strat_1")
        assert isinstance(metrics, StrategyMetrics)
        assert metrics.n_days == 100

    def test_sharpe_computed_correctly(self, tracker):
        """Trailing Sharpe is computed."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-02", periods=100)
        for dt in dates:
            tracker.update("strat_1", rng.normal(0.002, 0.01), dt)
        metrics = tracker.get_metrics("strat_1")
        # With positive mean return, Sharpe should be positive
        assert metrics.trailing_sharpe > 0

    def test_get_all_metrics(self, tracker):
        """get_all_metrics returns all tracked strategies."""
        tracker.update("a", 0.01, pd.Timestamp("2023-01-02"))
        tracker.update("b", -0.01, pd.Timestamp("2023-01-02"))
        metrics = tracker.get_all_metrics()
        assert len(metrics) == 2
