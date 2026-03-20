"""Unit tests for alpha discovery modules.

TESTING_PLAN.md Section 3.4 — feature_combinator, genetic_algorithm_search,
ml_feature_selector, symbolic_regression_engine.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.alpha_discovery.feature_combinator import (
    CandidateSignal,
    FeatureCombinator,
)
from quant_fund.alpha_discovery.genetic_algorithm_search import (
    GAResult,
    GeneticAlgorithmSearch,
)
from quant_fund.alpha_discovery.ml_feature_selector import (
    FeatureRanking,
    MLFeatureSelector,
)
from quant_fund.alpha_discovery.symbolic_regression_engine import (
    SymbolicExpression,
    SymbolicRegressionEngine,
)


@pytest.mark.unit
@pytest.mark.tier2
class TestFeatureCombinator:
    """FeatureCombinator — generates candidate signals from feature combinations."""

    @pytest.fixture
    def combinator(self):
        return FeatureCombinator(config={"n_random_combinations": 10, "random_seed": 42})

    @pytest.fixture
    def feature_matrix(self):
        rng = np.random.default_rng(42)
        tickers = [f"T{i:03d}" for i in range(50)]
        return pd.DataFrame({
            "momentum": rng.standard_normal(50),
            "value": rng.standard_normal(50),
            "quality": rng.standard_normal(50),
        }, index=tickers)

    def test_generates_candidates(self, combinator, feature_matrix):
        """generate_candidates returns list of CandidateSignal."""
        candidates = combinator.generate_candidates(feature_matrix)
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert all(isinstance(c, CandidateSignal) for c in candidates)

    def test_ratio_features_handle_zero_denominators(self, combinator):
        """Ratio features don't crash on zero denominators."""
        tickers = [f"T{i:03d}" for i in range(20)]
        fm = pd.DataFrame({
            "a": np.ones(20),
            "b": np.zeros(20),  # zero denominator
        }, index=tickers)
        candidates = combinator.generate_candidates(fm)
        # Should not raise — zeros handled gracefully
        assert isinstance(candidates, list)


@pytest.mark.unit
@pytest.mark.tier2
class TestGeneticAlgorithmSearch:
    """GeneticAlgorithmSearch — evolutionary optimization."""

    @pytest.fixture
    def ga(self):
        return GeneticAlgorithmSearch(config={
            "population_size": 20,
            "generations": 10,
            "random_seed": 42,
        })

    def test_converges_with_known_fitness(self, ga):
        """GA converges toward optimal parameters for known fitness."""
        # Simple quadratic fitness: maximize -(x-5)^2 -(y-3)^2
        def fitness(params):
            return -(params["x"] - 5) ** 2 - (params["y"] - 3) ** 2

        result = ga.search(
            param_ranges={"x": (0, 10), "y": (0, 10)},
            fitness_fn=fitness,
        )
        assert isinstance(result, GAResult)
        assert result.best_fitness > -10  # should be near 0

    def test_returns_ga_result(self, ga):
        """search returns GAResult with expected fields."""
        result = ga.search(
            param_ranges={"x": (0, 1)},
            fitness_fn=lambda p: -p["x"] ** 2,
        )
        assert isinstance(result, GAResult)
        assert result.generations_run > 0


@pytest.mark.unit
@pytest.mark.tier2
class TestMLFeatureSelector:
    """MLFeatureSelector — feature importance and redundancy removal."""

    @pytest.fixture
    def selector(self):
        return MLFeatureSelector(config={"correlation_threshold": 0.85, "random_seed": 42})

    def test_selects_features(self, selector):
        """select returns FeatureRanking with selected features."""
        rng = np.random.default_rng(42)
        n = 100
        features = pd.DataFrame({
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
        })
        target = pd.Series(rng.standard_normal(n))
        result = selector.select(features, target)
        assert isinstance(result, FeatureRanking)
        assert len(result.selected_features) > 0

    def test_removes_redundant_features(self, selector):
        """Highly correlated features are removed."""
        rng = np.random.default_rng(42)
        n = 100
        base = rng.standard_normal(n)
        features = pd.DataFrame({
            "f1": base,
            "f2": base + rng.standard_normal(n) * 0.01,  # nearly identical to f1
            "f3": rng.standard_normal(n),
        })
        target = pd.Series(rng.standard_normal(n))
        result = selector.select(features, target)
        # One of f1/f2 should be removed as redundant
        assert len(result.removed_features) >= 1 or len(result.selected_features) <= 2


@pytest.mark.unit
@pytest.mark.tier2
class TestSymbolicRegressionEngine:
    """SymbolicRegressionEngine — discovers mathematical relationships."""

    @pytest.fixture
    def engine(self):
        return SymbolicRegressionEngine(config={
            "population_size": 30,
            "generations": 5,
            "random_seed": 42,
        })

    def test_search_returns_expressions(self, engine):
        """search returns list of SymbolicExpression."""
        rng = np.random.default_rng(42)
        n = 100
        features = pd.DataFrame({
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        })
        target = pd.Series(rng.standard_normal(n))
        results = engine.search(features, target, n_best=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(e, SymbolicExpression) for e in results)

    def test_expressions_have_valid_strings(self, engine):
        """Output expressions have non-empty expression strings."""
        rng = np.random.default_rng(42)
        n = 100
        features = pd.DataFrame({"x1": rng.standard_normal(n)})
        target = pd.Series(rng.standard_normal(n))
        results = engine.search(features, target, n_best=3)
        for expr in results:
            assert len(expr.expression_str) > 0
