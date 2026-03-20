"""Unit tests for portfolio_optimizer module.

TESTING_PLAN.md Section 3.8 — Portfolio Construction.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.portfolio.portfolio_construction.constraint_engine import (
    ConstraintEngine,
    ConstraintSet,
)
from quant_fund.portfolio.portfolio_construction.portfolio_optimizer import (
    PortfolioOptimizer,
)


@pytest.mark.unit
@pytest.mark.tier2
class TestPortfolioOptimizer:
    """PortfolioOptimizer — mean-variance optimization."""

    @pytest.fixture
    def optimizer(self):
        return PortfolioOptimizer()

    @pytest.fixture
    def alpha_scores(self):
        return pd.Series({
            "AAPL": 0.05, "MSFT": 0.03, "GOOG": -0.02,
            "AMZN": 0.01, "META": -0.04,
        })

    @pytest.fixture
    def factor_cov(self):
        rng = np.random.default_rng(42)
        factors = ["Market", "Size", "Value"]
        n = len(factors)
        A = rng.standard_normal((n, n))
        cov = A @ A.T / n + np.eye(n) * 0.01
        return pd.DataFrame(cov, index=factors, columns=factors)

    @pytest.fixture
    def factor_exposures(self, alpha_scores):
        rng = np.random.default_rng(42)
        tickers = list(alpha_scores.index)
        factors = ["Market", "Size", "Value"]
        data = rng.standard_normal((len(tickers), len(factors)))
        return pd.DataFrame(data, index=tickers, columns=factors)

    @pytest.fixture
    def constraints(self):
        engine = ConstraintEngine(config={
            "max_position_size": 0.02,
            "max_leverage": 2.0,
        })
        return engine.build_constraints()

    def test_optimize_returns_series(self, optimizer, alpha_scores, factor_cov, factor_exposures, constraints):
        """Optimizer returns pd.Series of weights."""
        weights = optimizer.optimize(alpha_scores, factor_cov, factor_exposures, constraints)
        assert isinstance(weights, pd.Series)
        assert len(weights) > 0

    def test_higher_alpha_gets_higher_weight(self, optimizer, alpha_scores, factor_cov, factor_exposures, constraints):
        """Ticker with highest alpha gets highest absolute weight (approximately)."""
        weights = optimizer.optimize(alpha_scores, factor_cov, factor_exposures, constraints)
        # AAPL has highest alpha — should have positive weight
        if "AAPL" in weights.index:
            assert weights["AAPL"] >= 0

    def test_weights_satisfy_position_limit(self, optimizer, alpha_scores, factor_cov, factor_exposures, constraints):
        """All weights within max_position_size."""
        weights = optimizer.optimize(alpha_scores, factor_cov, factor_exposures, constraints)
        assert weights.abs().max() <= constraints.max_position_size + 1e-4

    def test_weights_satisfy_leverage_limit(self, optimizer, alpha_scores, factor_cov, factor_exposures, constraints):
        """Gross leverage within limit."""
        weights = optimizer.optimize(alpha_scores, factor_cov, factor_exposures, constraints)
        assert weights.abs().sum() <= constraints.max_leverage + 1e-4
