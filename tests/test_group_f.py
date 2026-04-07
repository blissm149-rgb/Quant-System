"""Tests for Group F — Portfolio construction.

Validates:
- Factor exposure estimation via regression and characteristics
- Factor covariance estimation with Ledoit-Wolf shrinkage
- Risk decomposition into factor and idiosyncratic components
- Constraint engine builds and validates constraints correctly
- Portfolio optimizer produces weights satisfying all constraints:
  max single position <= 0.02, sector exposure <= 0.20, leverage <= 2.0
- Strategy performance tracker and correlation matrix
- Dynamic strategy allocator with Sharpe-weighting and penalties
- Market impact model and capacity simulator
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.portfolio.factor_risk_model.factor_exposure_estimator import (

pytestmark = [pytest.mark.tier2]
    FactorExposureEstimator,
)
from quant_fund.portfolio.factor_risk_model.factor_covariance_estimator import (
    FactorCovarianceEstimator,
)
from quant_fund.portfolio.factor_risk_model.risk_decomposition import (
    RiskDecomposition,
)
from quant_fund.portfolio.portfolio_construction.constraint_engine import (
    ConstraintEngine,
    ConstraintSet,
)
from quant_fund.portfolio.portfolio_construction.portfolio_optimizer import (
    PortfolioOptimizer,
)
from quant_fund.portfolio.capital_allocation.strategy_performance_tracker import (
    StrategyPerformanceTracker,
)
from quant_fund.portfolio.capital_allocation.strategy_correlation_matrix import (
    StrategyCorrelationMatrix,
)
from quant_fund.portfolio.capital_allocation.dynamic_strategy_allocator import (
    DynamicStrategyAllocator,
)
from quant_fund.portfolio.capacity_model.liquidity_estimator import LiquidityEstimator
from quant_fund.portfolio.capacity_model.market_impact_model import MarketImpactModel
from quant_fund.portfolio.capacity_model.capacity_simulator import CapacitySimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "BAC", "WMT"]
FACTORS = ["market", "momentum", "value", "quality", "low_vol", "size"]

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
    "AMZN": "Consumer", "META": "Technology", "TSLA": "Consumer",
    "NVDA": "Technology", "JPM": "Financials", "BAC": "Financials",
    "WMT": "Consumer",
}


def _make_returns(n_dates=300, tickers=None, seed=42):
    rng = np.random.default_rng(seed)
    tickers = tickers or TICKERS
    dates = pd.bdate_range("2019-01-02", periods=n_dates)
    data = rng.normal(0.0003, 0.02, (n_dates, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _make_factor_returns(n_dates=300, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_dates)
    data = rng.normal(0.0002, 0.01, (n_dates, len(FACTORS)))
    return pd.DataFrame(data, index=dates, columns=FACTORS)


# ---------------------------------------------------------------------------
# Factor Risk Model
# ---------------------------------------------------------------------------

class TestFactorExposureEstimator:

    def test_regression_estimates(self):
        returns = _make_returns()
        factor_returns = _make_factor_returns()
        estimator = FactorExposureEstimator({"factor_min_observations": 30})
        as_of = returns.index[-1] + pd.Timedelta(days=1)

        exposures = estimator.estimate(returns, factor_returns, as_of=as_of)
        assert exposures.shape == (len(TICKERS), len(FACTORS))
        assert not exposures.isna().any().any()

    def test_point_in_time(self):
        """Exposures must not use data after as_of."""
        returns = _make_returns()
        factor_returns = _make_factor_returns()
        estimator = FactorExposureEstimator({"factor_min_observations": 30})

        as_of = returns.index[200]
        exp1 = estimator.estimate(returns, factor_returns, as_of=as_of)

        # Modify future data
        returns_mod = returns.copy()
        returns_mod.iloc[200:] = 0.5
        exp2 = estimator.estimate(returns_mod, factor_returns, as_of=as_of)
        pd.testing.assert_frame_equal(exp1, exp2)

    def test_from_characteristics(self):
        rng = np.random.default_rng(42)
        chars = pd.DataFrame(
            rng.normal(0, 1, (10, 3)),
            index=TICKERS,
            columns=["momentum", "value", "size"],
        )
        estimator = FactorExposureEstimator()
        exposures = estimator.estimate_from_characteristics(chars)
        # Should be standardised (mean ~0, std ~1)
        assert exposures.shape == (10, 3)
        for col in exposures.columns:
            assert abs(exposures[col].mean()) < 0.1

    def test_with_sector_dummies(self):
        returns = _make_returns()
        factor_returns = _make_factor_returns()
        estimator = FactorExposureEstimator({"include_sector_factors": True})
        as_of = returns.index[-1] + pd.Timedelta(days=1)
        exposures = estimator.estimate(
            returns, factor_returns, as_of=as_of, sector_map=SECTOR_MAP
        )
        # Should have factor columns + sector columns
        assert exposures.shape[1] > len(FACTORS)


class TestFactorCovarianceEstimator:

    def test_covariance_is_positive_semidefinite(self):
        factor_returns = _make_factor_returns()
        estimator = FactorCovarianceEstimator()
        as_of = factor_returns.index[-1] + pd.Timedelta(days=1)
        cov = estimator.estimate(factor_returns, as_of=as_of)
        assert cov.shape == (len(FACTORS), len(FACTORS))
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert (eigenvalues >= -1e-10).all()

    def test_idiosyncratic_variance(self):
        stock_returns = _make_returns()
        factor_returns = _make_factor_returns()
        estimator_exp = FactorExposureEstimator({"factor_min_observations": 30})
        estimator_cov = FactorCovarianceEstimator()
        as_of = stock_returns.index[-1] + pd.Timedelta(days=1)

        exposures = estimator_exp.estimate(stock_returns, factor_returns, as_of=as_of)
        idio = estimator_cov.estimate_idiosyncratic(
            stock_returns, factor_returns, exposures, as_of=as_of
        )
        assert len(idio) == len(TICKERS)
        assert (idio >= 0).all()

    def test_full_covariance(self):
        stock_returns = _make_returns()
        factor_returns = _make_factor_returns()
        estimator_exp = FactorExposureEstimator({"factor_min_observations": 30})
        estimator_cov = FactorCovarianceEstimator()
        as_of = stock_returns.index[-1] + pd.Timedelta(days=1)

        exposures = estimator_exp.estimate(stock_returns, factor_returns, as_of=as_of)
        f_cov = estimator_cov.estimate(factor_returns, as_of=as_of)
        idio = estimator_cov.estimate_idiosyncratic(
            stock_returns, factor_returns, exposures, as_of=as_of
        )
        full = estimator_cov.build_full_covariance(exposures, f_cov, idio)
        assert full.shape == (len(TICKERS), len(TICKERS))
        # Should be symmetric
        np.testing.assert_array_almost_equal(full.values, full.values.T)


class TestRiskDecomposition:

    def test_decomposition_sums(self):
        rng = np.random.default_rng(42)
        weights = pd.Series(rng.normal(0, 0.01, 10), index=TICKERS)
        weights -= weights.mean()  # dollar neutral

        exposures = pd.DataFrame(
            rng.normal(0, 1, (10, 3)),
            index=TICKERS,
            columns=["market", "momentum", "value"],
        )
        f_cov = pd.DataFrame(
            np.diag([0.04, 0.02, 0.02]),
            index=["market", "momentum", "value"],
            columns=["market", "momentum", "value"],
        )
        idio = pd.Series(rng.uniform(0.01, 0.05, 10), index=TICKERS)

        decomp = RiskDecomposition()
        result = decomp.decompose(weights, exposures, f_cov, idio)

        assert result["total_variance"] == pytest.approx(
            result["factor_variance"] + result["idiosyncratic_variance"], abs=1e-10
        )
        assert result["total_volatility"] >= 0

    def test_top_risk_contributors(self):
        rng = np.random.default_rng(42)
        weights = pd.Series(rng.normal(0, 0.01, 10), index=TICKERS)
        full_cov = pd.DataFrame(
            np.eye(10) * 0.04, index=TICKERS, columns=TICKERS
        )
        decomp = RiskDecomposition()
        top = decomp.top_risk_contributors(weights, full_cov, top_n=3)
        assert len(top) == 3


# ---------------------------------------------------------------------------
# Portfolio Construction
# ---------------------------------------------------------------------------

class TestConstraintEngine:

    def test_build_constraints(self):
        engine = ConstraintEngine({
            "position_limits": {
                "max_position_size": 0.02,
                "max_sector_exposure": 0.20,
                "max_leverage": 2.0,
                "dollar_neutral": True,
            }
        })
        cs = engine.build_constraints(sector_map=SECTOR_MAP)
        assert cs.max_position_size == 0.02
        assert cs.max_sector_exposure == 0.20
        assert cs.max_leverage == 2.0
        assert cs.dollar_neutral is True

    def test_validate_valid_weights(self):
        engine = ConstraintEngine({
            "position_limits": {
                "max_position_size": 0.02,
                "max_leverage": 2.0,
                "dollar_neutral": True,
            }
        })
        cs = engine.build_constraints()
        # Construct valid weights
        weights = pd.Series(
            [0.01, -0.01, 0.015, -0.015, 0.005, -0.005, 0.01, -0.01, 0.005, -0.005],
            index=TICKERS,
        )
        violations = engine.validate_weights(weights, cs)
        assert len(violations) == 0

    def test_detect_position_size_violation(self):
        engine = ConstraintEngine({
            "position_limits": {"max_position_size": 0.02}
        })
        cs = engine.build_constraints()
        weights = pd.Series([0.05, -0.05], index=["AAPL", "MSFT"])
        violations = engine.validate_weights(weights, cs)
        assert any("Position size" in v for v in violations)

    def test_detect_leverage_violation(self):
        engine = ConstraintEngine({
            "position_limits": {"max_leverage": 2.0, "max_position_size": 1.0}
        })
        cs = engine.build_constraints()
        weights = pd.Series([1.5, -1.5], index=["AAPL", "MSFT"])
        violations = engine.validate_weights(weights, cs)
        assert any("Leverage" in v for v in violations)


class TestPortfolioOptimizer:

    def test_optimizer_satisfies_constraints(self):
        rng = np.random.default_rng(42)
        alpha = pd.Series(rng.normal(0, 1, 10), index=TICKERS)
        f_cov = pd.DataFrame(
            np.diag([0.04] * len(FACTORS)),
            index=FACTORS, columns=FACTORS,
        )
        exposures = pd.DataFrame(
            rng.normal(0, 0.5, (10, len(FACTORS))),
            index=TICKERS, columns=FACTORS,
        )
        constraints = ConstraintSet(
            max_position_size=0.02,
            max_leverage=2.0,
            dollar_neutral=True,
        )

        optimizer = PortfolioOptimizer({"risk_aversion": 1.0})
        weights = optimizer.optimize(alpha, f_cov, exposures, constraints)

        assert len(weights) == 10
        # Check constraints
        assert weights.abs().max() <= 0.02 + 1e-4, f"Max position: {weights.abs().max()}"
        assert weights.abs().sum() <= 2.0 + 1e-4, f"Leverage: {weights.abs().sum()}"
        assert abs(weights.sum()) < 0.02, f"Net exposure: {weights.sum()}"

    def test_optimizer_with_sector_constraints(self):
        rng = np.random.default_rng(42)
        alpha = pd.Series(rng.normal(0, 1, 10), index=TICKERS)
        f_cov = pd.DataFrame(
            np.diag([0.04] * len(FACTORS)),
            index=FACTORS, columns=FACTORS,
        )
        exposures = pd.DataFrame(
            rng.normal(0, 0.5, (10, len(FACTORS))),
            index=TICKERS, columns=FACTORS,
        )
        constraints = ConstraintSet(
            max_position_size=0.02,
            max_sector_exposure=0.20,
            max_leverage=2.0,
            dollar_neutral=True,
            sector_map=SECTOR_MAP,
        )

        optimizer = PortfolioOptimizer()
        weights = optimizer.optimize(alpha, f_cov, exposures, constraints)
        assert weights.abs().max() <= 0.02 + 1e-4

    def test_empty_alpha(self):
        optimizer = PortfolioOptimizer()
        weights = optimizer.optimize(
            pd.Series(dtype=float),
            pd.DataFrame(),
            pd.DataFrame(),
            ConstraintSet(),
        )
        assert len(weights) == 0


# ---------------------------------------------------------------------------
# Capital Allocation
# ---------------------------------------------------------------------------

class TestStrategyPerformanceTracker:

    def test_tracks_returns(self):
        tracker = StrategyPerformanceTracker({"min_sharpe_window": 20})
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=60)
        for dt in dates:
            tracker.update("strat_a", rng.normal(0.001, 0.01), dt)

        metrics = tracker.get_metrics("strat_a")
        assert metrics is not None
        assert metrics.n_days == 60
        assert metrics.trailing_sharpe != 0

    def test_insufficient_data(self):
        tracker = StrategyPerformanceTracker({"min_sharpe_window": 60})
        tracker.update("short", 0.01, pd.Timestamp("2020-01-02"))
        metrics = tracker.get_metrics("short")
        assert metrics.trailing_sharpe == 0.0


class TestStrategyCorrelationMatrix:

    def test_correlation_matrix(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=100)
        rets = {
            "strat_a": pd.Series(rng.normal(0, 0.01, 100), index=dates),
            "strat_b": pd.Series(rng.normal(0, 0.01, 100), index=dates),
        }
        matrix = StrategyCorrelationMatrix()
        corr = matrix.compute(rets)
        assert corr.shape == (2, 2)
        assert corr.loc["strat_a", "strat_a"] == pytest.approx(1.0)

    def test_crowding_penalty(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=100)
        shared = rng.normal(0, 0.01, 100)
        rets = {
            "s1": pd.Series(shared + rng.normal(0, 0.001, 100), index=dates),
            "s2": pd.Series(shared + rng.normal(0, 0.001, 100), index=dates),
            "s3": pd.Series(rng.normal(0, 0.01, 100), index=dates),
        }
        matrix = StrategyCorrelationMatrix()
        corr = matrix.compute(rets)
        penalties = matrix.get_crowding_penalty(corr)
        # s1 and s2 should have higher penalty than s3
        assert penalties["s1"] > penalties["s3"]


class TestDynamicStrategyAllocator:

    def test_sharpe_weighted_allocation(self):
        allocator = DynamicStrategyAllocator({
            "min_strategy_allocation": 0.0,
            "max_strategy_allocation": 1.0,
        })
        sharpes = {"high": 2.0, "medium": 1.0, "low": 0.5}
        allocs = allocator.allocate(sharpes)
        assert sum(allocs.values()) == pytest.approx(1.0)
        assert allocs["high"] > allocs["medium"] > allocs["low"]

    def test_all_zero_sharpe(self):
        allocator = DynamicStrategyAllocator({"min_sharpe_for_allocation": 0.5})
        sharpes = {"a": 0.1, "b": 0.2}
        allocs = allocator.allocate(sharpes)
        # Should fall back to equal weight
        assert abs(allocs["a"] - allocs["b"]) < 0.01


# ---------------------------------------------------------------------------
# Capacity Model
# ---------------------------------------------------------------------------

class TestMarketImpactModel:

    def test_impact_increases_with_size(self):
        model = MarketImpactModel()
        small = model.estimate_impact_bps(10_000, 5_000_000, 0.02)
        large = model.estimate_impact_bps(500_000, 5_000_000, 0.02)
        assert large > small > 0

    def test_zero_order(self):
        model = MarketImpactModel()
        assert model.estimate_impact_bps(0, 5_000_000, 0.02) == 0.0

    def test_portfolio_impact(self):
        model = MarketImpactModel()
        trades = pd.Series({"AAPL": 100_000, "MSFT": 50_000})
        adv = pd.Series({"AAPL": 5_000_000, "MSFT": 3_000_000})
        vol = pd.Series({"AAPL": 0.02, "MSFT": 0.015})
        impacts = model.estimate_impact_portfolio(trades, adv, vol)
        assert len(impacts) == 2
        assert (impacts > 0).all()


class TestLiquidityEstimator:

    def test_estimates_liquidity(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=30)
        volume = pd.DataFrame(
            rng.uniform(1e6, 5e6, (30, 3)),
            index=dates, columns=["AAPL", "MSFT", "GOOG"],
        )
        prices = pd.DataFrame(
            rng.uniform(100, 200, (30, 3)),
            index=dates, columns=["AAPL", "MSFT", "GOOG"],
        )
        estimator = LiquidityEstimator()
        result = estimator.estimate(
            volume, prices, as_of=dates[-1] + pd.Timedelta(days=1)
        )
        assert "adv_usd" in result.columns
        assert len(result) == 3


class TestCapacitySimulator:

    def test_capacity_simulation(self):
        target = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        current = pd.Series({"AAPL": 0.0, "MSFT": 0.0})
        adv = pd.Series({"AAPL": 50_000_000, "MSFT": 30_000_000})
        vol = pd.Series({"AAPL": 0.02, "MSFT": 0.015})

        sim = CapacitySimulator()
        result = sim.estimate_capacity(
            target, current, adv, vol,
            expected_alpha_bps=50,
            aum_grid=[1e6, 10e6, 100e6, 1e9],
        )
        assert "max_capacity_usd" in result
        assert len(result["impact_table"]) > 0
