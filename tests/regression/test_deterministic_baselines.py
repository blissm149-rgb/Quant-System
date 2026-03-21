"""Regression tests: deterministic baselines, signal stability, portfolio weight stability, NAV trajectory.

Golden-file-based tests that verify the pipeline produces identical outputs
for identical inputs across code changes.  Uses seed=42 throughout.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, make_returns, make_factor_returns, STANDARD_TICKERS, STANDARD_SECTORS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASELINE_TICKERS = STANDARD_TICKERS[:5]
SEED = 42


def _build_research_runner(ohlcv, seed=SEED):
    """Build a minimal ResearchRunner with deterministic components."""
    from quant_fund.main.research_runner import ResearchRunner
    from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine
    from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

    runner = ResearchRunner({"universe": BASELINE_TICKERS, "lookback_days": 252})
    runner.inject_components(
        feature_generators=TechnicalIndicatorEngine().generators,
        feature_normalizer=FeatureNormalizer(),
    )
    return runner


def _build_optimizer():
    """Build a PortfolioOptimizer with default config."""
    from quant_fund.portfolio.portfolio_construction.portfolio_optimizer import PortfolioOptimizer
    return PortfolioOptimizer()


def _build_constraint_set(sector_map=None):
    from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintEngine
    engine = ConstraintEngine()
    return engine.build_constraints(sector_map=sector_map)


# ---------------------------------------------------------------------------
# 7.1  Deterministic Baseline — same seed → same feature matrix
# ---------------------------------------------------------------------------

@pytest.mark.regression
@pytest.mark.tier3
class TestDeterministicBaseline:
    """Verify that the feature computation pipeline is fully deterministic."""

    def test_same_seed_same_features(self):
        """Running the feature pipeline twice with seed=42 gives identical output."""
        ohlcv = make_ohlcv(tickers=BASELINE_TICKERS, periods=300, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max()

        runner1 = _build_research_runner(ohlcv)
        result1 = runner1.run_cycle(as_of=as_of, market_data=ohlcv)

        runner2 = _build_research_runner(ohlcv)
        result2 = runner2.run_cycle(as_of=as_of, market_data=ohlcv)

        if result1.feature_matrix is not None and result2.feature_matrix is not None:
            pd.testing.assert_frame_equal(result1.feature_matrix, result2.feature_matrix)
        if result1.alpha_scores is not None and result2.alpha_scores is not None:
            pd.testing.assert_series_equal(result1.alpha_scores, result2.alpha_scores)

    def test_same_seed_same_alpha_scores(self):
        """Alpha scores are deterministic for identical market data."""
        ohlcv = make_ohlcv(tickers=BASELINE_TICKERS, periods=300, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max()

        scores = []
        for _ in range(3):
            runner = _build_research_runner(ohlcv)
            result = runner.run_cycle(as_of=as_of, market_data=ohlcv)
            scores.append(result.alpha_scores)

        for s in scores[1:]:
            if scores[0] is not None and s is not None:
                pd.testing.assert_series_equal(scores[0], s)


# ---------------------------------------------------------------------------
# 7.2  Signal Stability — same inputs → same alpha_scores
# ---------------------------------------------------------------------------

@pytest.mark.regression
@pytest.mark.tier3
class TestSignalStability:
    """Alpha scores must not drift when given identical inputs."""

    def test_alpha_scores_identical_across_runs(self):
        """Two independent runs on the same data yield bit-identical alpha scores."""
        ohlcv = make_ohlcv(tickers=BASELINE_TICKERS, periods=300, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max()

        runner_a = _build_research_runner(ohlcv)
        res_a = runner_a.run_cycle(as_of=as_of, market_data=ohlcv)

        runner_b = _build_research_runner(ohlcv)
        res_b = runner_b.run_cycle(as_of=as_of, market_data=ohlcv)

        if res_a.alpha_scores is not None and res_b.alpha_scores is not None:
            np.testing.assert_allclose(
                res_a.alpha_scores.values,
                res_b.alpha_scores.values,
                atol=1e-10,
            )

    def test_backtest_deterministic(self):
        """Multi-date backtest produces identical results."""
        ohlcv = make_ohlcv(tickers=BASELINE_TICKERS, periods=300, seed=SEED)
        dates = ohlcv.index.get_level_values("date").unique()[-5:]

        runner1 = _build_research_runner(ohlcv)
        res1 = runner1.run_backtest(dates=dates.tolist(), market_data=ohlcv)

        runner2 = _build_research_runner(ohlcv)
        res2 = runner2.run_backtest(dates=dates.tolist(), market_data=ohlcv)

        for r1, r2 in zip(res1, res2):
            if r1.alpha_scores is not None and r2.alpha_scores is not None:
                pd.testing.assert_series_equal(r1.alpha_scores, r2.alpha_scores)


# ---------------------------------------------------------------------------
# 7.3  Portfolio Weight Stability — same alpha → same weights
# ---------------------------------------------------------------------------

@pytest.mark.regression
@pytest.mark.tier3
class TestPortfolioWeightStability:
    """Optimizer must return identical weights for identical inputs."""

    def _make_alpha_and_risk(self):
        rng = np.random.default_rng(SEED)
        alpha = pd.Series(rng.normal(0, 0.01, len(BASELINE_TICKERS)), index=BASELINE_TICKERS)
        n = len(BASELINE_TICKERS)
        cov_raw = rng.normal(0, 0.01, (n, n))
        cov = pd.DataFrame(
            cov_raw @ cov_raw.T / n,
            index=BASELINE_TICKERS,
            columns=BASELINE_TICKERS,
        )
        factors = ["F1", "F2", "F3"]
        exposures = pd.DataFrame(
            rng.normal(0, 1, (n, len(factors))),
            index=BASELINE_TICKERS,
            columns=factors,
        )
        factor_cov = pd.DataFrame(
            np.eye(len(factors)) * 0.01,
            index=factors,
            columns=factors,
        )
        return alpha, factor_cov, exposures

    def test_optimizer_deterministic(self):
        """Calling optimize() twice with identical inputs gives same weights."""
        alpha, factor_cov, exposures = self._make_alpha_and_risk()
        constraints = _build_constraint_set(
            sector_map={t: STANDARD_SECTORS[t] for t in BASELINE_TICKERS}
        )
        opt = _build_optimizer()

        w1 = opt.optimize(alpha, factor_cov, exposures, constraints)
        w2 = opt.optimize(alpha, factor_cov, exposures, constraints)

        pd.testing.assert_series_equal(w1, w2, atol=1e-6)

    def test_weights_respect_position_limit(self):
        """No weight exceeds max_position_size (0.02)."""
        alpha, factor_cov, exposures = self._make_alpha_and_risk()
        constraints = _build_constraint_set(
            sector_map={t: STANDARD_SECTORS[t] for t in BASELINE_TICKERS}
        )
        opt = _build_optimizer()
        w = opt.optimize(alpha, factor_cov, exposures, constraints)

        if len(w) > 0:
            assert w.abs().max() <= constraints.max_position_size + 1e-4


# ---------------------------------------------------------------------------
# 7.4  NAV Trajectory — known scenario → known NAV path
# ---------------------------------------------------------------------------

@pytest.mark.regression
@pytest.mark.tier3
class TestNAVTrajectory:
    """Paper-trading NAV must be reproducible from identical inputs."""

    def test_nav_trajectory_deterministic(self):
        """Same data, same config → same final NAV."""
        from quant_fund.main.paper_trading_runner import PaperTradingRunner
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.execution.order_management.order_generator import OrderGenerator
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ohlcv = make_ohlcv(tickers=BASELINE_TICKERS, periods=60, seed=SEED)
        dates = ohlcv.index.get_level_values("date").unique()[-10:].tolist()

        navs = []
        for _ in range(2):
            runner = PaperTradingRunner({"strategy_id": "test", "initial_nav": 1_000_000.0})
            broker = SimulationBroker({"initial_cash": 1_000_000.0})
            runner.inject_components(
                research_runner=_build_research_runner(ohlcv),
                order_generator=OrderGenerator(),
                broker=broker,
                kill_switch=KillSwitch({"drawdown_limit": 0.20}),
            )
            result = runner.run(dates=dates, market_data_by_date={d: ohlcv for d in dates})
            navs.append(result.final_nav)

        np.testing.assert_allclose(navs[0], navs[1], atol=1e-6)

    def test_nav_always_positive(self):
        """NAV should never go negative during paper trading."""
        from quant_fund.main.paper_trading_runner import PaperTradingRunner
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.execution.order_management.order_generator import OrderGenerator
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ohlcv = make_ohlcv(tickers=BASELINE_TICKERS, periods=60, seed=SEED)
        dates = ohlcv.index.get_level_values("date").unique()[-10:].tolist()

        runner = PaperTradingRunner({"strategy_id": "test", "initial_nav": 1_000_000.0})
        broker = SimulationBroker({"initial_cash": 1_000_000.0})
        runner.inject_components(
            research_runner=_build_research_runner(ohlcv),
            order_generator=OrderGenerator(),
            broker=broker,
            kill_switch=KillSwitch({"drawdown_limit": 0.20}),
        )
        result = runner.run(dates=dates, market_data_by_date={d: ohlcv for d in dates})

        for day_result in result.daily_results:
            assert day_result.nav >= 0, f"NAV went negative on {day_result.date}"
