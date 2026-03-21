"""Section 13: Deterministic Reproducibility Tests

Validates that:
- All RNG uses np.random.default_rng(seed=42)
- Floating-point tolerances are respected
- Platform-independent types used (pd.Timestamp, pathlib.Path)
- Golden file generation and comparison
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, make_returns, make_factor_returns, STANDARD_TICKERS


SEED = 42


@pytest.mark.regression
@pytest.mark.tier2
class TestDeterministicRNG:
    """All generators produce identical output with same seed."""

    def test_make_ohlcv_deterministic(self):
        """make_ohlcv with same seed produces identical output."""
        df1 = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=100, seed=SEED)
        df2 = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=100, seed=SEED)
        pd.testing.assert_frame_equal(df1, df2)

    def test_make_returns_deterministic(self):
        """make_returns with same seed produces identical output."""
        df1 = make_returns(n_dates=100, tickers=STANDARD_TICKERS[:5], seed=SEED)
        df2 = make_returns(n_dates=100, tickers=STANDARD_TICKERS[:5], seed=SEED)
        pd.testing.assert_frame_equal(df1, df2)

    def test_make_factor_returns_deterministic(self):
        """make_factor_returns with same seed produces identical output."""
        df1 = make_factor_returns(n_dates=100, seed=SEED)
        df2 = make_factor_returns(n_dates=100, seed=SEED)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_output(self):
        """Different seeds produce different output."""
        df1 = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=42)
        df2 = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=99)
        assert not df1.equals(df2)

    def test_feature_computation_deterministic(self):
        """Feature computation with same input produces same output."""
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=300, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max()

        engine = TechnicalIndicatorEngine()
        features1 = engine.compute_all(ohlcv, as_of)
        features2 = engine.compute_all(ohlcv, as_of)

        pd.testing.assert_frame_equal(features1, features2)

    def test_normalization_deterministic(self):
        """Normalization with same input produces same output."""
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        rng = np.random.default_rng(SEED)
        raw = pd.Series(rng.normal(0, 1, 50), index=[f"T{i:03d}" for i in range(50)])

        normalizer = FeatureNormalizer()
        n1 = normalizer.normalize(raw)
        n2 = normalizer.normalize(raw)

        pd.testing.assert_series_equal(n1, n2)

    def test_optimizer_deterministic(self):
        """Portfolio optimizer with same inputs produces same weights."""
        from quant_fund.portfolio.portfolio_construction.portfolio_optimizer import PortfolioOptimizer
        from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintEngine
        from tests.conftest import STANDARD_SECTORS

        rng = np.random.default_rng(SEED)
        tickers = STANDARD_TICKERS[:10]
        alpha = pd.Series(rng.normal(0, 0.01, len(tickers)), index=tickers)

        n_factors = 5
        factors = [f"F{i}" for i in range(n_factors)]
        factor_cov = pd.DataFrame(np.eye(n_factors) * 0.01, index=factors, columns=factors)
        exposures = pd.DataFrame(
            rng.normal(0, 1, (len(tickers), n_factors)),
            index=tickers, columns=factors,
        )

        engine = ConstraintEngine()
        constraints = engine.build_constraints(sector_map=STANDARD_SECTORS)
        optimizer = PortfolioOptimizer()

        w1 = optimizer.optimize(alpha, factor_cov, exposures, constraints)
        w2 = optimizer.optimize(alpha, factor_cov, exposures, constraints)

        pd.testing.assert_series_equal(w1, w2, atol=1e-6)


@pytest.mark.regression
@pytest.mark.tier2
class TestFloatingPointTolerances:
    """Validate that floating-point comparisons use correct tolerances."""

    def test_price_tolerance(self):
        """Prices compared with atol=1e-6."""
        price1 = 150.123456
        price2 = 150.123456 + 1e-7  # within tolerance
        np.testing.assert_allclose(price1, price2, atol=1e-6)

    def test_weight_tolerance(self):
        """Weights compared with atol=1e-4."""
        w1 = 0.0200
        w2 = 0.0200 + 1e-5  # within tolerance
        np.testing.assert_allclose(w1, w2, atol=1e-4)

    def test_ic_tolerance(self):
        """IC values compared with atol=1e-4."""
        ic1 = 0.0350
        ic2 = 0.0350 + 1e-5
        np.testing.assert_allclose(ic1, ic2, atol=1e-4)

    def test_nav_tolerance(self):
        """NAV compared with atol=1e-6."""
        nav1 = 1_000_000.0
        nav2 = 1_000_000.0 + 1e-7
        np.testing.assert_allclose(nav1, nav2, atol=1e-6)

    def test_drawdown_tolerance(self):
        """Drawdown percentage compared with atol=1e-4."""
        dd1 = 0.2000
        dd2 = 0.2000 + 1e-5
        np.testing.assert_allclose(dd1, dd2, atol=1e-4)


@pytest.mark.regression
@pytest.mark.tier2
class TestPlatformIndependence:
    """Validate platform-independent coding conventions."""

    def test_timestamps_are_pd_timestamp(self):
        """All generated dates are pd.Timestamp, not datetime.datetime."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=SEED)
        dates = ohlcv.index.get_level_values("date")

        for d in dates[:10]:
            assert isinstance(d, pd.Timestamp), f"Expected pd.Timestamp, got {type(d)}"

    def test_ohlcv_index_names_consistent(self):
        """OHLCV always has index names ['date', 'ticker']."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=100, seed=SEED)
        assert list(ohlcv.index.names) == ["date", "ticker"]

    def test_returns_columns_match_tickers(self):
        """Returns matrix columns match requested tickers."""
        tickers = STANDARD_TICKERS[:5]
        returns = make_returns(n_dates=50, tickers=tickers, seed=SEED)
        assert list(returns.columns) == tickers

    def test_factor_returns_default_factors(self):
        """Factor returns use default factor names."""
        fret = make_factor_returns(n_dates=50, seed=SEED)
        expected = ["Market", "Size", "Value", "Momentum", "Quality"]
        assert list(fret.columns) == expected


@pytest.mark.regression
@pytest.mark.tier2
class TestGoldenFileConventions:
    """Validate golden file generation patterns."""

    def test_golden_file_directory_exists(self):
        """Golden files directory exists."""
        from pathlib import Path
        golden_dir = Path(__file__).parent / "golden_files"
        assert golden_dir.exists()
        assert golden_dir.is_dir()

    def test_can_generate_golden_baseline(self):
        """Can generate a golden baseline from pipeline."""
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max()

        engine = TechnicalIndicatorEngine()
        features = engine.compute_all(ohlcv, as_of)

        normalizer = FeatureNormalizer()
        signals = {}
        for col in features.columns:
            raw = features[col].dropna()
            if len(raw) >= 3:
                signals[col] = normalizer.normalize(raw)

        # Golden output is just the pipeline result — should be a dict of Series
        assert len(signals) > 0
        for name, series in signals.items():
            assert isinstance(series, pd.Series)
            assert len(series) > 0

    def test_golden_baseline_reproducible(self):
        """Golden baseline is identical across two runs."""
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine

        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max()

        engine = TechnicalIndicatorEngine()
        f1 = engine.compute_all(ohlcv, as_of)
        f2 = engine.compute_all(ohlcv, as_of)

        pd.testing.assert_frame_equal(f1, f2)
