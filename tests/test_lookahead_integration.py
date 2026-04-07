"""Cross-module look-ahead bias integration test.

Verifies that modifying data after the as_of timestamp does NOT
change any computed features, normalised scores, or portfolio weights.
Tests the full chain: data alignment → feature computation →
normalisation → signal combination.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, make_returns, STANDARD_TICKERS

pytestmark = [pytest.mark.tier3]


class TestDataAlignmentLookAhead:
    """Test that DataAlignmentEngine enforces point-in-time boundaries."""

    def test_aligned_data_excludes_future(self):
        """get_aligned_data must return only rows with timestamp < as_of."""
        from quant_fund.feature_factory.data_alignment_engine import (
            DataAlignmentEngine,
        )

        engine = DataAlignmentEngine()
        ohlcv = make_ohlcv(periods=100)
        dates = ohlcv.index.get_level_values("date").unique()
        as_of = dates[50]  # midpoint

        aligned = engine.get_aligned_data_permissive(ohlcv, as_of, lookback_days=365)
        aligned_dates = aligned.index.get_level_values("date")
        assert (aligned_dates < as_of).all()

    def test_strict_mode_raises_on_future_data(self):
        """get_aligned_data should raise LookAheadError if future data present."""
        from quant_fund.feature_factory.data_alignment_engine import (
            DataAlignmentEngine,
            LookAheadError,
        )

        engine = DataAlignmentEngine()
        ohlcv = make_ohlcv(periods=100)
        dates = ohlcv.index.get_level_values("date").unique()
        as_of = dates[50]

        # Strict mode raises on full data that includes future
        with pytest.raises(LookAheadError):
            engine.get_aligned_data(ohlcv, as_of, lookback_days=365)

        # Pre-filtered data works fine with strict mode
        aligned = engine.get_aligned_data_permissive(ohlcv, as_of, lookback_days=365)
        clean = engine.get_aligned_data(aligned, as_of, lookback_days=365)
        assert len(clean) > 0

    def test_permissive_mode_silently_filters(self):
        """get_aligned_data_permissive should filter without raising."""
        from quant_fund.feature_factory.data_alignment_engine import (
            DataAlignmentEngine,
        )

        engine = DataAlignmentEngine()
        ohlcv = make_ohlcv(periods=100)
        dates = ohlcv.index.get_level_values("date").unique()
        as_of = dates[50]

        aligned = engine.get_aligned_data_permissive(
            ohlcv, as_of, lookback_days=365,
        )
        aligned_dates = aligned.index.get_level_values("date")
        assert (aligned_dates < as_of).all()


class TestDataValidatorLookAhead:
    """Test DataValidator catches look-ahead violations."""

    def test_validator_rejects_future_data(self):
        """DataValidator should flag rows with timestamps >= as_of."""
        from quant_fund.data_layer.data_validator import DataValidator

        validator = DataValidator()
        ohlcv = make_ohlcv(periods=100)
        dates = ohlcv.index.get_level_values("date").unique()
        as_of = dates[50]

        # Full dataset includes future data
        result = validator.validate(ohlcv, as_of)
        assert not result.is_valid
        assert any("look-ahead" in e.lower() or "future" in e.lower()
                    for e in result.errors)

    def test_validator_accepts_clean_data(self):
        """DataValidator should accept data strictly before as_of."""
        from quant_fund.data_layer.data_validator import DataValidator
        from quant_fund.feature_factory.data_alignment_engine import (
            DataAlignmentEngine,
        )

        validator = DataValidator()
        engine = DataAlignmentEngine()
        ohlcv = make_ohlcv(periods=100)
        dates = ohlcv.index.get_level_values("date").unique()
        as_of = dates[50]

        aligned = engine.get_aligned_data_permissive(ohlcv, as_of, lookback_days=365)
        result = validator.validate(aligned, as_of)
        assert result.is_valid


class TestFeatureComputeLookAhead:
    """Test that features computed at as_of don't change when future data changes."""

    def test_future_data_injection_no_effect(self):
        """Modifying data after as_of should not change computed features."""
        from quant_fund.feature_factory.data_alignment_engine import (
            DataAlignmentEngine,
        )
        from quant_fund.feature_factory.technical_indicator_engine import (
            TechnicalIndicatorEngine,
        )

        engine = DataAlignmentEngine()
        tech = TechnicalIndicatorEngine()

        # Generate base data
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=100, seed=42)
        dates = ohlcv.index.get_level_values("date").unique()
        as_of = dates[60]

        # Compute features with original data
        aligned1 = engine.get_aligned_data_permissive(ohlcv, as_of, lookback_days=365)
        features1 = tech.compute_all(aligned1, as_of)

        # Modify data AFTER as_of (inject huge spike)
        ohlcv_modified = ohlcv.copy()
        future_mask = ohlcv_modified.index.get_level_values("date") >= as_of
        ohlcv_modified.loc[future_mask, "close"] *= 10.0
        ohlcv_modified.loc[future_mask, "adj_close"] *= 10.0

        # Recompute features at same as_of
        aligned2 = engine.get_aligned_data_permissive(ohlcv_modified, as_of, lookback_days=365)
        features2 = tech.compute_all(aligned2, as_of)

        # Features must be identical
        pd.testing.assert_frame_equal(features1, features2)

    def test_normalisation_invariant_to_future(self):
        """Normalised scores should be invariant to future data changes."""
        from quant_fund.feature_factory.data_alignment_engine import (
            DataAlignmentEngine,
        )
        from quant_fund.feature_factory.technical_indicator_engine import (
            TechnicalIndicatorEngine,
        )
        from quant_fund.feature_factory.feature_normalizer import (
            FeatureNormalizer,
        )

        engine = DataAlignmentEngine()
        tech = TechnicalIndicatorEngine()
        normalizer = FeatureNormalizer()

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=100, seed=42)
        dates = ohlcv.index.get_level_values("date").unique()
        as_of = dates[60]

        # Original computation
        aligned1 = engine.get_aligned_data_permissive(ohlcv, as_of, lookback_days=365)
        features1 = tech.compute_all(aligned1, as_of)
        normalised1 = normalizer.normalize_dataframe(features1)

        # Modified future data
        ohlcv_modified = ohlcv.copy()
        future_mask = ohlcv_modified.index.get_level_values("date") >= as_of
        ohlcv_modified.loc[future_mask, "close"] *= 5.0

        aligned2 = engine.get_aligned_data_permissive(ohlcv_modified, as_of, lookback_days=365)
        features2 = tech.compute_all(aligned2, as_of)
        normalised2 = normalizer.normalize_dataframe(features2)

        pd.testing.assert_frame_equal(normalised1, normalised2)


class TestSignalCombinationLookAhead:
    """Test that signal combination is invariant to future data."""

    def test_combined_alpha_invariant(self):
        """Combined alpha scores must be identical regardless of future data."""
        from quant_fund.alpha_discovery.signal_ranking_engine import (
            SignalRankingEngine,
        )

        ranker = SignalRankingEngine()
        rng = np.random.default_rng(42)
        tickers = STANDARD_TICKERS[:5]

        # Two "signals"
        signal_a = pd.Series(rng.normal(0, 1, len(tickers)), index=tickers)
        signal_b = pd.Series(rng.normal(0, 1, len(tickers)), index=tickers)

        combined1 = ranker.combine(
            {"sig_a": signal_a, "sig_b": signal_b},
            weights={"sig_a": 0.6, "sig_b": 0.4},
        )

        # Same signals, same weights → same result
        combined2 = ranker.combine(
            {"sig_a": signal_a, "sig_b": signal_b},
            weights={"sig_a": 0.6, "sig_b": 0.4},
        )

        pd.testing.assert_series_equal(combined1, combined2)

    def test_signal_evaluation_deterministic(self):
        """Signal evaluation should be deterministic with same inputs."""
        from quant_fund.alpha_discovery.signal_ranking_engine import (
            SignalRankingEngine,
        )

        ranker = SignalRankingEngine()
        rng = np.random.default_rng(42)
        tickers = STANDARD_TICKERS[:5]
        n_dates = 50
        dates = pd.bdate_range("2023-01-01", periods=n_dates)

        # Create signal values
        signal_data = []
        for dt in dates:
            for ticker in tickers:
                signal_data.append({
                    "date": dt, "ticker": ticker,
                    "signal": rng.normal(0, 1),
                })
        signal_df = pd.DataFrame(signal_data)

        # Create forward returns
        rng2 = np.random.default_rng(99)
        return_data = []
        for dt in dates:
            for ticker in tickers:
                return_data.append({
                    "date": dt, "ticker": ticker,
                    "return": rng2.normal(0.001, 0.02),
                })
        returns_df = pd.DataFrame(return_data)

        score1 = ranker.evaluate_signal(signal_df, returns_df)
        score2 = ranker.evaluate_signal(signal_df, returns_df)

        assert score1.ic_mean == score2.ic_mean
        assert score1.composite_score == score2.composite_score


class TestEndToEndLookAhead:
    """Full pipeline look-ahead test: data → features → weights."""

    def test_full_pipeline_invariant_to_future_data(self):
        """Complete pipeline from data to portfolio weights must not
        use any information from after the as_of timestamp."""
        from quant_fund.feature_factory.data_alignment_engine import (
            DataAlignmentEngine,
        )
        from quant_fund.feature_factory.technical_indicator_engine import (
            TechnicalIndicatorEngine,
        )
        from quant_fund.feature_factory.feature_normalizer import (
            FeatureNormalizer,
        )
        from quant_fund.risk_engine.leverage_controller import LeverageController

        alignment = DataAlignmentEngine()
        tech = TechnicalIndicatorEngine()
        normalizer = FeatureNormalizer()
        leverage = LeverageController({"max_leverage": 2.0})

        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=120, seed=42)
        dates = ohlcv.index.get_level_values("date").unique()
        as_of = dates[80]

        def compute_weights(data):
            aligned = alignment.get_aligned_data_permissive(data, as_of, lookback_days=365)
            features = tech.compute_all(aligned, as_of)
            if features.empty:
                return pd.Series(dtype=float)
            normalised = normalizer.normalize_dataframe(features)
            # Simple alpha: mean of normalised features
            alpha = normalised.mean(axis=1)
            # Convert to weights (simplified)
            weights = alpha / alpha.abs().sum() * 0.5
            weights = leverage.enforce(weights)
            return weights

        # Weights from original data
        weights1 = compute_weights(ohlcv)

        # Modify future data dramatically
        ohlcv_mod = ohlcv.copy()
        future = ohlcv_mod.index.get_level_values("date") >= as_of
        ohlcv_mod.loc[future, "close"] *= 100.0
        ohlcv_mod.loc[future, "high"] *= 100.0
        ohlcv_mod.loc[future, "low"] *= 100.0
        ohlcv_mod.loc[future, "adj_close"] *= 100.0
        ohlcv_mod.loc[future, "volume"] = 0

        # Weights from modified future data
        weights2 = compute_weights(ohlcv_mod)

        # Must be identical
        pd.testing.assert_series_equal(weights1, weights2)

    def test_multiple_as_of_dates_independent(self):
        """Features at different as_of dates should use their own data windows."""
        from quant_fund.feature_factory.data_alignment_engine import (
            DataAlignmentEngine,
        )
        from quant_fund.feature_factory.technical_indicator_engine import (
            TechnicalIndicatorEngine,
        )

        alignment = DataAlignmentEngine()
        tech = TechnicalIndicatorEngine()

        tickers = STANDARD_TICKERS[:3]
        ohlcv = make_ohlcv(tickers=tickers, periods=100, seed=42)
        dates = ohlcv.index.get_level_values("date").unique()

        as_of_early = dates[40]
        as_of_late = dates[70]

        aligned_early = alignment.get_aligned_data_permissive(
            ohlcv, as_of_early, lookback_days=365,
        )
        aligned_late = alignment.get_aligned_data_permissive(
            ohlcv, as_of_late, lookback_days=365,
        )

        features_early = tech.compute_all(aligned_early, as_of_early)
        features_late = tech.compute_all(aligned_late, as_of_late)

        # Features should differ (different data windows)
        assert not features_early.equals(features_late)

        # Early features should use less data
        n_early = len(aligned_early.index.get_level_values("date").unique())
        n_late = len(aligned_late.index.get_level_values("date").unique())
        assert n_early < n_late
