"""Integration Chain 1: Data → Features → Alpha

Tests the pipeline: historical_data_loader → corporate_action_adjuster →
data_validator → data_alignment_engine → technical_indicator_engine →
feature_normalizer → signal_ranking_engine (combine)
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, STANDARD_TICKERS, STANDARD_SECTORS


@pytest.mark.integration
@pytest.mark.tier3
class TestDataToFeaturesChain:
    """Raw OHLCV flows through the pipeline to computed features."""

    def test_ohlcv_through_validator_to_features(self):
        """OHLCV → DataValidator → TechnicalIndicatorEngine produces feature matrix."""
        from quant_fund.data_layer.data_validator import DataValidator
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine

        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=42)
        as_of = ohlcv.index.get_level_values("date").max() + pd.Timedelta(days=1)

        # Step 1: Validate
        validator = DataValidator()
        result = validator.validate(ohlcv, as_of=as_of)
        assert result.is_valid, f"Validation failed: {result.errors}"

        # Step 2: Compute features
        engine = TechnicalIndicatorEngine()
        features = engine.compute_all(ohlcv, as_of=ohlcv.index.get_level_values("date").max())

        assert features is not None
        assert len(features) == len(tickers)
        assert features.shape[1] > 0  # at least one feature column

    def test_adjusted_prices_used_for_features(self):
        """Corporate action adjustment flows into feature computation."""
        from quant_fund.data_layer.corporate_action_adjuster import (
            CorporateActionAdjuster, CorporateAction,
        )
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine

        tickers = STANDARD_TICKERS[:3]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=42)

        # Register a 2:1 split for the first ticker
        adjuster = CorporateActionAdjuster()
        dates = ohlcv.index.get_level_values("date").unique()
        split_date = dates[150]
        adjuster.register_actions([
            CorporateAction(
                ticker=tickers[0],
                ex_date=split_date,
                action_type="split",
                adjustment_factor=2.0,
            )
        ])

        adjusted = adjuster.adjust_dataframe(ohlcv)
        as_of = dates[-1]

        # Features from adjusted data should differ from raw
        engine = TechnicalIndicatorEngine()
        features_raw = engine.compute_all(ohlcv, as_of)
        features_adj = engine.compute_all(adjusted, as_of)

        # At minimum, the split ticker's features should differ
        if features_raw is not None and features_adj is not None:
            assert not features_raw.equals(features_adj) or tickers[0] not in features_raw.index

    def test_no_lookahead_in_feature_pipeline(self):
        """Features computed at as_of don't use future data."""
        from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine

        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=42)
        dates = ohlcv.index.get_level_values("date").unique()
        mid_date = dates[200]
        as_of = mid_date + pd.Timedelta(days=1)

        # Use permissive mode to filter out future data (simulating real pipeline)
        alignment = DataAlignmentEngine()
        aligned = alignment.get_aligned_data_permissive(ohlcv, as_of=as_of, lookback_days=252)

        # All aligned data should be before as_of
        aligned_dates = aligned.index.get_level_values("date")
        assert aligned_dates.max() < as_of

        # Features should compute on aligned data
        engine = TechnicalIndicatorEngine()
        features = engine.compute_all(aligned, as_of=mid_date)
        assert features is not None


@pytest.mark.integration
@pytest.mark.tier3
class TestFeaturesToAlpha:
    """Normalized features flow into alpha score generation."""

    def test_normalized_features_have_zero_mean_unit_std(self):
        """FeatureNormalizer produces zero-mean, unit-std features."""
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:10], periods=300, seed=42)
        as_of = ohlcv.index.get_level_values("date").max()

        engine = TechnicalIndicatorEngine()
        features = engine.compute_all(ohlcv, as_of)
        assert features is not None

        normalizer = FeatureNormalizer()
        for col in features.columns:
            raw = features[col].dropna()
            if len(raw) < 3:
                continue
            normalized = normalizer.normalize(raw, method="zscore")
            # Mean should be approximately 0
            assert abs(normalized.mean()) < 0.1, f"{col}: mean={normalized.mean():.4f}"
            # Std should be approximately 1 (unless all values identical)
            if raw.std() > 1e-10:
                assert abs(normalized.std() - 1.0) < 0.3, f"{col}: std={normalized.std():.4f}"

    def test_signal_combination_produces_alpha_scores(self):
        """SignalRankingEngine.combine() merges signals into alpha scores."""
        from quant_fund.alpha_discovery.signal_ranking_engine import SignalRankingEngine
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        tickers = STANDARD_TICKERS[:10]
        rng = np.random.default_rng(42)

        # Create fake normalized signals
        normalizer = FeatureNormalizer()
        signals = {}
        for name in ["momentum", "value", "quality"]:
            raw = pd.Series(rng.normal(0, 1, len(tickers)), index=tickers)
            signals[name] = normalizer.normalize(raw)

        ranker = SignalRankingEngine()
        alpha = ranker.combine(signals)

        assert isinstance(alpha, pd.Series)
        assert len(alpha) == len(tickers)
        assert not alpha.isna().all()

    def test_full_data_to_alpha_pipeline(self):
        """Full chain: OHLCV → validate → features → normalize → combine."""
        from quant_fund.data_layer.data_validator import DataValidator
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer
        from quant_fund.alpha_discovery.signal_ranking_engine import SignalRankingEngine

        tickers = STANDARD_TICKERS[:10]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=42)
        as_of = ohlcv.index.get_level_values("date").max()

        # Validate
        validator = DataValidator()
        val_result = validator.validate(ohlcv, as_of=as_of + pd.Timedelta(days=1))
        assert val_result.is_valid

        # Compute features
        engine = TechnicalIndicatorEngine()
        features = engine.compute_all(ohlcv, as_of)
        assert features is not None

        # Normalize each feature column
        normalizer = FeatureNormalizer()
        signals = {}
        for col in features.columns:
            raw = features[col].dropna()
            if len(raw) >= 3:
                signals[col] = normalizer.normalize(raw)

        assert len(signals) > 0

        # Combine into alpha
        ranker = SignalRankingEngine()
        alpha = ranker.combine(signals)

        assert isinstance(alpha, pd.Series)
        assert len(alpha) > 0
        assert not alpha.isna().all()
