"""Unit tests for feature_normalizer module.

TESTING_PLAN.md Section 3.3 — Feature Factory.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer


@pytest.mark.unit
@pytest.mark.tier1
class TestFeatureNormalizer:
    """FeatureNormalizer — z-score, rank, percentile normalization."""

    @pytest.fixture
    def normalizer(self):
        return FeatureNormalizer()

    @pytest.fixture
    def raw_scores(self):
        rng = np.random.default_rng(42)
        tickers = [f"T{i:03d}" for i in range(100)]
        return pd.Series(rng.normal(50, 20, 100), index=tickers, name="signal")

    def test_zscore_mean_near_zero(self, normalizer, raw_scores):
        """Z-score output has mean approximately 0."""
        result = normalizer.normalize(raw_scores, method="zscore")
        assert abs(result.mean()) < 0.01

    def test_zscore_std_near_one(self, normalizer, raw_scores):
        """Z-score output has std approximately 1 (after winsorization)."""
        result = normalizer.normalize(raw_scores, method="zscore")
        assert abs(result.std() - 1.0) < 0.2  # allow some tolerance due to winsorization

    def test_winsorization_clips_extremes(self, normalizer):
        """Winsorization reduces the range of extreme outliers."""
        rng = np.random.default_rng(42)
        tickers = [f"T{i:03d}" for i in range(100)]
        scores = pd.Series(rng.standard_normal(100), index=tickers)
        scores.iloc[0] = 1000.0  # extreme outlier
        scores.iloc[1] = -1000.0
        result = normalizer.normalize(scores, method="zscore", winsorize_std=3.0)
        # After winsorization, range should be much smaller than raw z-scores of ±1000
        assert result.max() < 50
        assert result.min() > -50

    def test_rank_normalization_uniform(self, normalizer, raw_scores):
        """Rank normalization produces values in [-1, 1]."""
        result = normalizer.normalize(raw_scores, method="rank")
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_percentile_normalization_bounded(self, normalizer, raw_scores):
        """Percentile normalization produces values in [0, 1]."""
        result = normalizer.normalize(raw_scores, method="percentile")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_unknown_method_raises(self, normalizer, raw_scores):
        """Unknown normalization method raises ValueError."""
        with pytest.raises(ValueError):
            normalizer.normalize(raw_scores, method="unknown_method")

    def test_normalize_dataframe(self, normalizer):
        """normalize_dataframe applies to each column."""
        rng = np.random.default_rng(42)
        tickers = [f"T{i:03d}" for i in range(50)]
        df = pd.DataFrame({
            "signal_a": rng.normal(0, 1, 50),
            "signal_b": rng.normal(10, 5, 50),
        }, index=tickers)
        result = normalizer.normalize_dataframe(df, method="zscore")
        assert result.shape == df.shape
        for col in result.columns:
            assert abs(result[col].mean()) < 0.1

    def test_preserves_index(self, normalizer, raw_scores):
        """Normalized output preserves the original index."""
        result = normalizer.normalize(raw_scores, method="zscore")
        assert list(result.index) == list(raw_scores.index)

    def test_single_value_series(self, normalizer):
        """Single-value series normalizes without error."""
        scores = pd.Series([42.0], index=["AAPL"])
        result = normalizer.normalize(scores, method="zscore")
        assert len(result) == 1
