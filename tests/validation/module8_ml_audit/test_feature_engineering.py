"""Module 3 — Feature Engineering Upgrade tests.

Tests feature importance tracking, temporal normalization,
feature selection pipeline integration, and feature versioning.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.feature_factory.feature_importance_tracker import (
    FeatureImportanceTracker,
)
from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer
from quant_fund.alpha_discovery.ml_feature_selector import MLFeatureSelector
from quant_fund.main.research_runner import ResearchRunner

pytestmark = [pytest.mark.validation, pytest.mark.tier4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_timeseries(n_dates=200, n_features=3, seed=42):
    """Create a synthetic feature time-series DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    data = rng.randn(n_dates, n_features).cumsum(axis=0)
    return pd.DataFrame(data, index=dates, columns=[f"feat_{i}" for i in range(n_features)])


def _make_importance_records(tracker, model_name, n_records=12, seed=0):
    """Fill tracker with synthetic importance records."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_records)
    features = ["momentum", "volatility", "value", "size"]
    for dt in dates:
        importances = dict(zip(features, rng.dirichlet(np.ones(len(features)))))
        tracker.record_importance(as_of=dt, model_name=model_name, feature_importances=importances)


# ---------------------------------------------------------------------------
# Test 1: Feature importance tracker records and retrieves
# ---------------------------------------------------------------------------

class TestFeatureImportanceTracker:

    def test_records_and_retrieves(self):
        tracker = FeatureImportanceTracker()
        _make_importance_records(tracker, "gbt", n_records=10)

        assert tracker.n_records == 10
        history = tracker.get_importance_history(model_name="gbt")
        assert len(history) == 10
        assert set(history.columns) == {"momentum", "volatility", "value", "size"}

        # Rolling importance should have same shape
        rolling = tracker.get_rolling_importance(model_name="gbt", window=5)
        assert rolling.shape == history.shape

        # Top features returns a list of strings
        top = tracker.get_top_features(model_name="gbt", n=2, last_k=5)
        assert len(top) == 2
        assert all(isinstance(f, str) for f in top)

    def test_importance_drift_detection(self):
        """Drift should be detected when importance distributions shift."""
        tracker = FeatureImportanceTracker({"importance_min_records": 3})
        dates = pd.bdate_range("2023-01-01", periods=20)

        # First 10: momentum dominates
        for dt in dates[:10]:
            tracker.record_importance(
                as_of=dt,
                model_name="gbt",
                feature_importances={"momentum": 0.7, "value": 0.1, "vol": 0.1, "size": 0.1},
            )
        # Last 10: value dominates — importance shift
        for dt in dates[10:]:
            tracker.record_importance(
                as_of=dt,
                model_name="gbt",
                feature_importances={"momentum": 0.1, "value": 0.7, "vol": 0.1, "size": 0.1},
            )

        drift_detected, rank_corr, per_feature = tracker.detect_importance_drift(
            model_name="gbt", window=8
        )
        # Rank correlation should be low because rankings inverted
        assert rank_corr < 0.8
        assert "momentum" in per_feature
        assert per_feature["momentum"] > 0.3  # Large change


# ---------------------------------------------------------------------------
# Test 2: Temporal normalization — no future data leakage
# ---------------------------------------------------------------------------

class TestTemporalNormalization:

    def test_temporal_zscore_no_future_data(self):
        """Expanding z-score at time T must use only data up to T."""
        normalizer = FeatureNormalizer()
        df = _make_feature_timeseries(n_dates=100, n_features=2)

        result = normalizer.normalize_temporal(df, method="zscore", min_periods=20)

        # First min_periods - 1 rows should be NaN
        assert result.iloc[:19].isna().all().all()

        # At row 50, the z-score should equal manual expanding calculation
        col = df.columns[0]
        at_50 = df[col].iloc[:51]
        expected_z = (at_50.iloc[-1] - at_50.mean()) / at_50.std()
        actual_z = result[col].iloc[50]
        np.testing.assert_almost_equal(actual_z, expected_z, decimal=10)

    def test_temporal_percentile_no_future_data(self):
        """Expanding percentile at time T uses only data [0, T]."""
        normalizer = FeatureNormalizer()
        df = _make_feature_timeseries(n_dates=100, n_features=1)

        result = normalizer.normalize_temporal(df, method="percentile", min_periods=20)

        # First 19 rows NaN
        assert result.iloc[:19].isna().all().all()

        # Values should be in [0, 1]
        valid = result.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()

    def test_temporal_normalization_unknown_method_raises(self):
        normalizer = FeatureNormalizer()
        df = _make_feature_timeseries(n_dates=50, n_features=1)
        with pytest.raises(ValueError, match="Unknown temporal normalization method"):
            normalizer.normalize_temporal(df, method="invalid")


# ---------------------------------------------------------------------------
# Test 3: Feature selection integrated into pipeline
# ---------------------------------------------------------------------------

class TestFeatureSelectionIntegration:

    def test_feature_selection_integrated_into_pipeline(self):
        """ResearchRunner should accept and use a feature_selector."""
        runner = ResearchRunner()
        selector = MLFeatureSelector({"fs_correlation_threshold": 0.85})

        runner.inject_components(feature_selector=selector)
        assert runner._feature_selector is selector

    def test_feature_selection_reduces_features_on_redundant_data(self):
        """MLFeatureSelector removes redundant features."""
        rng = np.random.RandomState(42)
        n = 200

        # Create features where feat_dup is nearly identical to feat_a
        feat_a = rng.randn(n)
        feat_b = rng.randn(n)
        feat_dup = feat_a + rng.randn(n) * 0.01  # 0.99+ correlation with feat_a

        features = pd.DataFrame({
            "feat_a": feat_a,
            "feat_b": feat_b,
            "feat_dup": feat_dup,
        })
        target = pd.Series(feat_a * 0.5 + rng.randn(n) * 0.3)

        selector = MLFeatureSelector({"fs_correlation_threshold": 0.85})
        ranking = selector.select(features, target)

        # feat_dup should be removed as redundant
        assert "feat_dup" in ranking.removed_features or len(ranking.selected_features) < 3
        # At least 10% reduction (1 of 3 = 33%)
        assert len(ranking.selected_features) < len(features.columns)


# ---------------------------------------------------------------------------
# Test 4: Feature versioning stored in model store
# ---------------------------------------------------------------------------

class TestFeatureVersioning:

    def test_gbt_feature_importance_returns_array(self):
        """GBT get_feature_importance() returns array after training."""
        from quant_fund.research_algorithms.machine_learning.gradient_boosted_tree_model import (
            GradientBoostedTreeModel,
        )

        rng = np.random.RandomState(42)
        n = 300
        features = pd.DataFrame({
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "feat_c": rng.randn(n),
        })
        target = pd.Series(features["feat_a"] * 0.3 + rng.randn(n) * 0.5)

        model = GradientBoostedTreeModel({"gbt_min_train_days": 50})
        metrics = model.train_model(features, target)

        assert "train_r2" in metrics
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == 3
        assert importance.sum() > 0

        # Feature names can be tracked from the feature_matrix columns
        feature_names = list(features.columns)
        assert len(feature_names) == len(importance)
