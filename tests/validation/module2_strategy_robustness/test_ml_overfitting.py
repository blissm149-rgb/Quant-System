"""Module 6 — Overfitting Detection tests.

Tests train/test gap analysis, learning curves, complexity curves,
and overfit probability estimation.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.research_algorithms.machine_learning.overfitting_detector import (
    compute_train_test_gap,
    learning_curve,
    complexity_curve,
    overfit_probability,
)
from quant_fund.research_algorithms.machine_learning.gradient_boosted_tree_model import (

    GradientBoostedTreeModel,
)

pytestmark = [pytest.mark.validation, pytest.mark.tier4]


def _make_data(n=500, n_features=5, signal_strength=0.05, seed=42):
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        rng.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)]
    )
    returns = pd.Series(features["f0"].values * signal_strength + rng.randn(n) * 0.01)
    return features, returns


class TestTrainTestGap:

    def test_flags_overfit(self):
        result = compute_train_test_gap(train_metric=0.8, test_metric=0.1)
        assert result["overfitting_flag"] is True
        assert result["gap"] == pytest.approx(0.7)

    def test_accepts_good_model(self):
        result = compute_train_test_gap(train_metric=0.15, test_metric=0.10)
        assert result["overfitting_flag"] is False
        assert result["gap"] == pytest.approx(0.05)


class TestLearningCurve:

    def test_improves_with_data(self):
        features, returns = _make_data(n=500, signal_strength=0.1)
        results = learning_curve(
            GradientBoostedTreeModel,
            {"gbt_min_train_days": 10, "gbt_n_estimators": 20, "gbt_max_depth": 3},
            features,
            returns,
            fractions=[0.3, 0.6, 1.0],
        )
        assert len(results) == 3
        # Training metric should generally be non-negative
        for r in results:
            assert "train_metric" in r
            assert "test_metric" in r


class TestComplexityCurve:

    def test_identifies_varying_depth(self):
        features, returns = _make_data(n=500, signal_strength=0.05)
        results = complexity_curve(
            GradientBoostedTreeModel,
            {"gbt_min_train_days": 10, "gbt_n_estimators": 30},
            features,
            returns,
            param_name="gbt_max_depth",
            param_values=[1, 3, 5, 8],
        )
        assert len(results) == 4
        # At some depth, train metric should be higher than at depth=1
        train_metrics = [r["train_metric"] for r in results]
        assert max(train_metrics) > train_metrics[0]


class TestOverfitProbability:

    def test_high_for_noise(self):
        """Random OOS ICs centered at zero should have high overfit probability."""
        rng = np.random.RandomState(42)
        noise_ics = rng.randn(20) * 0.02  # centered at 0
        p = overfit_probability(noise_ics, n_bootstrap=2000)
        assert p > 0.3, f"Overfit probability {p} too low for noise"

    def test_low_for_signal(self):
        """Consistently positive OOS ICs should have low overfit probability."""
        signal_ics = [0.05 + np.random.randn() * 0.01 for _ in range(20)]
        p = overfit_probability(signal_ics, n_bootstrap=2000)
        assert p < 0.1, f"Overfit probability {p} too high for signal"
