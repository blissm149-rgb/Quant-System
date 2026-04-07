"""Module 7 — Stress & Adversarial Testing.

Tests model robustness to noise injection, feature perturbation,
label noise, and adversarial features.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.research_algorithms.machine_learning.adversarial_feature_generator import (
    generate_noise_features,
    generate_lookahead_feature,
    generate_nonstationary_feature,
    perturb_features,
    add_label_noise,
)
from quant_fund.research_algorithms.machine_learning.gradient_boosted_tree_model import (
    GradientBoostedTreeModel,
)
from quant_fund.research_algorithms.machine_learning.model_ensemble import (

    ModelEnsemble,
)

pytestmark = [pytest.mark.validation, pytest.mark.tier4]


def _make_data(n=500, n_features=5, seed=42):
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        rng.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)]
    )
    returns = pd.Series(features["f0"].values * 0.05 + rng.randn(n) * 0.01)
    return features, returns


def _train_and_oos_ic(features, returns, config=None):
    """Train GBT and return OOS IC."""
    cfg = config or {"gbt_min_train_days": 50, "gbt_n_estimators": 50, "gbt_max_depth": 3}
    model = GradientBoostedTreeModel(cfg)
    metrics = model.train_model(features, returns)
    return metrics.get("oos_ic", 0.0)


class TestNoiseFeatureInjection:

    def test_random_features_do_not_improve_model(self):
        """Adding noise features should not improve OOS IC."""
        features, returns = _make_data()

        ic_original = _train_and_oos_ic(features, returns)

        noise = generate_noise_features(len(features), n_features=10)
        noise.index = features.index
        features_with_noise = pd.concat([features, noise], axis=1)

        ic_with_noise = _train_and_oos_ic(features_with_noise, returns)

        improvement = ic_with_noise - ic_original
        assert improvement < 0.02, f"Noise features improved IC by {improvement:.4f}"


class TestFeaturePerturbation:

    def test_model_robust_to_perturbation(self):
        """Model IC should not degrade more than 50% under small perturbation."""
        features, returns = _make_data()
        ic_original = _train_and_oos_ic(features, returns)

        perturbed = perturb_features(features, noise_fraction=0.1)
        ic_perturbed = _train_and_oos_ic(perturbed, returns)

        if abs(ic_original) > 0.01:
            degradation = 1 - ic_perturbed / ic_original
            assert degradation < 0.50, f"IC degradation {degradation:.1%} > 50%"


class TestLabelNoise:

    def test_model_robust_to_label_noise(self):
        """Model should still have some signal under moderate label noise."""
        features, returns = _make_data()
        ic_original = _train_and_oos_ic(features, returns)

        noisy_returns = add_label_noise(returns, noise_fraction=0.5)
        ic_noisy = _train_and_oos_ic(features, noisy_returns)

        if abs(ic_original) > 0.01:
            degradation = 1 - ic_noisy / ic_original
            assert degradation < 0.70, f"IC degradation {degradation:.1%} > 70%"


class TestNonstationaryFeature:

    def test_nonstationary_feature_generated(self):
        feat = generate_nonstationary_feature(100, break_point=0.5)
        assert len(feat) == 100
        # Second half should have different mean
        first_half = feat.iloc[:50]
        second_half = feat.iloc[50:]
        assert abs(second_half.mean() - first_half.mean()) > 0.5


class TestLookaheadFeature:

    def test_lookahead_feature_correlated_with_returns(self):
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.randn(100) * 0.01)
        lookahead = generate_lookahead_feature(returns, noise_level=0.001)
        corr = returns.corr(lookahead)
        assert corr > 0.9, "Lookahead feature should be highly correlated"


class TestEnsembleRobustness:

    def test_ensemble_handles_nan_from_failed_model(self):
        """Ensemble should degrade gracefully when one model fails."""
        class GoodModel:
            def compute(self, data, as_of):
                return pd.Series({"A": 1.0, "B": 2.0})

        class FailModel:
            def compute(self, data, as_of):
                return pd.Series({"A": np.nan, "B": np.nan})

        ensemble = ModelEnsemble()
        ensemble.add_model("good", GoodModel(), weight=1.0)
        ensemble.add_model("fail", FailModel(), weight=1.0)

        result = ensemble.predict(pd.DataFrame(), as_of=pd.Timestamp("2024-01-01"))
        # Should fall back to good model's predictions
        assert result["A"] == 1.0
        assert result["B"] == 2.0
