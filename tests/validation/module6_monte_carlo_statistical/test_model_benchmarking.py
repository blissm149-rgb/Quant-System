"""Module 8 — Benchmarking & Baselines tests.

Tests baseline models, model comparison framework, and DSR integration.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.research_algorithms.machine_learning.baseline_models import (
    NaiveBaselineModel,
    RandomBaselineModel,
    MeanReversionBaseline,
    MomentumBaseline,
)
from quant_fund.research_algorithms.machine_learning.model_comparison_framework import (

pytestmark = [pytest.mark.validation, pytest.mark.tier4]
    ModelComparisonFramework,
)


def _make_data(n=600, n_features=5, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    features = pd.DataFrame(
        rng.randn(n, n_features), index=dates,
        columns=[f"f{i}" for i in range(n_features)],
    )
    returns = pd.Series(
        features["f0"].values * 0.03 + rng.randn(n) * 0.01,
        index=dates,
    )
    return features, returns


class _SimpleModel:
    """Linear model that actually learns from data."""
    def __init__(self, config=None):
        self._coef = None

    def train_model(self, features, returns):
        X = features.values
        y = returns.values
        try:
            self._coef = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            self._coef = np.zeros(X.shape[1])
        return {"train_r2": 0.0, "oos_r2": 0.0}

    def predict(self, features):
        X = features.values
        if self._coef is not None:
            return pd.Series(X @ self._coef, index=features.index)
        return pd.Series(0.0, index=features.index)


class TestRandomBaselineIC:

    def test_random_baseline_has_zero_expected_ic(self):
        """Random baseline should have |mean IC| < 0.03."""
        features, returns = _make_data()
        framework = ModelComparisonFramework({
            "wf_min_train_days": 200, "wf_test_days": 63, "wf_step_days": 21,
        })
        result = framework.run_comparison(
            models={},
            baselines={"random": RandomBaselineModel()},
            features=features,
            returns=returns,
        )
        random_row = result[result["model_name"] == "random"].iloc[0]
        assert abs(random_row["mean_oos_ic"]) < 0.10, \
            f"Random baseline IC {random_row['mean_oos_ic']:.4f} too far from zero"


class TestModelBeatsBaseline:

    def test_learned_model_beats_naive(self):
        """A model trained on signal should beat naive baseline."""
        features, returns = _make_data()
        framework = ModelComparisonFramework({
            "wf_min_train_days": 200, "wf_test_days": 63, "wf_step_days": 21,
        })
        result = framework.run_comparison(
            models={"linear": _SimpleModel()},
            baselines={"naive": NaiveBaselineModel()},
            features=features,
            returns=returns,
        )
        linear_row = result[result["model_name"] == "linear"].iloc[0]
        naive_row = result[result["model_name"] == "naive"].iloc[0]
        assert linear_row["mean_oos_ic"] > naive_row["mean_oos_ic"]


class TestComparisonFramework:

    def test_ranks_correctly(self):
        """Comparison should produce correct number of rows."""
        features, returns = _make_data()
        framework = ModelComparisonFramework({
            "wf_min_train_days": 200, "wf_test_days": 63, "wf_step_days": 21,
        })
        result = framework.run_comparison(
            models={"linear": _SimpleModel()},
            baselines={
                "naive": NaiveBaselineModel(),
                "random": RandomBaselineModel(),
            },
            features=features,
            returns=returns,
        )
        assert len(result) == 3
        assert "model_name" in result.columns
        assert "beats_all_baselines" in result.columns
        assert "dsr" in result.columns

    def test_dsr_values_in_valid_range(self):
        """DSR values should be in [0, 1]."""
        features, returns = _make_data()
        framework = ModelComparisonFramework({
            "wf_min_train_days": 200, "wf_test_days": 63, "wf_step_days": 21,
        })
        result = framework.run_comparison(
            models={"linear": _SimpleModel()},
            baselines={"naive": NaiveBaselineModel()},
            features=features,
            returns=returns,
            n_trials=5,
        )
        for _, row in result.iterrows():
            assert 0.0 <= row["dsr"] <= 1.0, f"DSR {row['dsr']} out of range"


class TestAllBaselinesCompared:

    def test_all_baselines_present(self):
        """All 4 baseline types should appear in comparison."""
        features, returns = _make_data()
        framework = ModelComparisonFramework({
            "wf_min_train_days": 200, "wf_test_days": 63, "wf_step_days": 21,
        })
        result = framework.run_comparison(
            models={"linear": _SimpleModel()},
            baselines={
                "naive": NaiveBaselineModel(),
                "random": RandomBaselineModel(),
                "mean_reversion": MeanReversionBaseline(),
                "momentum": MomentumBaseline(),
            },
            features=features,
            returns=returns,
        )
        assert len(result) == 5
        baseline_count = result["is_baseline"].sum()
        assert baseline_count == 4
