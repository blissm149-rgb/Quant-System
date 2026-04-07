"""Module 9 — Reproducibility & Experiment Tracking tests.

Tests experiment tracker, seed manager, model reproducibility,
and experiment comparison.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.infrastructure.experiment_tracker import ExperimentTracker
from quant_fund.infrastructure.seed_manager import SeedManager
from quant_fund.research_algorithms.machine_learning.gradient_boosted_tree_model import (

pytestmark = [pytest.mark.tier3]
    GradientBoostedTreeModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=300, seed=42):
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        rng.randn(n, 3), columns=["f0", "f1", "f2"]
    )
    returns = pd.Series(features["f0"].values * 0.05 + rng.randn(n) * 0.01)
    return features, returns


# ---------------------------------------------------------------------------
# Test 1: Experiment tracker records and retrieves
# ---------------------------------------------------------------------------

class TestExperimentTracker:

    def test_records_and_retrieves(self):
        tracker = ExperimentTracker()
        tracker.start_experiment("exp_001", description="Test run")
        tracker.log_params({"learning_rate": 0.01, "n_estimators": 100})
        tracker.log_metrics({"train_r2": 0.15, "oos_r2": 0.08})
        tracker.log_feature_set(["feat_a", "feat_b", "feat_c"])
        tracker.log_artifact("/path/to/model.joblib")
        record = tracker.end_experiment()

        assert record["id"] == "exp_001"
        assert record["status"] == "completed"
        assert record["params"]["learning_rate"] == 0.01
        assert record["metrics"]["oos_r2"] == 0.08
        assert len(record["feature_set"]) == 3
        assert len(record["artifacts"]) == 1

    def test_multiple_experiments(self):
        tracker = ExperimentTracker()
        tracker.start_experiment("exp_a")
        tracker.log_metrics({"oos_r2": 0.05})
        tracker.end_experiment()

        tracker.start_experiment("exp_b")
        tracker.log_metrics({"oos_r2": 0.10})
        tracker.end_experiment()

        assert len(tracker.all_experiments) == 2


# ---------------------------------------------------------------------------
# Test 2: Experiment comparison
# ---------------------------------------------------------------------------

class TestExperimentComparison:

    def test_compare_experiments(self):
        tracker = ExperimentTracker()

        tracker.start_experiment("run_1")
        tracker.log_metrics({"oos_r2": 0.05, "oos_ic": 0.03})
        tracker.log_feature_set(["f0", "f1"])
        tracker.end_experiment()

        tracker.start_experiment("run_2")
        tracker.log_metrics({"oos_r2": 0.10, "oos_ic": 0.06})
        tracker.log_feature_set(["f0", "f1", "f2"])
        tracker.end_experiment()

        comparison = tracker.compare_experiments(["run_1", "run_2"])
        assert len(comparison) == 2
        assert comparison[0]["oos_r2"] == 0.05
        assert comparison[1]["n_features"] == 3


# ---------------------------------------------------------------------------
# Test 3: Seed manager deterministic
# ---------------------------------------------------------------------------

class TestSeedManager:

    def test_deterministic(self):
        """Same base_seed produces same derived seeds."""
        sm1 = SeedManager(base_seed=42)
        sm2 = SeedManager(base_seed=42)

        seed1 = sm1.get_seed("gbt_model")
        seed2 = sm2.get_seed("gbt_model")
        assert seed1 == seed2

    def test_different_components_different_seeds(self):
        """Different component names produce different seeds."""
        sm = SeedManager(base_seed=42)
        seed_gbt = sm.get_seed("gbt_model")
        seed_rf = sm.get_seed("rf_model")
        seed_nn = sm.get_seed("nn_model")

        assert seed_gbt != seed_rf
        assert seed_gbt != seed_nn
        assert seed_rf != seed_nn

    def test_different_base_seeds(self):
        """Different base seeds produce different results."""
        sm1 = SeedManager(base_seed=42)
        sm2 = SeedManager(base_seed=99)

        assert sm1.get_seed("gbt") != sm2.get_seed("gbt")

    def test_get_all_seeds(self):
        sm = SeedManager(base_seed=42)
        sm.get_seed("a")
        sm.get_seed("b")
        all_seeds = sm.get_all_seeds()
        assert len(all_seeds) == 2
        assert "a" in all_seeds


# ---------------------------------------------------------------------------
# Test 4: Model reproducibility with seed manager
# ---------------------------------------------------------------------------

class TestModelReproducibility:

    def test_same_seed_same_metrics(self):
        """Identical seed produces identical training metrics."""
        features, returns = _make_data()

        model1 = GradientBoostedTreeModel({
            "gbt_min_train_days": 50,
            "gbt_n_estimators": 50,
            "random_seed": 42,
        })
        metrics1 = model1.train_model(features, returns)

        model2 = GradientBoostedTreeModel({
            "gbt_min_train_days": 50,
            "gbt_n_estimators": 50,
            "random_seed": 42,
        })
        metrics2 = model2.train_model(features, returns)

        assert metrics1["train_r2"] == metrics2["train_r2"]
        assert metrics1["oos_r2"] == metrics2["oos_r2"]

    def test_different_seeds_different_metrics(self):
        """Different seeds may produce different OOS metrics."""
        features, returns = _make_data()

        model1 = GradientBoostedTreeModel({
            "gbt_min_train_days": 50,
            "gbt_n_estimators": 50,
            "random_seed": 42,
        })
        metrics1 = model1.train_model(features, returns)

        model2 = GradientBoostedTreeModel({
            "gbt_min_train_days": 50,
            "gbt_n_estimators": 50,
            "random_seed": 99,
        })
        metrics2 = model2.train_model(features, returns)

        # With GBT + subsample, different seeds should produce different train R2
        # (subsample introduces randomness)
        assert metrics1["train_r2"] != metrics2["train_r2"] or \
               metrics1["oos_r2"] != metrics2["oos_r2"]


# ---------------------------------------------------------------------------
# Test 5: Tracker integration
# ---------------------------------------------------------------------------

class TestTrackerIntegration:

    def test_full_workflow(self):
        """End-to-end: train model, log to tracker, compare."""
        tracker = ExperimentTracker()
        features, returns = _make_data()

        for seed in [42, 99]:
            exp_id = f"gbt_seed_{seed}"
            tracker.start_experiment(exp_id, description=f"GBT with seed {seed}")
            tracker.log_params({"seed": seed, "n_estimators": 50})

            model = GradientBoostedTreeModel({
                "gbt_min_train_days": 50,
                "gbt_n_estimators": 50,
                "random_seed": seed,
            })
            metrics = model.train_model(features, returns)
            tracker.log_metrics(metrics)
            tracker.log_feature_set(list(features.columns))
            tracker.end_experiment()

        comparison = tracker.compare_experiments(["gbt_seed_42", "gbt_seed_99"])
        assert len(comparison) == 2
        # Both should have metrics logged
        for row in comparison:
            assert "train_r2" in row
            assert "oos_r2" in row
            assert row["n_features"] == 3
