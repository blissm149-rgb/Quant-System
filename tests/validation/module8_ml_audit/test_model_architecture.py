"""Module 4 — Model Architecture Upgrade tests.

Tests regularized linear models, GBT/RF with regularization and OOS metrics,
LSTM/Transformer training loops, and model ensembling.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.research_algorithms.machine_learning.regularized_linear_model import (
    RegularizedLinearModel,
)
from quant_fund.research_algorithms.machine_learning.gradient_boosted_tree_model import (
    GradientBoostedTreeModel,
)
from quant_fund.research_algorithms.machine_learning.random_forest_model import (
    RandomForestModel,
)
from quant_fund.research_algorithms.machine_learning.model_ensemble import (
    ModelEnsemble,
)
from quant_fund.representation_learning.temporal_model_lstm import TemporalModelLSTM
from quant_fund.representation_learning.temporal_model_transformer import (

pytestmark = [pytest.mark.validation, pytest.mark.tier4]
    TemporalModelTransformer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_training_data(n=500, n_features=5, seed=42):
    rng = np.random.RandomState(seed)
    features = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Linear signal + noise
    returns = pd.Series(
        features["feat_0"].values * 0.05 + rng.randn(n) * 0.01
    )
    return features, returns


def _make_sequences(n_tickers=5, seq_len=80, n_features=4, seed=42):
    rng = np.random.RandomState(seed)
    sequences = {}
    for i in range(n_tickers):
        sequences[f"T{i}"] = pd.DataFrame(
            rng.randn(seq_len, n_features),
            columns=[f"f{j}" for j in range(n_features)],
        )
    return sequences


# ---------------------------------------------------------------------------
# Test 1: Regularized linear model trains and predicts
# ---------------------------------------------------------------------------

class TestRegularizedLinearModel:

    def test_ridge_trains_and_returns_oos_metrics(self):
        features, returns = _make_training_data()
        model = RegularizedLinearModel({
            "linear_model_type": "ridge",
            "linear_min_train_days": 50,
        })
        metrics = model.train_model(features, returns)
        assert "train_r2" in metrics
        assert "oos_r2" in metrics
        assert metrics["n_val_samples"] > 0

    def test_lasso_trains(self):
        features, returns = _make_training_data()
        model = RegularizedLinearModel({
            "linear_model_type": "lasso",
            "linear_min_train_days": 50,
        })
        metrics = model.train_model(features, returns)
        assert "oos_r2" in metrics

    def test_elasticnet_trains(self):
        features, returns = _make_training_data()
        model = RegularizedLinearModel({
            "linear_model_type": "elasticnet",
            "linear_min_train_days": 50,
        })
        metrics = model.train_model(features, returns)
        assert metrics["model_type"] == "elasticnet"


# ---------------------------------------------------------------------------
# Test 2: GBT with regularization reduces overfitting
# ---------------------------------------------------------------------------

class TestGBTRegularization:

    def test_gbt_returns_oos_metrics(self):
        features, returns = _make_training_data(n=500)
        model = GradientBoostedTreeModel({"gbt_min_train_days": 50})
        metrics = model.train_model(features, returns)
        assert "oos_r2" in metrics
        assert "oos_ic" in metrics
        assert metrics["n_val_samples"] > 0

    def test_gbt_reads_regularization_params(self):
        model = GradientBoostedTreeModel({
            "gbt_min_samples_leaf": 25,
            "gbt_max_leaf_nodes": 40,
            "gbt_subsample": 0.7,
            "gbt_max_features": 0.6,
        })
        assert model._min_samples_leaf == 25
        assert model._max_leaf_nodes == 40
        assert model._subsample == 0.7
        assert model._max_features == 0.6

    def test_gbt_on_noise_low_oos_r2(self):
        """GBT on pure noise should have low OOS R2."""
        rng = np.random.RandomState(99)
        n = 500
        features = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        returns = pd.Series(rng.randn(n) * 0.01)
        model = GradientBoostedTreeModel({"gbt_min_train_days": 50})
        metrics = model.train_model(features, returns)
        assert metrics["oos_r2"] < 0.05, f"OOS R2 too high on noise: {metrics['oos_r2']}"


# ---------------------------------------------------------------------------
# Test 3: RF with regularization
# ---------------------------------------------------------------------------

class TestRFRegularization:

    def test_rf_returns_oos_accuracy(self):
        features, returns = _make_training_data(n=500)
        model = RandomForestModel({"rf_min_train_days": 50})
        metrics = model.train_model(features, returns)
        assert "oos_accuracy" in metrics
        assert metrics["n_val_samples"] > 0

    def test_rf_reads_regularization_params(self):
        model = RandomForestModel({
            "rf_min_samples_leaf": 15,
            "rf_max_features": "sqrt",
        })
        assert model._min_samples_leaf == 15
        assert model._max_features == "sqrt"


# ---------------------------------------------------------------------------
# Test 4: LSTM weights change after training
# ---------------------------------------------------------------------------

class TestLSTMTraining:

    def test_lstm_weights_change_after_training(self):
        """LSTM fit() must modify weights via BPTT, not just initialize."""
        sequences = _make_sequences(n_tickers=3, seq_len=80, n_features=4)
        model = TemporalModelLSTM({
            "lstm_hidden_dim": 16,
            "lstm_sequence_length": 20,
            "lstm_output_dim": 8,
            "lstm_epochs": 10,
            "lstm_learning_rate": 0.01,
        })

        # First fit to get initial weights
        metrics = model.fit(sequences)
        assert model.is_fitted
        assert "initial_loss" in metrics
        assert "final_loss" in metrics

        # Loss should decrease
        assert metrics["final_loss"] < metrics["initial_loss"], \
            f"Loss did not decrease: {metrics['initial_loss']:.4f} -> {metrics['final_loss']:.4f}"

    def test_lstm_loss_reduction_significant(self):
        """LSTM final loss should be at least 30% lower than initial."""
        sequences = _make_sequences(n_tickers=5, seq_len=80, n_features=4)
        model = TemporalModelLSTM({
            "lstm_hidden_dim": 16,
            "lstm_sequence_length": 20,
            "lstm_output_dim": 4,
            "lstm_epochs": 50,
            "lstm_learning_rate": 0.005,
        })
        metrics = model.fit(sequences)
        reduction = 1 - metrics["final_loss"] / metrics["initial_loss"]
        assert reduction > 0.30, f"Loss reduction only {reduction:.1%}, need >30%"


# ---------------------------------------------------------------------------
# Test 5: Transformer weights change after training
# ---------------------------------------------------------------------------

class TestTransformerTraining:

    def test_transformer_weights_change_after_training(self):
        sequences = _make_sequences(n_tickers=3, seq_len=40, n_features=4)
        model = TemporalModelTransformer({
            "transformer_d_model": 16,
            "transformer_n_heads": 4,
            "transformer_sequence_length": 20,
            "transformer_output_dim": 4,
            "transformer_epochs": 10,
            "transformer_learning_rate": 0.01,
        })
        metrics = model.fit(sequences)
        assert model.is_fitted
        assert "initial_loss" in metrics
        assert "final_loss" in metrics
        assert metrics["final_loss"] < metrics["initial_loss"], \
            f"Loss did not decrease: {metrics['initial_loss']:.4f} -> {metrics['final_loss']:.4f}"


# ---------------------------------------------------------------------------
# Test 6: Model ensemble combines signals
# ---------------------------------------------------------------------------

class TestModelEnsemble:

    def test_ensemble_combines_predictions(self):
        """Ensemble prediction should be weighted average of members."""
        class MockModel:
            def __init__(self, val):
                self._val = val
            def compute(self, data, as_of):
                return pd.Series({"A": self._val, "B": self._val * 2})

        ensemble = ModelEnsemble()
        ensemble.add_model("m1", MockModel(1.0), weight=1.0)
        ensemble.add_model("m2", MockModel(2.0), weight=1.0)

        result = ensemble.predict(pd.DataFrame(), as_of=pd.Timestamp("2024-01-01"))
        # Equal weight: (1+2)/2 = 1.5 for A, (2+4)/2 = 3.0 for B
        np.testing.assert_almost_equal(result["A"], 1.5)
        np.testing.assert_almost_equal(result["B"], 3.0)

    def test_ensemble_diversity_warning(self):
        """Two identical predictions should trigger diversity warning."""
        preds = {
            "m1": pd.Series(np.arange(100, dtype=float)),
            "m2": pd.Series(np.arange(100, dtype=float)),
        }
        ensemble = ModelEnsemble({"ensemble_diversity_threshold": 0.7})
        result = ensemble.evaluate_diversity(preds)
        assert result["mean_correlation"] > 0.7
        assert result["diversity_warning"] is True

    def test_ensemble_diversity_ok_for_different_signals(self):
        rng = np.random.RandomState(42)
        preds = {
            "m1": pd.Series(rng.randn(100)),
            "m2": pd.Series(rng.randn(100)),
        }
        ensemble = ModelEnsemble({"ensemble_diversity_threshold": 0.7})
        result = ensemble.evaluate_diversity(preds)
        assert result["diversity_warning"] is False
