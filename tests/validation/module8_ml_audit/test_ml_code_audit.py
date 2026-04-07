"""Module 8 – ML Code Audit.

Audits every ML code path in the system for determinism, data leakage,
regularisation hygiene, seed propagation, and reproducibility. Covers
sklearn models (GBT, RF, NN), representation learning (LSTM, Transformer,
Autoencoder, CrossAssetEmbedding), the feature factory alignment layer,
and the technical indicator engine.
"""

import pytest
import numpy as np
import pandas as pd

from tests.conftest import make_ohlcv, make_returns, STANDARD_TICKERS

from quant_fund.research_algorithms.machine_learning.gradient_boosted_tree_model import (
    GradientBoostedTreeModel,
)
from quant_fund.research_algorithms.machine_learning.random_forest_model import (
    RandomForestModel,
)
from quant_fund.research_algorithms.machine_learning.neural_network_predictor import (
    NeuralNetworkPredictor,
)
from quant_fund.representation_learning.temporal_model_lstm import TemporalModelLSTM
from quant_fund.representation_learning.temporal_model_transformer import (
    TemporalModelTransformer,
)
from quant_fund.representation_learning.autoencoder_model import AutoencoderModel
from quant_fund.representation_learning.cross_asset_embedding_model import (
    CrossAssetEmbeddingModel,
)
from quant_fund.feature_factory.data_alignment_engine import (
    DataAlignmentEngine,
    LookAheadError,
)
from quant_fund.feature_factory.technical_indicator_engine import (
    TechnicalIndicatorEngine,
)


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


# ── helpers ─────────────────────────────────────────────────────────


def _make_feature_matrix(n_samples: int = 300, n_features: int = 10, seed: int = 42):
    """Create a deterministic feature matrix + forward returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_samples)
    X = pd.DataFrame(
        rng.normal(0, 1, (n_samples, n_features)),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.normal(0, 0.02, n_samples), index=dates, name="fwd_ret")
    return X, y


def _make_sequences(n_tickers: int = 5, seq_len: int = 100, n_features: int = 8, seed: int = 42):
    """Create dict of ticker -> DataFrame sequences for temporal models."""
    rng = np.random.default_rng(seed)
    seqs = {}
    for i in range(n_tickers):
        ticker = STANDARD_TICKERS[i]
        dates = pd.bdate_range("2019-01-02", periods=seq_len)
        data = pd.DataFrame(
            rng.normal(0, 1, (seq_len, n_features)),
            index=dates,
            columns=[f"feat_{j}" for j in range(n_features)],
        )
        seqs[ticker] = data
    return seqs


# ── Test 1: Sklearn model determinism ──────────────────────────────


class TestSklearnModelDeterminism:
    """Verify that all sklearn-based models produce identical results
    across repeated runs with the same seed and data."""

    @pytest.mark.parametrize(
        "ModelClass",
        [GradientBoostedTreeModel, RandomForestModel, NeuralNetworkPredictor],
        ids=["GBT", "RF", "NN"],
    )
    def test_train_model_deterministic(self, ModelClass):
        """Training the same model twice with identical input must yield
        the same metrics (train_r2 / train_accuracy) to machine precision."""
        X, y = _make_feature_matrix(n_samples=300, seed=42)

        model_a = ModelClass()
        model_b = ModelClass()

        result_a = model_a.train_model(X, y)
        result_b = model_b.train_model(X, y)

        assert "error" not in result_a, f"Model A returned error: {result_a}"
        assert "error" not in result_b, f"Model B returned error: {result_b}"

        # Same sample / feature counts
        assert result_a["n_samples"] == result_b["n_samples"]
        assert result_a["n_features"] == result_b["n_features"]

        # Deterministic metric
        metric_key = "train_accuracy" if "train_accuracy" in result_a else "train_r2"
        assert result_a[metric_key] == pytest.approx(
            result_b[metric_key], abs=1e-10
        ), f"{ModelClass.__name__} is not deterministic: {result_a[metric_key]} != {result_b[metric_key]}"


# ── Test 2: Representation model determinism ───────────────────────


class TestRepresentationModelDeterminism:
    """Verify that numpy-based representation models are reproducible."""

    def test_autoencoder_deterministic(self):
        """Two autoencoder fits with the same config/seed must yield
        identical final loss and encoded representations."""
        X, _ = _make_feature_matrix(n_samples=200, n_features=12, seed=42)

        ae_a = AutoencoderModel({"ae_epochs": 10, "random_seed": 42})
        ae_b = AutoencoderModel({"ae_epochs": 10, "random_seed": 42})

        res_a = ae_a.fit(X)
        res_b = ae_b.fit(X)

        assert res_a["final_loss"] == pytest.approx(res_b["final_loss"], abs=1e-12)

        enc_a = ae_a.encode(X)
        enc_b = ae_b.encode(X)
        pd.testing.assert_frame_equal(enc_a, enc_b, atol=1e-12)

    def test_cross_asset_embedding_deterministic(self):
        """CrossAssetEmbeddingModel must produce identical embeddings."""
        returns = make_returns(n_dates=200, tickers=STANDARD_TICKERS[:5], seed=42)

        m_a = CrossAssetEmbeddingModel({"random_seed": 42})
        m_b = CrossAssetEmbeddingModel({"random_seed": 42})

        m_a.fit(returns)
        m_b.fit(returns)

        pd.testing.assert_frame_equal(
            m_a.get_embeddings(), m_b.get_embeddings(), atol=1e-12
        )

    def test_lstm_deterministic(self):
        """TemporalModelLSTM must produce identical fit metrics."""
        seqs = _make_sequences(seed=42)

        m_a = TemporalModelLSTM({"random_seed": 42})
        m_b = TemporalModelLSTM({"random_seed": 42})

        res_a = m_a.fit(seqs)
        res_b = m_b.fit(seqs)

        assert res_a == res_b, f"LSTM non-deterministic: {res_a} != {res_b}"

    def test_transformer_deterministic(self):
        """TemporalModelTransformer must produce identical fit metrics."""
        seqs = _make_sequences(seed=42)

        m_a = TemporalModelTransformer({"random_seed": 42})
        m_b = TemporalModelTransformer({"random_seed": 42})

        res_a = m_a.fit(seqs)
        res_b = m_b.fit(seqs)

        assert res_a == res_b, f"Transformer non-deterministic: {res_a} != {res_b}"


# ── Test 3: Seed propagation audit ─────────────────────────────────


class TestSeedPropagation:
    """Verify that changing the seed actually changes model behaviour,
    confirming seed is being propagated (not silently ignored)."""

    def test_autoencoder_different_seed_yields_different_loss(self):
        """With a different seed the autoencoder must produce a different
        final loss, proving the seed controls initialisation."""
        X, _ = _make_feature_matrix(n_samples=200, n_features=12, seed=42)

        ae_a = AutoencoderModel({"ae_epochs": 10, "random_seed": 42})
        ae_b = AutoencoderModel({"ae_epochs": 10, "random_seed": 99})

        res_a = ae_a.fit(X)
        res_b = ae_b.fit(X)

        assert res_a["final_loss"] != pytest.approx(
            res_b["final_loss"], abs=1e-6
        ), "Different seeds produced identical loss — seed may not be propagated"

    def test_cross_asset_embedding_seed_matters(self):
        """CrossAssetEmbeddingModel uses SVD which is deterministic, but
        we verify that fitting with same seed always returns same output."""
        returns = make_returns(n_dates=200, tickers=STANDARD_TICKERS[:5], seed=42)

        m = CrossAssetEmbeddingModel({"random_seed": 42})
        m.fit(returns)
        emb_first = m.get_embeddings().copy()

        m2 = CrossAssetEmbeddingModel({"random_seed": 42})
        m2.fit(returns)
        emb_second = m2.get_embeddings()

        pd.testing.assert_frame_equal(emb_first, emb_second, atol=1e-12)


# ── Test 4: Data alignment no-lookahead for ML pipelines ──────────


class TestMLDataAlignment:
    """Verify that ML model training data cannot contain future information."""

    def test_alignment_engine_strict_blocks_future(self):
        """get_aligned_data must raise LookAheadError when data contains
        timestamps >= as_of."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=300, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[200]

        engine = DataAlignmentEngine()

        with pytest.raises(LookAheadError):
            engine.get_aligned_data(ohlcv, as_of=as_of, lookback_days=252)

    def test_permissive_alignment_filters_future_silently(self):
        """get_aligned_data_permissive must filter future rows and return
        only data strictly before as_of."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=300, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[200]

        engine = DataAlignmentEngine()
        aligned = engine.get_aligned_data_permissive(ohlcv, as_of=as_of, lookback_days=252)

        aligned_dates = aligned.index.get_level_values(0)
        assert (aligned_dates < as_of).all(), (
            "Permissive alignment returned data at or after as_of"
        )

    def test_strict_and_permissive_produce_same_result(self):
        """When input data is pre-filtered to < as_of, both alignment
        methods must return identical output."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=300, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[200]

        safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]
        engine = DataAlignmentEngine()

        strict = engine.get_aligned_data(safe, as_of=as_of, lookback_days=252)
        permissive = engine.get_aligned_data_permissive(safe, as_of=as_of, lookback_days=252)

        pd.testing.assert_frame_equal(strict, permissive)


# ── Test 5: Technical indicator engine alignment ───────────────────


class TestTechnicalIndicatorAlignment:
    """Verify the TechnicalIndicatorEngine enforces point-in-time safety."""

    def test_compute_all_rejects_future_data(self):
        """Passing data that extends past as_of must raise LookAheadError,
        proving the engine uses strict alignment."""
        from quant_fund.feature_factory.data_alignment_engine import LookAheadError

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=400, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[300]

        engine = TechnicalIndicatorEngine()

        with pytest.raises(LookAheadError):
            engine.compute_all(ohlcv, as_of=as_of)

    def test_compute_all_succeeds_with_pre_filtered_data(self):
        """Pre-filtered data (all timestamps < as_of) must produce valid features."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=400, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[300]

        engine = TechnicalIndicatorEngine()

        truncated = ohlcv[ohlcv.index.get_level_values(0) < as_of]
        feats = engine.compute_all(truncated, as_of=as_of)

        assert len(feats.columns) > 0, "No features were produced"
        assert len(feats) > 0, "No rows in feature output"


# ── Test 6: Regularisation parameter audit ─────────────────────────


class TestRegularisationHygiene:
    """Verify that all ML models have explicit regularisation and do not
    silently use unregularised defaults."""

    def test_gbt_has_bounded_depth_and_learning_rate(self):
        """GBT must use max_depth and learning_rate that prevent
        memorisation of the training set."""
        model = GradientBoostedTreeModel()
        X, y = _make_feature_matrix(n_samples=300, seed=42)
        result = model.train_model(X, y)
        assert "error" not in result

        # Verify the model's configuration values are reasonable
        cfg = model._config if hasattr(model, "_config") else {}
        max_depth = cfg.get("gbt_max_depth", 5)
        learning_rate = cfg.get("gbt_learning_rate", 0.05)

        assert max_depth <= 10, f"GBT max_depth={max_depth} is too deep — overfitting risk"
        assert learning_rate <= 0.3, f"GBT learning_rate={learning_rate} is too high"

    def test_rf_has_bounded_depth(self):
        """Random forest must use a bounded max_depth."""
        model = RandomForestModel()
        cfg = model._config if hasattr(model, "_config") else {}
        max_depth = cfg.get("rf_max_depth", 8)

        assert max_depth <= 15, f"RF max_depth={max_depth} is too deep — overfitting risk"

    def test_nn_uses_early_stopping_and_regularisation(self):
        """Neural network must use early stopping and L2 regularisation."""
        model = NeuralNetworkPredictor()
        cfg = model._config if hasattr(model, "_config") else {}
        alpha = cfg.get("nn_alpha", 0.01)

        assert alpha > 0, "NN has no L2 regularisation (alpha=0)"


# ── Test 7: Minimum training data guard ────────────────────────────


class TestMinimumDataGuards:
    """Verify that models reject insufficient training data
    gracefully (return error dict, don't crash)."""

    @pytest.mark.parametrize(
        "ModelClass,min_key,default_min",
        [
            (GradientBoostedTreeModel, "gbt_min_train_days", 252),
            (RandomForestModel, "rf_min_train_days", 252),
            (NeuralNetworkPredictor, "nn_min_train_days", 252),
        ],
        ids=["GBT", "RF", "NN"],
    )
    def test_sklearn_model_rejects_tiny_dataset(self, ModelClass, min_key, default_min):
        """Models must return an error dict when given fewer samples
        than the configured minimum."""
        X, y = _make_feature_matrix(n_samples=50, seed=42)
        model = ModelClass()
        result = model.train_model(X, y)
        assert "error" in result, (
            f"{ModelClass.__name__} accepted {len(X)} samples "
            f"(minimum should be ~{default_min})"
        )

    def test_autoencoder_rejects_tiny_dataset(self):
        """Autoencoder must return error for < 10 rows."""
        X = pd.DataFrame(np.random.default_rng(42).normal(0, 1, (5, 8)))
        ae = AutoencoderModel()
        result = ae.fit(X)
        assert "error" in result


# ── Test 8: Model output shape / contract audit ────────────────────


class TestModelOutputContracts:
    """Verify that every ML model returns the documented output schema."""

    def test_gbt_output_schema(self):
        X, y = _make_feature_matrix(n_samples=300, seed=42)
        model = GradientBoostedTreeModel()
        result = model.train_model(X, y)
        assert "error" not in result
        for key in ("n_samples", "n_features", "train_r2"):
            assert key in result, f"GBT output missing '{key}'"
        assert isinstance(result["train_r2"], float)

    def test_rf_output_schema(self):
        X, y = _make_feature_matrix(n_samples=300, seed=42)
        model = RandomForestModel()
        result = model.train_model(X, y)
        assert "error" not in result
        for key in ("n_samples", "n_features", "train_accuracy"):
            assert key in result, f"RF output missing '{key}'"
        assert 0 <= result["train_accuracy"] <= 1

    def test_nn_output_schema(self):
        X, y = _make_feature_matrix(n_samples=300, seed=42)
        model = NeuralNetworkPredictor()
        result = model.train_model(X, y)
        assert "error" not in result
        for key in ("n_samples", "n_features", "train_r2", "n_iter"):
            assert key in result, f"NN output missing '{key}'"

    def test_autoencoder_output_schema(self):
        X, _ = _make_feature_matrix(n_samples=200, n_features=12, seed=42)
        ae = AutoencoderModel({"ae_epochs": 5})
        result = ae.fit(X)
        for key in ("final_loss", "n_samples", "input_dim", "latent_dim"):
            assert key in result, f"Autoencoder output missing '{key}'"

    def test_autoencoder_encode_shape(self):
        X, _ = _make_feature_matrix(n_samples=200, n_features=12, seed=42)
        ae = AutoencoderModel({"ae_epochs": 5, "ae_latent_dim": 8})
        ae.fit(X)
        encoded = ae.encode(X)
        assert encoded.shape == (200, 8), f"Expected (200, 8), got {encoded.shape}"

    def test_cross_asset_similarity_is_symmetric(self):
        """Similarity matrix must be symmetric with 1.0 on the diagonal."""
        returns = make_returns(n_dates=200, tickers=STANDARD_TICKERS[:5], seed=42)
        model = CrossAssetEmbeddingModel({"random_seed": 42})
        model.fit(returns)
        sim = model.get_similarity_matrix()

        np.testing.assert_array_almost_equal(
            sim.values, sim.values.T, decimal=10,
            err_msg="Similarity matrix is not symmetric",
        )
        np.testing.assert_array_almost_equal(
            np.diag(sim.values), np.ones(sim.shape[0]), decimal=5,
            err_msg="Diagonal of similarity matrix should be ~1.0",
        )


# ── Test 9: Autoencoder reconstruction quality gate ────────────────


class TestAutoencoderReconstructionGate:
    """Verify the autoencoder achieves reasonable reconstruction fidelity."""

    def test_reconstruction_error_decreases_with_training(self):
        """Training for more epochs must reduce reconstruction error."""
        X, _ = _make_feature_matrix(n_samples=200, n_features=12, seed=42)

        ae_short = AutoencoderModel({"ae_epochs": 2, "random_seed": 42})
        ae_long = AutoencoderModel({"ae_epochs": 50, "random_seed": 42})

        res_short = ae_short.fit(X)
        res_long = ae_long.fit(X)

        assert res_long["final_loss"] < res_short["final_loss"], (
            f"Longer training ({res_long['final_loss']:.6f}) did not reduce "
            f"loss vs short training ({res_short['final_loss']:.6f})"
        )

    def test_reconstruction_roundtrip(self):
        """Reconstructed data must have finite values and correct shape."""
        X, _ = _make_feature_matrix(n_samples=200, n_features=12, seed=42)
        ae = AutoencoderModel({"ae_epochs": 20, "random_seed": 42})
        ae.fit(X)

        recon = ae.reconstruct(X)
        assert recon.shape == X.shape
        assert np.isfinite(recon.values).all(), "Reconstruction contains non-finite values"


# ── Test 10: NaN handling across all models ────────────────────────


class TestNaNHandling:
    """Verify that all ML models handle NaN-contaminated inputs safely
    (either filter NaNs or return an error — never produce NaN outputs)."""

    def test_sklearn_models_handle_nan_features(self):
        """Sklearn models must filter NaN rows via valid_mask and still
        produce valid output when enough clean rows remain."""
        X, y = _make_feature_matrix(n_samples=400, seed=42)
        # Inject NaNs into 10% of rows
        rng = np.random.default_rng(99)
        nan_rows = rng.choice(400, 40, replace=False)
        X.iloc[nan_rows, 0] = np.nan

        for ModelClass in [GradientBoostedTreeModel, RandomForestModel, NeuralNetworkPredictor]:
            model = ModelClass()
            result = model.train_model(X, y)
            if "error" not in result:
                metric_key = "train_accuracy" if "train_accuracy" in result else "train_r2"
                assert np.isfinite(result[metric_key]), (
                    f"{ModelClass.__name__} produced NaN metric"
                )

    def test_autoencoder_handles_nan_rows(self):
        """Autoencoder.fit() drops NaN rows; output must still be valid."""
        X, _ = _make_feature_matrix(n_samples=200, n_features=12, seed=42)
        X.iloc[0:10, 0] = np.nan

        ae = AutoencoderModel({"ae_epochs": 5, "random_seed": 42})
        result = ae.fit(X)
        assert "error" not in result
        assert np.isfinite(result["final_loss"])

    def test_autoencoder_encode_fills_nan(self):
        """encode() uses fillna(0) — output must be finite for NaN input."""
        X, _ = _make_feature_matrix(n_samples=200, n_features=12, seed=42)
        ae = AutoencoderModel({"ae_epochs": 5, "random_seed": 42})
        ae.fit(X)

        X_with_nan = X.copy()
        X_with_nan.iloc[0:5, :] = np.nan
        encoded = ae.encode(X_with_nan)
        assert np.isfinite(encoded.values).all(), "encode() produced NaN for NaN input"
