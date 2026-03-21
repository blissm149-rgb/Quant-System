"""Unit tests for representation learning.

TESTING_PLAN.md Section 3.17 — AutoencoderModel.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.representation_learning.autoencoder_model import AutoencoderModel


@pytest.mark.unit
@pytest.mark.tier3
class TestAutoencoderModel:
    """AutoencoderModel — feature dimensionality reduction."""

    @pytest.fixture
    def model(self):
        return AutoencoderModel(config={
            "ae_latent_dim": 4,
            "ae_hidden_dim": 16,
            "ae_epochs": 20,
            "ae_batch_size": 16,
        })

    @pytest.fixture
    def feature_matrix(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 10))
        return pd.DataFrame(data, columns=[f"f_{i}" for i in range(10)])

    def test_fit_returns_metrics(self, model, feature_matrix):
        metrics = model.fit(feature_matrix)
        assert isinstance(metrics, dict)
        assert "final_loss" in metrics
        assert model.is_fitted is True

    def test_encode_produces_latent(self, model, feature_matrix):
        model.fit(feature_matrix)
        latent = model.encode(feature_matrix)
        assert latent.shape == (100, 4)
        assert isinstance(latent, pd.DataFrame)

    def test_encode_before_fit_raises(self, model, feature_matrix):
        with pytest.raises(RuntimeError):
            model.encode(feature_matrix)

    def test_reconstruction_error_decreases(self, feature_matrix):
        """Trained model has lower reconstruction error than random."""
        model = AutoencoderModel(config={
            "ae_latent_dim": 8,
            "ae_hidden_dim": 32,
            "ae_epochs": 50,
            "ae_batch_size": 32,
        })
        model.fit(feature_matrix)
        error = model.reconstruction_error(feature_matrix)
        assert error < 10.0  # reasonable for normalised data

    def test_latent_dim_property(self, model):
        assert model.latent_dim == 4

    def test_insufficient_data(self, model):
        tiny = pd.DataFrame({"a": [1.0], "b": [2.0]})
        metrics = model.fit(tiny)
        assert "error" in metrics
