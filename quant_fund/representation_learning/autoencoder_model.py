"""Bottleneck autoencoder for feature dimensionality reduction.

Trained on the full feature matrix. Produces latent representations
(embeddings) of configurable dimension. Embeddings are stored per date
and treated as features — all point-in-time rules apply.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AutoencoderModel:
    """Standard bottleneck autoencoder for feature compression.

    Uses a symmetric encoder-decoder architecture with configurable
    latent dimension. Implements a simple numpy-based autoencoder
    (single hidden layer) for portability; swap in PyTorch for production.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._latent_dim = cfg.get("ae_latent_dim", 16)
        self._hidden_dim = cfg.get("ae_hidden_dim", 64)
        self._learning_rate = cfg.get("ae_learning_rate", 0.001)
        self._epochs = cfg.get("ae_epochs", 100)
        self._batch_size = cfg.get("ae_batch_size", 64)
        self._seed = cfg.get("random_seed", 42)
        self._encoder_weights: Optional[dict] = None
        self._decoder_weights: Optional[dict] = None
        self._input_dim: Optional[int] = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def fit(self, feature_matrix: pd.DataFrame) -> dict:
        """Train the autoencoder on a feature matrix.

        Args:
            feature_matrix: DataFrame (samples × features). NaN rows are dropped.

        Returns:
            Training metrics dict.
        """
        X = feature_matrix.dropna().values.astype(np.float64)
        if len(X) < 10:
            logger.warning("Insufficient data for autoencoder: %d rows", len(X))
            return {"error": "insufficient data"}

        self._input_dim = X.shape[1]
        rng = np.random.default_rng(self._seed)

        # Xavier initialisation
        scale_enc = np.sqrt(2.0 / (self._input_dim + self._hidden_dim))
        scale_lat = np.sqrt(2.0 / (self._hidden_dim + self._latent_dim))

        W1 = rng.normal(0, scale_enc, (self._input_dim, self._hidden_dim))
        b1 = np.zeros(self._hidden_dim)
        W2 = rng.normal(0, scale_lat, (self._hidden_dim, self._latent_dim))
        b2 = np.zeros(self._latent_dim)
        W3 = rng.normal(0, scale_lat, (self._latent_dim, self._hidden_dim))
        b3 = np.zeros(self._hidden_dim)
        W4 = rng.normal(0, scale_enc, (self._hidden_dim, self._input_dim))
        b4 = np.zeros(self._input_dim)

        # Normalise input
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-8] = 1.0
        X_norm = (X - self._mean) / self._std

        losses = []
        for epoch in range(self._epochs):
            indices = rng.permutation(len(X_norm))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(X_norm), self._batch_size):
                batch = X_norm[indices[start : start + self._batch_size]]

                # Forward
                h1 = np.maximum(0, batch @ W1 + b1)  # ReLU
                z = h1 @ W2 + b2  # latent
                h3 = np.maximum(0, z @ W3 + b3)  # ReLU
                recon = h3 @ W4 + b4

                loss = np.mean((recon - batch) ** 2)
                epoch_loss += loss
                n_batches += 1

                # Backward (simplified gradient descent)
                d_recon = 2 * (recon - batch) / batch.shape[0]
                dW4 = h3.T @ d_recon
                db4 = d_recon.sum(axis=0)
                d_h3 = d_recon @ W4.T
                d_h3[h3 <= 0] = 0  # ReLU grad
                dW3 = z.T @ d_h3
                db3 = d_h3.sum(axis=0)
                d_z = d_h3 @ W3.T
                dW2 = h1.T @ d_z
                db2 = d_z.sum(axis=0)
                d_h1 = d_z @ W2.T
                d_h1[h1 <= 0] = 0
                dW1 = batch.T @ d_h1
                db1 = d_h1.sum(axis=0)

                lr = self._learning_rate
                W4 -= lr * dW4
                b4 -= lr * db4
                W3 -= lr * dW3
                b3 -= lr * db3
                W2 -= lr * dW2
                b2 -= lr * db2
                W1 -= lr * dW1
                b1 -= lr * db1

            losses.append(epoch_loss / max(n_batches, 1))

        self._encoder_weights = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        self._decoder_weights = {"W3": W3, "b3": b3, "W4": W4, "b4": b4}
        self._is_fitted = True

        return {
            "final_loss": losses[-1] if losses else 0.0,
            "n_samples": len(X),
            "input_dim": self._input_dim,
            "latent_dim": self._latent_dim,
        }

    def encode(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Encode features into latent space.

        Args:
            feature_matrix: DataFrame (samples × features).

        Returns:
            DataFrame of latent representations (samples × latent_dim).
        """
        if not self._is_fitted:
            raise RuntimeError("Autoencoder must be fitted before encoding")

        X = feature_matrix.fillna(0.0).values.astype(np.float64)
        X_norm = (X - self._mean) / self._std

        h1 = np.maximum(0, X_norm @ self._encoder_weights["W1"] + self._encoder_weights["b1"])
        z = h1 @ self._encoder_weights["W2"] + self._encoder_weights["b2"]

        cols = [f"ae_latent_{i}" for i in range(self._latent_dim)]
        return pd.DataFrame(z, index=feature_matrix.index, columns=cols)

    def reconstruct(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Encode and decode to measure reconstruction quality."""
        if not self._is_fitted:
            raise RuntimeError("Autoencoder must be fitted before reconstruction")

        X = feature_matrix.fillna(0.0).values.astype(np.float64)
        X_norm = (X - self._mean) / self._std

        w = self._encoder_weights
        d = self._decoder_weights
        h1 = np.maximum(0, X_norm @ w["W1"] + w["b1"])
        z = h1 @ w["W2"] + w["b2"]
        h3 = np.maximum(0, z @ d["W3"] + d["b3"])
        recon = h3 @ d["W4"] + d["b4"]

        recon_denorm = recon * self._std + self._mean
        return pd.DataFrame(
            recon_denorm, index=feature_matrix.index, columns=feature_matrix.columns
        )

    def reconstruction_error(self, feature_matrix: pd.DataFrame) -> float:
        """Compute mean squared reconstruction error."""
        recon = self.reconstruct(feature_matrix)
        original = feature_matrix.fillna(0.0)
        return float(((recon - original) ** 2).mean().mean())
