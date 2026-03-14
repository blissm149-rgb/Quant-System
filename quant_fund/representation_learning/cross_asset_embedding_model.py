"""Cross-asset embedding model for joint representations.

Learns joint representations across the stock universe, capturing
cross-sectional relationships (sector, style, co-movement). Outputs
a pairwise similarity matrix for use by stat-arb pair selection.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrossAssetEmbeddingModel:
    """Learns joint embeddings across the stock universe.

    Uses a factored covariance approach: embeddings are learned such that
    inner products approximate the return correlation structure.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._embedding_dim = cfg.get("cross_asset_embedding_dim", 16)
        self._learning_rate = cfg.get("cross_asset_lr", 0.01)
        self._epochs = cfg.get("cross_asset_epochs", 50)
        self._seed = cfg.get("random_seed", 42)
        self._embeddings: Optional[pd.DataFrame] = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, returns: pd.DataFrame) -> dict:
        """Learn embeddings from return covariance structure.

        Args:
            returns: DataFrame (dates × tickers) of daily returns.

        Returns:
            Training metrics.
        """
        clean_returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
        if clean_returns.shape[1] < 3 or clean_returns.shape[0] < 30:
            return {"error": "insufficient data"}

        # Compute correlation matrix
        corr_matrix = clean_returns.corr().values
        n = corr_matrix.shape[0]
        tickers = clean_returns.columns.tolist()

        # Learn embeddings via truncated SVD of correlation matrix
        rng = np.random.default_rng(self._seed)
        k = min(self._embedding_dim, n - 1)

        # Power iteration for top-k singular vectors
        U, S, Vt = self._truncated_svd(corr_matrix, k, rng)
        embeddings = U * np.sqrt(S)[np.newaxis, :]

        self._embeddings = pd.DataFrame(
            embeddings,
            index=tickers,
            columns=[f"cross_asset_{i}" for i in range(k)],
        )
        self._is_fitted = True

        # Reconstruction quality
        approx = embeddings @ embeddings.T
        recon_error = np.mean((corr_matrix - approx) ** 2)

        return {
            "n_tickers": n,
            "embedding_dim": k,
            "n_dates": clean_returns.shape[0],
            "reconstruction_mse": recon_error,
        }

    def get_embeddings(self) -> pd.DataFrame:
        """Get current ticker embeddings."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting embeddings")
        return self._embeddings.copy()

    def get_similarity_matrix(self) -> pd.DataFrame:
        """Compute pairwise cosine similarity from embeddings."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before computing similarity")

        emb = self._embeddings.values
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        normed = emb / norms
        sim = normed @ normed.T

        return pd.DataFrame(
            sim,
            index=self._embeddings.index,
            columns=self._embeddings.index,
        )

    def find_similar(self, ticker: str, top_n: int = 10) -> pd.Series:
        """Find the most similar tickers to a given ticker."""
        sim_matrix = self.get_similarity_matrix()
        if ticker not in sim_matrix.index:
            return pd.Series(dtype=float)
        sims = sim_matrix[ticker].drop(ticker, errors="ignore")
        return sims.nlargest(top_n)

    def _truncated_svd(self, A, k, rng):
        """Compute truncated SVD using numpy."""
        try:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            return U[:, :k], s[:k], Vt[:k, :]
        except np.linalg.LinAlgError:
            # Fallback: eigendecomposition for symmetric matrix
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            idx = np.argsort(eigenvalues)[::-1][:k]
            S = np.abs(eigenvalues[idx])
            U = eigenvectors[:, idx]
            return U, S, U.T
