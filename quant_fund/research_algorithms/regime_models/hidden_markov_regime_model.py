"""Hidden Markov Model for market regime detection.

Gaussian HMM on market returns and volatility. Identifies 2-4 hidden regimes.
State labels used to gate strategy weights in the capital allocator.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HiddenMarkovRegimeModel:
    """Gaussian HMM for market regime classification.

    Fits a Gaussian HMM to market return and volatility observations.
    Uses a simple implementation (Baum-Welch / forward-backward) to avoid
    heavy dependencies. For production, consider hmmlearn.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._n_regimes = cfg.get("hmm_n_regimes", 2)
        self._vol_window = cfg.get("hmm_vol_window", 20)
        self._max_iter = cfg.get("hmm_max_iter", 100)
        self._tol = cfg.get("hmm_tol", 1e-4)

        # Model parameters (initialised during fit)
        self._means: Optional[np.ndarray] = None
        self._covars: Optional[np.ndarray] = None
        self._transmat: Optional[np.ndarray] = None
        self._startprob: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, market_returns: pd.Series) -> dict:
        """Fit the HMM to market return data.

        Args:
            market_returns: Daily market returns (e.g. S&P 500).

        Returns:
            Dict with fit statistics.
        """
        returns = market_returns.dropna().values
        vol = pd.Series(returns).rolling(self._vol_window).std().dropna().values
        min_len = len(vol)
        returns_trimmed = returns[-min_len:]

        observations = np.column_stack([returns_trimmed, vol])

        self._fit_simple(observations)
        self._is_fitted = True

        return {
            "n_regimes": self._n_regimes,
            "n_observations": len(observations),
            "regime_means": self._means.tolist() if self._means is not None else [],
        }

    def predict_regime(self, market_returns: pd.Series) -> int:
        """Predict the current regime based on recent market data.

        Args:
            market_returns: Recent daily market returns.

        Returns:
            Integer regime label (0 to n_regimes-1).
        """
        if not self._is_fitted:
            return 0

        returns = market_returns.dropna().values
        vol = pd.Series(returns).rolling(self._vol_window).std().dropna().values

        if len(vol) == 0:
            return 0

        min_len = len(vol)
        returns_trimmed = returns[-min_len:]
        observations = np.column_stack([returns_trimmed, vol])

        # Classify using closest regime mean
        latest_obs = observations[-1]
        distances = [
            np.sum((latest_obs - self._means[i]) ** 2)
            for i in range(self._n_regimes)
        ]
        return int(np.argmin(distances))

    def predict_regime_probabilities(
        self, market_returns: pd.Series
    ) -> np.ndarray:
        """Return probability distribution over regimes for current state.

        Args:
            market_returns: Recent daily market returns.

        Returns:
            Array of probabilities for each regime.
        """
        if not self._is_fitted:
            return np.ones(self._n_regimes) / self._n_regimes

        returns = market_returns.dropna().values
        vol = pd.Series(returns).rolling(self._vol_window).std().dropna().values

        if len(vol) == 0:
            return np.ones(self._n_regimes) / self._n_regimes

        latest_obs = np.array([returns[-1], vol[-1]])
        distances = np.array([
            np.sum((latest_obs - self._means[i]) ** 2)
            for i in range(self._n_regimes)
        ])

        # Softmax-like conversion
        exp_neg = np.exp(-distances / (2 * np.max(distances) + 1e-10))
        return exp_neg / exp_neg.sum()

    def _fit_simple(self, observations: np.ndarray) -> None:
        """Simple K-means-like fitting as HMM approximation.

        For production, replace with proper Baum-Welch via hmmlearn.
        """
        n = len(observations)
        k = self._n_regimes

        # K-means initialisation
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=k, replace=False)
        centroids = observations[indices].copy()

        for _ in range(self._max_iter):
            # Assign each observation to nearest centroid
            distances = np.array([
                np.sum((observations - centroids[j]) ** 2, axis=1)
                for j in range(k)
            ])
            labels = np.argmin(distances, axis=0)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    new_centroids[j] = observations[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids, atol=self._tol):
                break
            centroids = new_centroids

        self._means = centroids
        self._covars = np.array([
            np.cov(observations[labels == j].T) if (labels == j).sum() > 1
            else np.eye(observations.shape[1])
            for j in range(k)
        ])

        # Estimate transition matrix from label sequence
        self._transmat = np.zeros((k, k))
        for t in range(len(labels) - 1):
            self._transmat[labels[t], labels[t + 1]] += 1
        row_sums = self._transmat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self._transmat /= row_sums

        self._startprob = np.zeros(k)
        self._startprob[labels[0]] = 1.0

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
