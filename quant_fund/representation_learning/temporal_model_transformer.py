"""Transformer temporal model for sequence-based representations.

Processes rolling windows of feature data using self-attention.
Preferred for longer contexts; produces per-ticker latent vectors.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalModelTransformer:
    """Transformer-based temporal model for sequence feature extraction.

    Uses simplified self-attention mechanism for portability.
    For production, swap in PyTorch nn.TransformerEncoder.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._d_model = cfg.get("transformer_d_model", 32)
        self._n_heads = cfg.get("transformer_n_heads", 4)
        self._sequence_length = cfg.get("transformer_sequence_length", 60)
        self._output_dim = cfg.get("transformer_output_dim", 16)
        self._seed = cfg.get("random_seed", 42)
        self._is_fitted = False
        self._weights = None
        self._input_dim = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, sequences: dict) -> dict:
        """Initialise transformer weights based on input dimensions.

        Args:
            sequences: Dict mapping ticker -> DataFrame (time × features).

        Returns:
            Initialisation metrics.
        """
        for ticker, df in sequences.items():
            if len(df) >= 2:
                self._input_dim = df.shape[1]
                break

        if self._input_dim is None:
            return {"error": "insufficient data"}

        rng = np.random.default_rng(self._seed)
        d = self._d_model

        # Input projection
        scale_in = np.sqrt(2.0 / (self._input_dim + d))
        # Attention weights (Q, K, V per head)
        head_dim = d // self._n_heads
        scale_attn = np.sqrt(2.0 / (d + head_dim))

        self._weights = {
            "W_in": rng.normal(0, scale_in, (self._input_dim, d)),
            "b_in": np.zeros(d),
            "W_q": rng.normal(0, scale_attn, (d, d)),
            "W_k": rng.normal(0, scale_attn, (d, d)),
            "W_v": rng.normal(0, scale_attn, (d, d)),
            "W_out": rng.normal(0, np.sqrt(2.0 / (d + self._output_dim)), (d, self._output_dim)),
            "b_out": np.zeros(self._output_dim),
        }

        self._is_fitted = True
        return {
            "input_dim": self._input_dim,
            "d_model": self._d_model,
            "n_heads": self._n_heads,
            "output_dim": self._output_dim,
        }

    def encode(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """Produce latent vectors for each ticker using self-attention.

        Args:
            data: DataFrame with MultiIndex (date, ticker) and feature columns.
            as_of: Point-in-time boundary.

        Returns:
            DataFrame indexed by ticker with latent vector columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before encoding")

        tickers = data.index.get_level_values("ticker").unique()
        results = {}

        for ticker in tickers:
            try:
                td = data.xs(ticker, level="ticker").sort_index()
            except KeyError:
                continue
            if len(td) < 2:
                continue
            seq = td.iloc[-self._sequence_length :].values.astype(np.float64)
            latent = self._forward(seq)
            results[ticker] = latent

        if not results:
            return pd.DataFrame()

        cols = [f"transformer_latent_{i}" for i in range(self._output_dim)]
        return pd.DataFrame.from_dict(results, orient="index", columns=cols)

    def _forward(self, seq: np.ndarray) -> np.ndarray:
        """Forward pass: project -> self-attention -> mean pool -> output."""
        w = self._weights
        # Input projection: (T, input_dim) -> (T, d_model)
        X = seq @ w["W_in"] + w["b_in"]

        # Add positional encoding (sinusoidal)
        T = X.shape[0]
        pos_enc = self._positional_encoding(T, self._d_model)
        X = X + pos_enc[:T]

        # Self-attention
        Q = X @ w["W_q"]
        K = X @ w["W_k"]
        V = X @ w["W_v"]

        d_k = self._d_model / self._n_heads
        scores = Q @ K.T / np.sqrt(d_k)
        attn = self._softmax(scores)
        context = attn @ V

        # Residual connection
        X = X + context

        # Mean pool over time dimension
        pooled = X.mean(axis=0)

        # Output projection
        output = pooled @ w["W_out"] + w["b_out"]
        return output

    @staticmethod
    def _positional_encoding(max_len: int, d_model: int) -> np.ndarray:
        """Sinusoidal positional encoding."""
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[: d_model // 2])
        return pe

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Row-wise softmax."""
        x_max = x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
