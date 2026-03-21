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
        self._learning_rate = cfg.get("transformer_learning_rate", 0.001)
        self._epochs = cfg.get("transformer_epochs", 50)
        self._seed = cfg.get("random_seed", 42)
        self._is_fitted = False
        self._weights = None
        self._input_dim = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, sequences: dict) -> dict:
        """Train transformer on sequences using self-supervised reconstruction.

        The model learns to reconstruct a target derived from each sequence's
        mean values. Weights are updated via gradient descent with backprop
        through the attention and projection layers.

        Args:
            sequences: Dict mapping ticker -> DataFrame (time × features).

        Returns:
            Training metrics including initial and final loss.
        """
        all_seqs = []
        for ticker, df in sequences.items():
            X = df.values.astype(np.float64)
            if len(X) < 2:
                continue
            seq_len = min(self._sequence_length, len(X))
            for i in range(len(X) - seq_len + 1):
                all_seqs.append(X[i : i + seq_len])

        if not all_seqs:
            return {"error": "insufficient data"}

        self._input_dim = all_seqs[0].shape[1]

        rng = np.random.default_rng(self._seed)
        d = self._d_model

        # Input projection
        scale_in = np.sqrt(2.0 / (self._input_dim + d))
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

        # Training targets: compressed representation from sequence
        targets = np.zeros((len(all_seqs), self._output_dim))
        for i, seq in enumerate(all_seqs):
            seq_mean = seq.mean(axis=0)
            targets[i] = seq_mean[:self._output_dim] if len(seq_mean) >= self._output_dim \
                else np.pad(seq_mean, (0, self._output_dim - len(seq_mean)))

        # Training loop with gradient descent
        lr = self._learning_rate
        max_grad_norm = 5.0
        initial_loss = None
        final_loss = None
        n_samples = len(all_seqs)
        batch_size = min(32, n_samples)

        for epoch in range(self._epochs):
            indices = rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, n_samples, batch_size):
                batch_idx = indices[batch_start:batch_start + batch_size]
                grads = {k: np.zeros_like(v) for k, v in self._weights.items()}
                batch_loss = 0.0

                for idx in batch_idx:
                    seq = all_seqs[idx]
                    target = targets[idx]

                    # Forward pass with cache
                    output, cache = self._forward_with_cache(seq)

                    # MSE loss
                    error = output - target
                    loss = 0.5 * np.sum(error ** 2)
                    batch_loss += loss

                    # Backprop
                    d_grads = self._backward(error, cache)
                    for k in grads:
                        grads[k] += d_grads[k]

                bs = len(batch_idx)
                for k in grads:
                    grads[k] /= bs

                # Gradient clipping
                total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
                if total_norm > max_grad_norm:
                    for k in grads:
                        grads[k] *= max_grad_norm / total_norm

                for k in self._weights:
                    self._weights[k] -= lr * grads[k]

                epoch_loss += batch_loss / bs
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if initial_loss is None:
                initial_loss = avg_loss
            final_loss = avg_loss

        self._is_fitted = True
        return {
            "input_dim": self._input_dim,
            "d_model": self._d_model,
            "n_heads": self._n_heads,
            "output_dim": self._output_dim,
            "initial_loss": float(initial_loss) if initial_loss is not None else 0.0,
            "final_loss": float(final_loss) if final_loss is not None else 0.0,
            "epochs": self._epochs,
        }

    def _forward_with_cache(self, seq: np.ndarray):
        """Forward pass returning output and cache for backprop."""
        w = self._weights
        X_proj = seq @ w["W_in"] + w["b_in"]

        T = X_proj.shape[0]
        pos_enc = self._positional_encoding(T, self._d_model)
        X = X_proj + pos_enc[:T]

        Q = X @ w["W_q"]
        K = X @ w["W_k"]
        V = X @ w["W_v"]

        d_k = self._d_model / self._n_heads
        scores = Q @ K.T / np.sqrt(d_k)
        attn = self._softmax(scores)
        context = attn @ V

        X_res = X + context
        pooled = X_res.mean(axis=0)
        output = pooled @ w["W_out"] + w["b_out"]

        cache = {
            "seq": seq, "X_proj": X_proj, "X": X, "Q": Q, "K": K, "V": V,
            "scores": scores, "attn": attn, "context": context,
            "X_res": X_res, "pooled": pooled, "T": T, "d_k": d_k,
        }
        return output, cache

    def _backward(self, d_output, cache):
        """Backprop through transformer layers."""
        w = self._weights
        grads = {}

        # Output layer
        grads["W_out"] = np.outer(cache["pooled"], d_output)
        grads["b_out"] = d_output.copy()

        d_pooled = d_output @ w["W_out"].T
        T = cache["T"]
        d_X_res = np.tile(d_pooled / T, (T, 1))

        # Through residual
        d_context = d_X_res.copy()
        d_X = d_X_res.copy()

        # Through attention: context = attn @ V
        d_attn = d_context @ cache["V"].T
        d_V = cache["attn"].T @ d_context

        # Through softmax (simplified)
        d_scores = d_attn * cache["attn"] - cache["attn"] * (d_attn * cache["attn"]).sum(axis=-1, keepdims=True)
        d_scores /= np.sqrt(cache["d_k"])

        # Q, K gradients
        d_Q = d_scores @ cache["K"]
        d_K = d_scores.T @ cache["Q"]

        grads["W_q"] = cache["X"].T @ d_Q
        grads["W_k"] = cache["X"].T @ d_K
        grads["W_v"] = cache["X"].T @ d_V

        d_X += d_Q @ w["W_q"].T + d_K @ w["W_k"].T + d_V @ w["W_v"].T

        # Through input projection
        grads["W_in"] = cache["seq"].T @ d_X
        grads["b_in"] = d_X.sum(axis=0)

        return grads

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
