"""LSTM temporal model for sequence-based representations.

Processes rolling windows of feature data to produce per-ticker latent
vectors representing recent behaviour trajectory. Uses a simplified
numpy-based LSTM for portability.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalModelLSTM:
    """LSTM-based temporal model for sequence feature extraction.

    Processes rolling windows of features and produces a latent vector
    per ticker capturing temporal dynamics.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._hidden_dim = cfg.get("lstm_hidden_dim", 32)
        self._sequence_length = cfg.get("lstm_sequence_length", 60)
        self._output_dim = cfg.get("lstm_output_dim", 16)
        self._learning_rate = cfg.get("lstm_learning_rate", 0.001)
        self._epochs = cfg.get("lstm_epochs", 50)
        self._seed = cfg.get("random_seed", 42)
        self._is_fitted = False
        self._weights = None
        self._input_dim = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, sequences: dict) -> dict:
        """Train the LSTM on sequences of feature data using BPTT.

        Uses autoencoder-style training: the model learns to reconstruct
        a compressed representation of each sequence. The loss is MSE
        between the output projection and a target derived from the
        sequence mean (a simple self-supervised objective).

        Args:
            sequences: Dict mapping ticker -> DataFrame (time × features),
                sorted chronologically.

        Returns:
            Training metrics including initial and final loss.
        """
        all_seqs = []
        for ticker, df in sequences.items():
            X = df.values.astype(np.float64)
            if len(X) < self._sequence_length:
                continue
            for i in range(len(X) - self._sequence_length + 1):
                all_seqs.append(X[i : i + self._sequence_length])

        if not all_seqs:
            return {"error": "insufficient sequence data"}

        all_seqs = np.array(all_seqs)  # (n_sequences, seq_len, n_features)
        self._input_dim = all_seqs.shape[2]

        rng = np.random.default_rng(self._seed)
        d = self._input_dim
        h = self._hidden_dim

        # LSTM gate weights (simplified: combined input+hidden)
        scale = np.sqrt(2.0 / (d + h))
        self._weights = {
            "Wf": rng.normal(0, scale, (d + h, h)),
            "bf": np.zeros(h),
            "Wi": rng.normal(0, scale, (d + h, h)),
            "bi": np.zeros(h),
            "Wc": rng.normal(0, scale, (d + h, h)),
            "bc": np.zeros(h),
            "Wo": rng.normal(0, scale, (d + h, h)),
            "bo": np.zeros(h),
            "Wy": rng.normal(0, np.sqrt(2.0 / (h + self._output_dim)), (h, self._output_dim)),
            "by": np.zeros(self._output_dim),
        }

        # Training targets: compressed representation of each sequence
        # Use mean of last few timesteps as target (self-supervised)
        targets = np.zeros((len(all_seqs), self._output_dim))
        for i, seq in enumerate(all_seqs):
            seq_mean = seq[-5:].mean(axis=0)  # last 5 steps mean
            # Project to output_dim via simple truncation/padding
            targets[i] = seq_mean[:self._output_dim] if len(seq_mean) >= self._output_dim \
                else np.pad(seq_mean, (0, self._output_dim - len(seq_mean)))

        # BPTT training loop
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
                batch_seqs = all_seqs[batch_idx]
                batch_targets = targets[batch_idx]

                # Accumulate gradients
                grads = {k: np.zeros_like(v) for k, v in self._weights.items()}
                batch_loss = 0.0

                for seq, target in zip(batch_seqs, batch_targets):
                    # Forward pass with caching for backprop
                    h_states, c_states, gates_cache = self._forward_with_cache(seq)
                    final_h = h_states[-1]
                    output = final_h @ self._weights["Wy"] + self._weights["by"]

                    # MSE loss
                    error = output - target
                    loss = 0.5 * np.sum(error ** 2)
                    batch_loss += loss

                    # Backprop through output layer
                    d_Wy = np.outer(final_h, error)
                    d_by = error.copy()
                    d_h = error @ self._weights["Wy"].T

                    # BPTT through time
                    d_weights = self._bptt(seq, h_states, c_states, gates_cache, d_h)

                    grads["Wy"] += d_Wy
                    grads["by"] += d_by
                    for k in d_weights:
                        grads[k] += d_weights[k]

                # Average gradients and clip
                bs = len(batch_idx)
                for k in grads:
                    grads[k] /= bs

                # Gradient clipping
                total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
                if total_norm > max_grad_norm:
                    clip_factor = max_grad_norm / total_norm
                    for k in grads:
                        grads[k] *= clip_factor

                # Update weights
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
            "n_sequences": len(all_seqs),
            "input_dim": self._input_dim,
            "hidden_dim": self._hidden_dim,
            "output_dim": self._output_dim,
            "initial_loss": float(initial_loss) if initial_loss is not None else 0.0,
            "final_loss": float(final_loss) if final_loss is not None else 0.0,
            "epochs": self._epochs,
        }

    def _forward_with_cache(self, seq: np.ndarray):
        """Forward pass caching intermediate states for backprop."""
        w = self._weights
        T = len(seq)
        h = np.zeros(self._hidden_dim)
        c = np.zeros(self._hidden_dim)

        h_states = [h.copy()]
        c_states = [c.copy()]
        gates_cache = []

        for t in range(T):
            x = seq[t]
            combined = np.concatenate([x, h])
            f = self._sigmoid(combined @ w["Wf"] + w["bf"])
            i = self._sigmoid(combined @ w["Wi"] + w["bi"])
            c_hat = np.tanh(combined @ w["Wc"] + w["bc"])
            c = f * c + i * c_hat
            o = self._sigmoid(combined @ w["Wo"] + w["bo"])
            h = o * np.tanh(c)

            h_states.append(h.copy())
            c_states.append(c.copy())
            gates_cache.append({
                "x": x, "combined": combined,
                "f": f, "i": i, "c_hat": c_hat, "o": o,
                "c_prev": c_states[-2],
            })

        return h_states, c_states, gates_cache

    def _bptt(self, seq, h_states, c_states, gates_cache, d_h_final):
        """Backpropagation through time for LSTM gates."""
        w = self._weights
        T = len(seq)
        grads = {k: np.zeros_like(v) for k, v in w.items() if k not in ("Wy", "by")}

        d_h = d_h_final.copy()
        d_c = np.zeros(self._hidden_dim)

        for t in reversed(range(T)):
            gc = gates_cache[t]
            c_t = c_states[t + 1]
            tanh_c = np.tanh(c_t)

            d_o = d_h * tanh_c
            d_c += d_h * gc["o"] * (1 - tanh_c ** 2)

            d_f = d_c * gc["c_prev"]
            d_i = d_c * gc["c_hat"]
            d_c_hat = d_c * gc["i"]
            d_c_prev = d_c * gc["f"]

            # Gate derivatives (sigmoid/tanh)
            d_f_raw = d_f * gc["f"] * (1 - gc["f"])
            d_i_raw = d_i * gc["i"] * (1 - gc["i"])
            d_c_hat_raw = d_c_hat * (1 - gc["c_hat"] ** 2)
            d_o_raw = d_o * gc["o"] * (1 - gc["o"])

            combined = gc["combined"]
            grads["Wf"] += np.outer(combined, d_f_raw)
            grads["bf"] += d_f_raw
            grads["Wi"] += np.outer(combined, d_i_raw)
            grads["bi"] += d_i_raw
            grads["Wc"] += np.outer(combined, d_c_hat_raw)
            grads["bc"] += d_c_hat_raw
            grads["Wo"] += np.outer(combined, d_o_raw)
            grads["bo"] += d_o_raw

            # Propagate to previous hidden state
            d_combined = (
                d_f_raw @ w["Wf"].T
                + d_i_raw @ w["Wi"].T
                + d_c_hat_raw @ w["Wc"].T
                + d_o_raw @ w["Wo"].T
            )
            d_h = d_combined[self._input_dim:]
            d_c = d_c_prev

        return grads

    def encode(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """Produce latent vectors for each ticker from recent data.

        Args:
            data: DataFrame with MultiIndex (date, ticker) and feature columns.
            as_of: Point-in-time boundary.

        Returns:
            DataFrame indexed by ticker with latent vector columns.
        """
        if not self._is_fitted:
            raise RuntimeError("LSTM must be fitted before encoding")

        tickers = data.index.get_level_values("ticker").unique()
        results = {}

        for ticker in tickers:
            try:
                td = data.xs(ticker, level="ticker").sort_index()
            except KeyError:
                continue
            if len(td) < self._sequence_length:
                continue
            seq = td.iloc[-self._sequence_length :].values.astype(np.float64)
            hidden = self._forward_sequence(seq)
            output = hidden @ self._weights["Wy"] + self._weights["by"]
            results[ticker] = output

        if not results:
            return pd.DataFrame()

        cols = [f"lstm_latent_{i}" for i in range(self._output_dim)]
        return pd.DataFrame.from_dict(results, orient="index", columns=cols)

    def _forward_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Run LSTM forward pass on a single sequence, return final hidden state."""
        h = np.zeros(self._hidden_dim)
        c = np.zeros(self._hidden_dim)
        w = self._weights

        for t in range(len(seq)):
            x = seq[t]
            combined = np.concatenate([x, h])
            f = self._sigmoid(combined @ w["Wf"] + w["bf"])
            i = self._sigmoid(combined @ w["Wi"] + w["bi"])
            c_hat = np.tanh(combined @ w["Wc"] + w["bc"])
            c = f * c + i * c_hat
            o = self._sigmoid(combined @ w["Wo"] + w["bo"])
            h = o * np.tanh(c)

        return h

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
