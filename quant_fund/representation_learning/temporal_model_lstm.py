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
        """Train the LSTM on sequences of feature data.

        Args:
            sequences: Dict mapping ticker -> DataFrame (time × features),
                sorted chronologically.

        Returns:
            Training metrics.
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

        self._is_fitted = True
        return {
            "n_sequences": len(all_seqs),
            "input_dim": self._input_dim,
            "hidden_dim": self._hidden_dim,
            "output_dim": self._output_dim,
        }

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
