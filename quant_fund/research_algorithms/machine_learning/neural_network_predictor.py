"""Neural network predictor for return forecasting.

MLP-based model using features (optionally including embeddings from
representation_learning). Higher capacity; more regularisation required.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator

logger = logging.getLogger(__name__)


class NeuralNetworkPredictor(BaseFeatureGenerator):
    """MLP-based predictor for forward return estimation.

    Uses sklearn MLPRegressor as the backend. For production use,
    consider replacing with PyTorch/TensorFlow for GPU acceleration.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._hidden_layers = tuple(cfg.get("nn_hidden_layers", [64, 32]))
        self._learning_rate = cfg.get("nn_learning_rate", 0.001)
        self._max_iter = cfg.get("nn_max_iter", 500)
        self._alpha = cfg.get("nn_alpha", 0.01)  # L2 regularisation
        self._forward_days = cfg.get("nn_forward_days", 5)
        self._min_train_days = cfg.get("nn_min_train_days", 252)
        self._model = None
        super().__init__(
            feature_name="nn_alpha",
            lookback_days=cfg.get("nn_lookback_days", 504),
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        if self._model is None:
            tickers = data.index.get_level_values("ticker").unique()
            return pd.Series(np.nan, index=tickers, name=self.feature_name)

        tickers = data.index.get_level_values("ticker").unique()
        feature_cols = [c for c in data.columns if c not in ("open", "high", "low", "close", "volume")]

        if not feature_cols:
            return pd.Series(np.nan, index=tickers, name=self.feature_name)

        result = {}
        for ticker in tickers:
            try:
                td = data.xs(ticker, level="ticker").sort_index()
            except KeyError:
                result[ticker] = np.nan
                continue
            if td.empty:
                result[ticker] = np.nan
                continue
            latest = td[feature_cols].iloc[-1:].values
            if np.any(np.isnan(latest)):
                result[ticker] = np.nan
                continue
            try:
                pred = self._model.predict(latest)[0]
                result[ticker] = float(pred)
            except Exception:
                result[ticker] = np.nan

        return pd.Series(result, name=self.feature_name)

    def train_model(
        self,
        feature_matrix: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> dict:
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn required for neural network model")
            return {"error": "sklearn not installed"}

        valid_mask = feature_matrix.notna().all(axis=1) & forward_returns.notna()
        X = feature_matrix[valid_mask].values
        y = forward_returns[valid_mask].values

        if len(X) < self._min_train_days:
            return {"error": "insufficient data", "n_samples": len(X)}

        # Temporal split: train on first 90%, evaluate OOS on last 10%.
        # Data is assumed chronologically ordered (by DatetimeIndex).
        split_idx = int(len(X) * 0.9)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Fit scaler on training data only to prevent information leakage.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = MLPRegressor(
            hidden_layer_sizes=self._hidden_layers,
            learning_rate_init=self._learning_rate,
            max_iter=self._max_iter,
            alpha=self._alpha,
            random_state=42,
            early_stopping=False,
        )
        model.fit(X_train_scaled, y_train)

        self._scaler = scaler
        self._model = model

        train_score = model.score(X_train_scaled, y_train)
        oos_score = model.score(X_val_scaled, y_val) if len(X_val) > 0 else float("nan")
        return {
            "n_samples": len(X),
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "n_features": X.shape[1],
            "train_r2": train_score,
            "oos_r2": oos_score,
            "n_iter": model.n_iter_,
        }

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.50)
