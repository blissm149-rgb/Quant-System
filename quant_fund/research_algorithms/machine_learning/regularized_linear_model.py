"""Regularized linear models for return prediction baselines.

Supports Ridge, Lasso, and ElasticNet regression with temporal
train/validation split and OOS evaluation.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator

logger = logging.getLogger(__name__)


class RegularizedLinearModel(BaseFeatureGenerator):
    """Regularized linear model (Ridge/Lasso/ElasticNet) for alpha prediction.

    Provides a simple, interpretable baseline against which tree-based
    and neural models should be compared.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._model_type = cfg.get("linear_model_type", "ridge")
        self._alpha = cfg.get("linear_alpha", 1.0)
        self._l1_ratio = cfg.get("linear_l1_ratio", 0.5)  # for ElasticNet
        self._forward_days = cfg.get("linear_forward_days", 5)
        self._min_train_days = cfg.get("linear_min_train_days", 252)
        self._seed = cfg.get("random_seed", 42)
        self._model = None
        self._scaler = None
        super().__init__(
            feature_name=f"{self._model_type}_alpha",
            lookback_days=cfg.get("linear_lookback_days", 504),
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
                if self._scaler is not None:
                    latest = self._scaler.transform(latest)
                result[ticker] = float(self._model.predict(latest)[0])
            except Exception:
                result[ticker] = np.nan

        return pd.Series(result, name=self.feature_name)

    def train_model(
        self,
        feature_matrix: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> dict:
        """Train regularized linear model with temporal OOS split."""
        try:
            from sklearn.linear_model import Ridge, Lasso, ElasticNet
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {"error": "sklearn not installed"}

        valid_mask = feature_matrix.notna().all(axis=1) & forward_returns.notna()
        X = feature_matrix[valid_mask].values
        y = forward_returns[valid_mask].values

        if len(X) < self._min_train_days:
            return {"error": "insufficient data", "n_samples": len(X)}

        # Temporal split: 90% train, 10% validation
        split_idx = int(len(X) * 0.9)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale on training data only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model_classes = {"ridge": Ridge, "lasso": Lasso, "elasticnet": ElasticNet}
        cls = model_classes.get(self._model_type, Ridge)

        kwargs = {"alpha": self._alpha, "random_state": self._seed}
        if self._model_type == "elasticnet":
            kwargs["l1_ratio"] = self._l1_ratio

        model = cls(**kwargs)
        model.fit(X_train_s, y_train)

        self._model = model
        self._scaler = scaler

        train_r2 = model.score(X_train_s, y_train)
        oos_r2 = model.score(X_val_s, y_val)

        return {
            "n_samples": len(X),
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "n_features": X.shape[1],
            "train_r2": train_r2,
            "oos_r2": oos_r2,
            "model_type": self._model_type,
        }

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.50)
