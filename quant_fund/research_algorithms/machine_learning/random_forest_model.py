"""Random forest model for directional return prediction.

Ensemble classifier used as a validation signal alongside GBT.
Rolling expanding-window approach.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator

logger = logging.getLogger(__name__)


class RandomForestModel(BaseFeatureGenerator):
    """Random forest classifier for directional prediction.

    Predicts probability of positive forward return. Used as a
    validation signal alongside the primary GBT model.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._n_estimators = cfg.get("rf_n_estimators", 200)
        self._max_depth = cfg.get("rf_max_depth", 8)
        self._forward_days = cfg.get("rf_forward_days", 5)
        self._min_train_days = cfg.get("rf_min_train_days", 252)
        self._model = None
        super().__init__(
            feature_name="rf_alpha",
            lookback_days=cfg.get("rf_lookback_days", 504),
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
            latest_features = td[feature_cols].iloc[-1:].values
            if np.any(np.isnan(latest_features)):
                result[ticker] = np.nan
                continue
            try:
                prob = self._model.predict_proba(latest_features)[0]
                # Use probability of positive class minus 0.5 as signal
                result[ticker] = float(prob[1] - 0.5) if len(prob) > 1 else 0.0
            except Exception:
                result[ticker] = np.nan

        return pd.Series(result, name=self.feature_name)

    def train_model(
        self,
        feature_matrix: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> dict:
        """Train the random forest classifier.

        Args:
            feature_matrix: Features DataFrame.
            forward_returns: Forward returns; converted to binary labels (>0 = 1).

        Returns:
            Training metrics dict.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            logger.error("scikit-learn required for RF model")
            return {"error": "sklearn not installed"}

        valid_mask = feature_matrix.notna().all(axis=1) & forward_returns.notna()
        X = feature_matrix[valid_mask].values
        y = (forward_returns[valid_mask] > 0).astype(int).values

        if len(X) < self._min_train_days:
            return {"error": "insufficient data", "n_samples": len(X)}

        model = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
        self._model = model

        train_accuracy = model.score(X, y)
        return {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "train_accuracy": train_accuracy,
        }

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.50)
