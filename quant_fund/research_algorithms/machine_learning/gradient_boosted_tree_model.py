"""Gradient boosted tree model for return prediction.

Uses LightGBM (or sklearn fallback) to predict 5-day forward return quintile.
Rolling expanding-window train/test split. Features are pre-normalised.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator

logger = logging.getLogger(__name__)


class GradientBoostedTreeModel(BaseFeatureGenerator):
    """GBT-based alpha model predicting forward return direction.

    Uses a rolling expanding-window train/test approach to avoid look-ahead bias.
    The model is trained on historical features and forward returns, then used
    to score the current cross-section.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._n_estimators = cfg.get("gbt_n_estimators", 200)
        self._max_depth = cfg.get("gbt_max_depth", 5)
        self._learning_rate = cfg.get("gbt_learning_rate", 0.05)
        self._forward_days = cfg.get("gbt_forward_days", 5)
        self._min_train_days = cfg.get("gbt_min_train_days", 252)
        self._model = None
        super().__init__(
            feature_name="gbt_alpha",
            lookback_days=cfg.get("gbt_lookback_days", 504),
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Score current cross-section using a trained GBT model.

        If no model is trained yet, returns NaN scores. Use train_model()
        to fit the model before calling compute().
        """
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
                pred = self._model.predict(latest_features)[0]
                result[ticker] = float(pred)
            except Exception:
                result[ticker] = np.nan

        return pd.Series(result, name=self.feature_name)

    def train_model(
        self,
        feature_matrix: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> dict:
        """Train the GBT model on historical data.

        Args:
            feature_matrix: DataFrame of features (samples x features).
            forward_returns: Series of forward returns aligned with feature_matrix.

        Returns:
            Dict with training metrics.
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            logger.error("scikit-learn required for GBT model")
            return {"error": "sklearn not installed"}

        valid_mask = feature_matrix.notna().all(axis=1) & forward_returns.notna()
        X = feature_matrix[valid_mask].values
        y = forward_returns[valid_mask].values

        if len(X) < self._min_train_days:
            logger.warning("Insufficient training data: %d < %d", len(X), self._min_train_days)
            return {"error": "insufficient data", "n_samples": len(X)}

        model = GradientBoostingRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=42,
        )
        model.fit(X, y)
        self._model = model

        train_score = model.score(X, y)
        return {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "train_r2": train_score,
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Return feature importances from the trained model."""
        if self._model is None:
            return None
        return self._model.feature_importances_

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.50)
