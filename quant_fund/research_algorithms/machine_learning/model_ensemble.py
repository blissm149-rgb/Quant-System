"""Model ensemble for combining multiple alpha signals.

Supports weighted averaging of model predictions with NaN handling,
correlation-based diversity evaluation, and configurable weighting.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """Weighted ensemble of multiple alpha models.

    Combines predictions from multiple models using configurable weights,
    handles missing predictions gracefully, and evaluates diversity.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._diversity_threshold = cfg.get("ensemble_diversity_threshold", 0.7)
        self._models: Dict[str, object] = {}
        self._weights: Dict[str, float] = {}

    def add_model(self, name: str, model, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self._models[name] = model
        self._weights[name] = weight

    def predict(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Generate ensemble prediction by weighted averaging.

        Models that return NaN for a ticker are excluded from that
        ticker's average (weight is redistributed).
        """
        if not self._models:
            return pd.Series(dtype=float)

        predictions = {}
        for name, model in self._models.items():
            try:
                if hasattr(model, "compute"):
                    pred = model.compute(data, as_of=as_of)
                elif hasattr(model, "predict"):
                    pred = model.predict(data)
                else:
                    continue
                predictions[name] = pred
            except Exception as e:
                logger.warning("Ensemble model %s failed: %s", name, e)

        if not predictions:
            return pd.Series(dtype=float)

        pred_df = pd.DataFrame(predictions)
        weight_series = pd.Series(self._weights)

        # Weighted average with NaN handling
        result = {}
        for idx in pred_df.index:
            row = pred_df.loc[idx]
            valid = row.dropna()
            if valid.empty:
                result[idx] = np.nan
                continue
            w = weight_series[valid.index]
            w_sum = w.sum()
            if w_sum == 0:
                result[idx] = np.nan
            else:
                result[idx] = (valid * w).sum() / w_sum

        return pd.Series(result, name="ensemble_alpha")

    def evaluate_diversity(
        self,
        predictions: Dict[str, pd.Series],
    ) -> Dict:
        """Evaluate diversity of ensemble members.

        Args:
            predictions: Dict of model name -> prediction Series.

        Returns:
            Dict with mean_correlation, pairwise_correlations, and
            diversity_warning flag.
        """
        if len(predictions) < 2:
            return {
                "mean_correlation": 0.0,
                "pairwise_correlations": {},
                "diversity_warning": False,
            }

        pred_df = pd.DataFrame(predictions).dropna()
        if len(pred_df) < 5:
            return {
                "mean_correlation": 0.0,
                "pairwise_correlations": {},
                "diversity_warning": False,
            }

        corr_matrix = pred_df.corr(method="spearman")
        names = list(predictions.keys())
        pairwise = {}
        corr_values = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                c = corr_matrix.loc[names[i], names[j]]
                if not np.isnan(c):
                    pairwise[f"{names[i]}_vs_{names[j]}"] = float(c)
                    corr_values.append(abs(c))

        mean_corr = float(np.mean(corr_values)) if corr_values else 0.0
        warning = mean_corr > self._diversity_threshold

        if warning:
            logger.warning(
                "Low ensemble diversity: mean |correlation| = %.3f > %.3f",
                mean_corr,
                self._diversity_threshold,
            )

        return {
            "mean_correlation": mean_corr,
            "pairwise_correlations": pairwise,
            "diversity_warning": warning,
        }
