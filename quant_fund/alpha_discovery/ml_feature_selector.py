"""ML-based feature selection using importance ranking and redundancy removal.

Applies permutation importance and mutual information to rank features
by predictive power, then removes redundant features with high pairwise
correlation to a higher-ranked feature.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureRanking:
    """Result of feature selection."""

    selected_features: List[str]
    feature_scores: pd.Series  # importance score per feature
    removed_features: List[str]
    removal_reasons: dict  # feature_name -> reason string


class MLFeatureSelector:
    """Selects the most predictive non-redundant features.

    1. Rank features by importance (permutation or mutual information).
    2. Remove redundant features (|correlation| > threshold with higher-ranked).
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._correlation_threshold = cfg.get("fs_correlation_threshold", 0.85)
        self._min_importance = cfg.get("fs_min_importance", 0.0)
        self._method = cfg.get("fs_method", "mutual_information")
        self._seed = cfg.get("random_seed", 42)

    def select(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        max_features: Optional[int] = None,
    ) -> FeatureRanking:
        """Select the most predictive non-redundant features.

        Args:
            features: DataFrame of candidate features (samples × features).
            target: Target variable (e.g. forward returns).
            max_features: Maximum number of features to keep.

        Returns:
            FeatureRanking with selected and removed features.
        """
        valid_mask = features.notna().all(axis=1) & target.notna()
        X = features[valid_mask]
        y = target[valid_mask]

        if len(X) < 30:
            logger.warning("Insufficient data for feature selection: %d rows", len(X))
            return FeatureRanking(
                selected_features=list(features.columns),
                feature_scores=pd.Series(0.0, index=features.columns),
                removed_features=[],
                removal_reasons={},
            )

        scores = self._compute_importance(X, y)

        ranked_features = scores.sort_values(ascending=False).index.tolist()
        selected, removed, reasons = self._remove_redundant(
            X, ranked_features, scores
        )

        low_importance = [
            f for f in selected if scores[f] < self._min_importance
        ]
        for f in low_importance:
            selected.remove(f)
            removed.append(f)
            reasons[f] = f"importance {scores[f]:.4f} below threshold {self._min_importance}"

        if max_features and len(selected) > max_features:
            excess = selected[max_features:]
            selected = selected[:max_features]
            for f in excess:
                removed.append(f)
                reasons[f] = "exceeded max_features limit"

        return FeatureRanking(
            selected_features=selected,
            feature_scores=scores,
            removed_features=removed,
            removal_reasons=reasons,
        )

    def _compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Compute feature importance scores."""
        if self._method == "mutual_information":
            return self._mutual_information_scores(X, y)
        elif self._method == "correlation":
            return self._correlation_scores(X, y)
        else:
            return self._correlation_scores(X, y)

    def _mutual_information_scores(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.Series:
        """Estimate mutual information via binned entropy approximation."""
        scores = {}
        n_bins = min(20, max(5, len(X) // 20))
        for col in X.columns:
            try:
                x = X[col].values
                y_vals = y.values
                finite_mask = np.isfinite(x) & np.isfinite(y_vals)
                x_clean = x[finite_mask]
                y_clean = y_vals[finite_mask]
                if len(x_clean) < 20:
                    scores[col] = 0.0
                    continue
                x_bins = np.digitize(x_clean, np.linspace(x_clean.min(), x_clean.max(), n_bins))
                y_bins = np.digitize(y_clean, np.linspace(y_clean.min(), y_clean.max(), n_bins))
                mi = self._compute_mi(x_bins, y_bins)
                scores[col] = mi
            except Exception:
                scores[col] = 0.0
        return pd.Series(scores)

    def _compute_mi(self, x_bins: np.ndarray, y_bins: np.ndarray) -> float:
        """Compute mutual information from binned data."""
        n = len(x_bins)
        joint = {}
        x_counts = {}
        y_counts = {}
        for xi, yi in zip(x_bins, y_bins):
            joint[(xi, yi)] = joint.get((xi, yi), 0) + 1
            x_counts[xi] = x_counts.get(xi, 0) + 1
            y_counts[yi] = y_counts.get(yi, 0) + 1

        mi = 0.0
        for (xi, yi), count in joint.items():
            p_xy = count / n
            p_x = x_counts[xi] / n
            p_y = y_counts[yi] / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))
        return max(0.0, mi)

    def _correlation_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Absolute Spearman correlation with target."""
        scores = {}
        for col in X.columns:
            corr = X[col].corr(y, method="spearman")
            scores[col] = abs(corr) if not np.isnan(corr) else 0.0
        return pd.Series(scores)

    def _remove_redundant(
        self,
        X: pd.DataFrame,
        ranked: List[str],
        scores: pd.Series,
    ) -> Tuple[List[str], List[str], dict]:
        """Remove features that are highly correlated with a higher-ranked feature."""
        selected = []
        removed = []
        reasons = {}

        for feat in ranked:
            redundant_with = None
            for kept in selected:
                corr = X[feat].corr(X[kept], method="spearman")
                if abs(corr) > self._correlation_threshold:
                    redundant_with = kept
                    break
            if redundant_with is None:
                selected.append(feat)
            else:
                removed.append(feat)
                reasons[feat] = (
                    f"redundant with {redundant_with} "
                    f"(|corr|={abs(X[feat].corr(X[redundant_with], method='spearman')):.3f})"
                )
        return selected, removed, reasons
