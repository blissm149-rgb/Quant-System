"""Feature importance tracking and drift detection.

Records feature importance scores from trained models over time and detects
when importance rankings shift significantly, which may indicate regime
changes or data quality issues.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ImportanceRecord:
    """A single importance observation."""

    as_of: pd.Timestamp
    model_name: str
    feature_importances: Dict[str, float]


class FeatureImportanceTracker:
    """Tracks feature importance scores over time.

    Records importance from each model training run and provides
    rolling aggregations, top-feature queries, and drift detection.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._drift_threshold = cfg.get("importance_drift_threshold", 0.3)
        self._min_records = cfg.get("importance_min_records", 5)
        self._records: List[ImportanceRecord] = []

    def record_importance(
        self,
        as_of: pd.Timestamp,
        model_name: str,
        feature_importances: Dict[str, float],
    ) -> None:
        """Record a feature importance observation.

        Args:
            as_of: Date of the training run.
            model_name: Name/identifier of the model.
            feature_importances: Dict mapping feature name to importance score.
        """
        self._records.append(
            ImportanceRecord(
                as_of=as_of,
                model_name=model_name,
                feature_importances=feature_importances,
            )
        )

    def get_importance_history(
        self, model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Return importance history as a DataFrame.

        Returns:
            DataFrame with DatetimeIndex, columns are feature names,
            values are importance scores.
        """
        records = self._records
        if model_name is not None:
            records = [r for r in records if r.model_name == model_name]

        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            row = {"as_of": r.as_of, "model_name": r.model_name}
            row.update(r.feature_importances)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index("as_of").drop(columns=["model_name"], errors="ignore")
        return df.sort_index()

    def get_rolling_importance(
        self,
        model_name: str,
        window: int = 10,
    ) -> pd.DataFrame:
        """Compute rolling mean importance over the last `window` records.

        Args:
            model_name: Model to filter by.
            window: Number of recent records to average.

        Returns:
            DataFrame with rolling mean importances.
        """
        history = self.get_importance_history(model_name=model_name)
        if history.empty:
            return pd.DataFrame()
        return history.rolling(window=min(window, len(history)), min_periods=1).mean()

    def get_top_features(
        self,
        model_name: str,
        n: int = 10,
        last_k: int = 5,
    ) -> List[str]:
        """Return the top-N features by mean importance over the last K records.

        Args:
            model_name: Model to filter by.
            n: Number of top features to return.
            last_k: Number of recent records to average.

        Returns:
            List of feature names sorted by descending importance.
        """
        history = self.get_importance_history(model_name=model_name)
        if history.empty:
            return []

        recent = history.tail(last_k)
        mean_importance = recent.mean().sort_values(ascending=False)
        return mean_importance.head(n).index.tolist()

    def detect_importance_drift(
        self,
        model_name: str,
        window: int = 5,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Detect if feature importance rankings have shifted significantly.

        Compares the rank correlation of feature importances between the
        first half and second half of the last `2 * window` records.

        Args:
            model_name: Model to check.
            window: Half-window size.

        Returns:
            Tuple of (drift_detected, rank_correlation, per_feature_drift).
            drift_detected is True if rank correlation < (1 - drift_threshold).
        """
        history = self.get_importance_history(model_name=model_name)
        if len(history) < 2 * self._min_records:
            return False, 1.0, {}

        recent = history.tail(2 * window)
        if len(recent) < 2 * self._min_records:
            return False, 1.0, {}

        mid = len(recent) // 2
        first_half = recent.iloc[:mid].mean()
        second_half = recent.iloc[mid:].mean()

        common = first_half.index.intersection(second_half.index)
        if len(common) < 2:
            return False, 1.0, {}

        rank_corr = first_half[common].corr(second_half[common], method="spearman")
        if np.isnan(rank_corr):
            rank_corr = 1.0

        per_feature_drift = {}
        for feat in common:
            drift = abs(second_half[feat] - first_half[feat])
            per_feature_drift[feat] = float(drift)

        drift_detected = rank_corr < (1.0 - self._drift_threshold)
        return drift_detected, float(rank_corr), per_feature_drift

    @property
    def n_records(self) -> int:
        return len(self._records)
