"""Walk-forward validation framework for ML models.

Implements expanding-window walk-forward cross-validation to evaluate
model performance on strictly out-of-sample data. No future data leaks
into training or validation.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold."""

    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    oos_ic: float
    oos_r2: float


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward validation."""

    fold_metrics: List[FoldMetrics]
    mean_oos_ic: float
    std_oos_ic: float
    ic_tstat: float
    mean_oos_r2: float

    def is_significant(self, min_ic: float = 0.03, min_tstat: float = 2.0) -> bool:
        """Check if the model shows statistically significant OOS performance."""
        return self.mean_oos_ic > min_ic and self.ic_tstat > min_tstat


class WalkForwardValidator:
    """Expanding-window walk-forward cross-validation.

    Parameters
    ----------
    min_train_days : int
        Minimum number of training observations before first test fold.
    test_days : int
        Number of observations in each test fold.
    step_days : int
        Number of observations to step forward between folds.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._min_train_days = cfg.get("wf_min_train_days", 504)
        self._test_days = cfg.get("wf_test_days", 63)
        self._step_days = cfg.get("wf_step_days", 21)

    def generate_splits(
        self, dates: pd.DatetimeIndex
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Generate (train_dates, test_dates) splits.

        Uses an expanding window: training always starts from the beginning,
        test window slides forward by step_days each fold.

        Args:
            dates: Sorted DatetimeIndex of all available dates.

        Returns:
            List of (train_dates, test_dates) tuples.
        """
        dates = dates.sort_values()
        n = len(dates)
        splits = []

        test_start_idx = self._min_train_days
        while test_start_idx + self._test_days <= n:
            test_end_idx = test_start_idx + self._test_days
            train_dates = dates[:test_start_idx]
            test_dates = dates[test_start_idx:test_end_idx]
            splits.append((train_dates, test_dates))
            test_start_idx += self._step_days

        return splits

    def validate_model(
        self,
        model_cls,
        model_config: dict,
        feature_matrix: pd.DataFrame,
        forward_returns: pd.Series,
        dates: pd.DatetimeIndex,
    ) -> WalkForwardResult:
        """Run walk-forward validation on a model.

        Args:
            model_cls: Model class with train_model(features, returns) and
                compute_scores(features) -> Series methods.
            model_config: Config dict passed to model constructor.
            feature_matrix: Full feature matrix indexed by date (or date+ticker).
            forward_returns: Forward returns aligned with feature_matrix.
            dates: DatetimeIndex for splitting.

        Returns:
            WalkForwardResult with fold-level and aggregate metrics.
        """
        splits = self.generate_splits(dates)

        if not splits:
            logger.warning("No valid walk-forward splits generated")
            return WalkForwardResult(
                fold_metrics=[],
                mean_oos_ic=0.0,
                std_oos_ic=0.0,
                ic_tstat=0.0,
                mean_oos_r2=0.0,
            )

        fold_metrics = []
        for fold_idx, (train_dates, test_dates) in enumerate(splits):
            train_mask = feature_matrix.index.isin(train_dates)
            test_mask = feature_matrix.index.isin(test_dates)

            X_train = feature_matrix[train_mask]
            y_train = forward_returns[train_mask]
            X_test = feature_matrix[test_mask]
            y_test = forward_returns[test_mask]

            # Train model
            model = model_cls(model_config)
            if hasattr(model, "train_model"):
                model.train_model(X_train, y_train)
            elif hasattr(model, "fit"):
                model.fit(X_train, y_train)

            # Predict
            if hasattr(model, "predict"):
                preds = model.predict(X_test)
            elif hasattr(model, "compute_scores"):
                preds = model.compute_scores(X_test)
            else:
                preds = pd.Series(0.0, index=X_test.index)

            if isinstance(preds, np.ndarray):
                preds = pd.Series(preds, index=X_test.index)

            # Compute OOS metrics
            oos_ic = _rank_ic(preds, y_test)
            oos_r2 = _r_squared(preds, y_test)

            fold_metrics.append(
                FoldMetrics(
                    fold_idx=fold_idx,
                    train_start=train_dates[0],
                    train_end=train_dates[-1],
                    test_start=test_dates[0],
                    test_end=test_dates[-1],
                    n_train=len(X_train),
                    n_test=len(X_test),
                    oos_ic=oos_ic,
                    oos_r2=oos_r2,
                )
            )

        ic_values = [fm.oos_ic for fm in fold_metrics]
        r2_values = [fm.oos_r2 for fm in fold_metrics]
        mean_ic = np.mean(ic_values)
        std_ic = np.std(ic_values, ddof=1) if len(ic_values) > 1 else 0.0
        ic_t = (mean_ic / std_ic * np.sqrt(len(ic_values))) if std_ic > 0 else 0.0

        return WalkForwardResult(
            fold_metrics=fold_metrics,
            mean_oos_ic=float(mean_ic),
            std_oos_ic=float(std_ic),
            ic_tstat=float(ic_t),
            mean_oos_r2=float(np.mean(r2_values)),
        )


def _rank_ic(predictions: pd.Series, actuals: pd.Series) -> float:
    """Spearman rank IC between predictions and actuals."""
    aligned = pd.concat([predictions, actuals], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def _r_squared(predictions: pd.Series, actuals: pd.Series) -> float:
    """Out-of-sample R-squared."""
    aligned = pd.concat([predictions, actuals], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    ss_res = ((aligned.iloc[:, 0] - aligned.iloc[:, 1]) ** 2).sum()
    ss_tot = ((aligned.iloc[:, 1] - aligned.iloc[:, 1].mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
