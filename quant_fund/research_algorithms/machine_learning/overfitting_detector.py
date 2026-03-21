"""Overfitting detection tools for ML models.

Provides train/test gap analysis, learning curves, complexity curves,
and overfit probability estimation via bootstrap.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_train_test_gap(
    train_metric: float,
    test_metric: float,
    gap_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Compute train-test gap and flag overfitting.

    Args:
        train_metric: Training set metric (e.g. R2).
        test_metric: Test set metric.
        gap_threshold: Threshold for overfitting flag.

    Returns:
        Dict with gap, ratio, and overfitting_flag.
    """
    gap = train_metric - test_metric
    ratio = test_metric / train_metric if train_metric != 0 else 0.0
    return {
        "gap": float(gap),
        "ratio": float(ratio),
        "overfitting_flag": gap > gap_threshold,
        "train_metric": float(train_metric),
        "test_metric": float(test_metric),
    }


def learning_curve(
    model_cls,
    model_config: dict,
    features: pd.DataFrame,
    returns: pd.Series,
    fractions: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate model at different training set sizes.

    Trains on 20%/40%/60%/80%/100% of data, evaluates on a fixed
    held-out test set (last 10%).

    Args:
        model_cls: Model class with train_model(features, returns) -> dict.
        model_config: Config for model constructor.
        features: Full feature matrix.
        returns: Full return series.
        fractions: List of training fractions.

    Returns:
        List of dicts with fraction, n_train, train_metric, test_metric.
    """
    if fractions is None:
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Hold out last 10% as fixed test set
    n = len(features)
    test_size = max(int(n * 0.1), 10)
    train_pool = features.iloc[:-test_size]
    returns_pool = returns.iloc[:-test_size]
    X_test = features.iloc[-test_size:]
    y_test = returns.iloc[-test_size:]

    results = []
    for frac in fractions:
        n_train = max(int(len(train_pool) * frac), 10)
        X_train = train_pool.iloc[:n_train]
        y_train = returns_pool.iloc[:n_train]

        model = model_cls(model_config)
        metrics = model.train_model(X_train, y_train)

        train_r2 = metrics.get("train_r2", 0.0)

        # Evaluate on test set
        if hasattr(model, "_model") and model._model is not None:
            try:
                if hasattr(model, "_scaler") and model._scaler is not None:
                    test_r2 = model._model.score(model._scaler.transform(X_test.values), y_test.values)
                else:
                    test_r2 = model._model.score(X_test.values, y_test.values)
            except Exception:
                test_r2 = 0.0
        else:
            test_r2 = metrics.get("oos_r2", 0.0)

        results.append({
            "fraction": frac,
            "n_train": n_train,
            "train_metric": train_r2,
            "test_metric": test_r2,
        })

    return results


def complexity_curve(
    model_cls,
    model_config: dict,
    features: pd.DataFrame,
    returns: pd.Series,
    param_name: str,
    param_values: List,
) -> List[Dict[str, Any]]:
    """Evaluate model at different complexity levels.

    Varies a single hyperparameter and reports train/OOS metrics.

    Args:
        model_cls: Model class.
        model_config: Base config.
        features: Feature matrix.
        returns: Returns series.
        param_name: Name of the config parameter to vary.
        param_values: List of values to try.

    Returns:
        List of dicts with param_value, train_metric, test_metric.
    """
    results = []
    for val in param_values:
        cfg = dict(model_config)
        cfg[param_name] = val
        model = model_cls(cfg)
        metrics = model.train_model(features, returns)
        results.append({
            "param_value": val,
            "train_metric": metrics.get("train_r2", 0.0),
            "test_metric": metrics.get("oos_r2", 0.0),
        })
    return results


def overfit_probability(
    fold_oos_ics: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """Estimate probability that true OOS IC <= 0.

    Uses bootstrap resampling of fold-level OOS ICs.

    Args:
        fold_oos_ics: List of OOS IC values from cross-validation folds.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        Probability that true OOS IC <= 0.
    """
    if len(fold_oos_ics) == 0:
        return 1.0

    rng = np.random.default_rng(seed)
    ics = np.array(fold_oos_ics)
    n = len(ics)

    n_negative = 0
    for _ in range(n_bootstrap):
        sample = ics[rng.integers(0, n, size=n)]
        if sample.mean() <= 0:
            n_negative += 1

    return n_negative / n_bootstrap
