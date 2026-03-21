"""Systematic model comparison and benchmarking framework.

Evaluates multiple models (including baselines) on the same walk-forward
splits and produces a comparison table with OOS IC, R2, Sharpe, and
Deflated Sharpe Ratio.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelComparisonFramework:
    """Compares models against baselines using walk-forward evaluation.

    Produces a DataFrame summarizing each model's OOS performance and
    whether it beats all baseline models.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._min_train_days = cfg.get("wf_min_train_days", 200)
        self._test_days = cfg.get("wf_test_days", 63)
        self._step_days = cfg.get("wf_step_days", 21)

    def run_comparison(
        self,
        models: Dict[str, object],
        baselines: Dict[str, object],
        features: pd.DataFrame,
        returns: pd.Series,
        n_trials: int = 1,
    ) -> pd.DataFrame:
        """Run walk-forward comparison of models vs baselines.

        Args:
            models: Dict of model name -> model instance.
            baselines: Dict of baseline name -> baseline instance.
            features: Feature matrix with DatetimeIndex.
            returns: Forward returns aligned with features.
            n_trials: Number of trials for DSR computation.

        Returns:
            DataFrame with columns: model_name, is_baseline, mean_oos_ic,
            ic_tstat, oos_r2, dsr, beats_all_baselines.
        """
        all_models = {}
        model_flags = {}
        for name, m in baselines.items():
            all_models[name] = m
            model_flags[name] = True
        for name, m in models.items():
            all_models[name] = m
            model_flags[name] = False

        # Generate walk-forward splits
        dates = features.index
        n = len(dates)
        splits = []
        test_start = self._min_train_days
        while test_start + self._test_days <= n:
            test_end = test_start + self._test_days
            splits.append((test_start, test_end))
            test_start += self._step_days

        if not splits:
            return pd.DataFrame()

        results = []
        baseline_ics = {}

        for name, model in all_models.items():
            fold_ics = []
            fold_r2s = []

            for train_end, test_end in splits:
                X_train = features.iloc[:train_end]
                y_train = returns.iloc[:train_end]
                X_test = features.iloc[train_end:test_end]
                y_test = returns.iloc[train_end:test_end]

                model_instance = type(model)(getattr(model, "_config", None) if hasattr(model, "_config") else None)
                if hasattr(model_instance, "train_model"):
                    model_instance.train_model(X_train, y_train)
                if hasattr(model_instance, "predict"):
                    preds = model_instance.predict(X_test)
                else:
                    preds = pd.Series(0.0, index=X_test.index)

                if isinstance(preds, np.ndarray):
                    preds = pd.Series(preds, index=X_test.index)

                # IC
                aligned = pd.concat([preds, y_test], axis=1).dropna()
                if len(aligned) >= 5:
                    ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
                    if not np.isnan(ic):
                        fold_ics.append(ic)

                # R2
                if len(aligned) >= 2:
                    ss_res = ((aligned.iloc[:, 0] - aligned.iloc[:, 1]) ** 2).sum()
                    ss_tot = ((aligned.iloc[:, 1] - aligned.iloc[:, 1].mean()) ** 2).sum()
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    fold_r2s.append(r2)

            mean_ic = np.mean(fold_ics) if fold_ics else 0.0
            std_ic = np.std(fold_ics, ddof=1) if len(fold_ics) > 1 else 0.0
            ic_t = (mean_ic / std_ic * np.sqrt(len(fold_ics))) if std_ic > 0 else 0.0
            mean_r2 = np.mean(fold_r2s) if fold_r2s else 0.0

            # DSR (simplified)
            dsr = _simplified_dsr(mean_ic, std_ic, len(fold_ics), n_trials)

            if model_flags[name]:
                baseline_ics[name] = mean_ic

            results.append({
                "model_name": name,
                "is_baseline": model_flags[name],
                "mean_oos_ic": float(mean_ic),
                "ic_tstat": float(ic_t),
                "oos_r2": float(mean_r2),
                "dsr": float(dsr),
            })

        # Determine which models beat all baselines
        max_baseline_ic = max(baseline_ics.values()) if baseline_ics else 0.0
        for r in results:
            r["beats_all_baselines"] = r["mean_oos_ic"] > max_baseline_ic and not r["is_baseline"]

        return pd.DataFrame(results)


def _simplified_dsr(
    mean_ic: float, std_ic: float, n_folds: int, n_trials: int
) -> float:
    """Simplified DSR approximation using IC statistics."""
    if n_folds < 2 or std_ic == 0:
        return 0.0
    try:
        from scipy import stats as scipy_stats
        # Approximate: DSR = Phi((IC_mean - E[max IC under null]) / SE(IC))
        gamma = 0.5772
        if n_trials > 1:
            e_max = (1 - gamma) * scipy_stats.norm.ppf(1 - 1 / n_trials) + \
                    gamma * scipy_stats.norm.ppf(1 - 1 / (n_trials * np.e))
            e_max *= std_ic / np.sqrt(n_folds)
        else:
            e_max = 0.0
        se = std_ic / np.sqrt(n_folds)
        z = (mean_ic - e_max) / se
        return float(scipy_stats.norm.cdf(z))
    except Exception:
        return 0.5
