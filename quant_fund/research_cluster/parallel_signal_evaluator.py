"""Parallel signal evaluator.

Evaluates a list of candidate signals in parallel on the same
out-of-sample test period. Returns a ranked DataFrame of signals
sorted by IC t-statistic.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class SignalEvaluation:
    """Evaluation result for a single signal."""

    signal_id: str
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_tstat: float = 0.0
    ic_ir: float = 0.0
    hit_rate: float = 0.0
    turnover: float = 0.0
    sharpe_estimate: float = 0.0
    num_periods: int = 0
    status: str = "completed"


def _evaluate_single_signal(
    signal_id: str,
    signal_values: np.ndarray,
    forward_returns: np.ndarray,
) -> SignalEvaluation:
    """Evaluate a single signal against forward returns.

    Computes cross-sectional IC (Spearman rank correlation)
    for each time period, then summarises.

    Parameters
    ----------
    signal_id : str
        Signal identifier.
    signal_values : np.ndarray
        Shape (T, N) — signal values for T periods, N assets.
    forward_returns : np.ndarray
        Shape (T, N) — forward returns matching signal_values.

    Returns
    -------
    SignalEvaluation
    """
    try:
        T, N = signal_values.shape
        if T < 2 or N < 3:
            return SignalEvaluation(
                signal_id=signal_id, status="insufficient_data"
            )

        ics = []
        for t in range(T):
            sig = signal_values[t]
            ret = forward_returns[t]
            valid = np.isfinite(sig) & np.isfinite(ret)
            if valid.sum() < 3:
                continue
            corr, _ = scipy_stats.spearmanr(sig[valid], ret[valid])
            if np.isfinite(corr):
                ics.append(corr)

        if len(ics) < 2:
            return SignalEvaluation(
                signal_id=signal_id, status="insufficient_ic_data"
            )

        ics = np.array(ics)
        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics, ddof=1))
        ic_tstat = ic_mean / ic_std * np.sqrt(len(ics)) if ic_std > 0 else 0.0
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

        # Hit rate: fraction of periods with positive IC
        hit_rate = float(np.mean(ics > 0))

        # Turnover: average absolute change in signal ranks
        turnover = 0.0
        rank_changes = []
        for t in range(1, T):
            prev_ranks = scipy_stats.rankdata(signal_values[t - 1])
            curr_ranks = scipy_stats.rankdata(signal_values[t])
            rank_changes.append(np.mean(np.abs(curr_ranks - prev_ranks)) / N)
        if rank_changes:
            turnover = float(np.mean(rank_changes))

        # Rough Sharpe estimate: IC * sqrt(252) * info_ratio_scaling
        sharpe_est = ic_mean * np.sqrt(252) * 2.0

        return SignalEvaluation(
            signal_id=signal_id,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_tstat=ic_tstat,
            ic_ir=ic_ir,
            hit_rate=hit_rate,
            turnover=turnover,
            sharpe_estimate=sharpe_est,
            num_periods=len(ics),
        )
    except Exception as e:
        return SignalEvaluation(
            signal_id=signal_id, status=f"failed: {e}"
        )


class ParallelSignalEvaluator:
    """Evaluates candidate signals in parallel.

    Each signal is evaluated on the same out-of-sample test period
    using cross-sectional Spearman IC as the primary metric.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        import os as _os
        _default_workers = max(1, (_os.cpu_count() or 2) - 1)
        self._max_workers = cfg.get("max_workers", _default_workers)
        self._min_ic_tstat = cfg.get("min_ic_tstat", 1.5)

    def evaluate_signals(
        self,
        signals: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        parallel: bool = True,
    ) -> pd.DataFrame:
        """Evaluate multiple signals and return ranked results.

        Parameters
        ----------
        signals : dict
            Mapping signal_id → np.ndarray of shape (T, N).
        forward_returns : np.ndarray
            Shape (T, N) — same test period for all signals.
        parallel : bool
            Whether to run in parallel.

        Returns
        -------
        pd.DataFrame
            Ranked by IC t-statistic descending.
        """
        if not signals:
            return pd.DataFrame()

        evals: List[SignalEvaluation] = []

        if parallel and len(signals) > 1:
            evals = self._evaluate_parallel(signals, forward_returns)
        else:
            for sid, sv in signals.items():
                evals.append(
                    _evaluate_single_signal(sid, sv, forward_returns)
                )

        rows = []
        for ev in evals:
            rows.append({
                "signal_id": ev.signal_id,
                "ic_mean": ev.ic_mean,
                "ic_std": ev.ic_std,
                "ic_tstat": ev.ic_tstat,
                "ic_ir": ev.ic_ir,
                "hit_rate": ev.hit_rate,
                "turnover": ev.turnover,
                "sharpe_estimate": ev.sharpe_estimate,
                "num_periods": ev.num_periods,
                "status": ev.status,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ic_tstat", ascending=False).reset_index(drop=True)
        return df

    def filter_significant(
        self, results: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter to signals with IC t-stat above threshold."""
        if results.empty:
            return results
        mask = (results["ic_tstat"] >= self._min_ic_tstat) & (
            results["status"] == "completed"
        )
        return results[mask].reset_index(drop=True)

    def _evaluate_parallel(
        self,
        signals: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> List[SignalEvaluation]:
        """Run evaluations in parallel."""
        evals: List[SignalEvaluation] = []
        try:
            with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(
                        _evaluate_single_signal, sid, sv, forward_returns
                    ): sid
                    for sid, sv in signals.items()
                }
                for future in as_completed(futures):
                    evals.append(future.result())
        except Exception as e:
            logger.warning("Parallel evaluation failed, falling back: %s", e)
            for sid, sv in signals.items():
                evals.append(
                    _evaluate_single_signal(sid, sv, forward_returns)
                )
        return evals
