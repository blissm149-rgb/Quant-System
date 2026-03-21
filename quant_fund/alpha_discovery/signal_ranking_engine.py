"""Signal ranking engine for scoring and combining alpha signals.

Scores candidate signals on out-of-sample IC, IC stability, turnover,
and correlation with existing live signals. Combines top signals into
a composite alpha score.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _benjamini_hochberg(p_values: list, fdr: float = 0.10) -> list:
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    thresholds = np.array([(i + 1) / m * fdr for i in range(m)])
    reject = sorted_p <= thresholds
    if not np.any(reject):
        return [False] * m
    max_k = np.max(np.where(reject))
    result = [False] * m
    for i in range(max_k + 1):
        result[sorted_indices[i]] = True
    return result


@dataclass
class SignalScore:
    """Evaluation score for a candidate signal."""

    signal_name: str
    ic_mean: float
    ic_tstat: float
    ic_std: float
    turnover: float
    correlation_with_existing: float
    composite_score: float
    passed: bool


class SignalRankingEngine:
    """Evaluates and ranks candidate alpha signals.

    Scoring criteria:
    - Out-of-sample IC (primary)
    - IC t-statistic (statistical significance)
    - IC stability (rolling IC standard deviation)
    - Turnover implied by the signal
    - Correlation with existing live signals

    Signals below IC threshold or t-stat threshold are rejected.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._min_ic = cfg.get("min_ic", 0.03)
        self._min_tstat = cfg.get("min_ic_tstat", 2.0)
        self._min_eval_days = cfg.get("min_eval_days", 252)
        self._ic_weight = cfg.get("ic_weight", 0.4)
        self._tstat_weight = cfg.get("tstat_weight", 0.3)
        self._stability_weight = cfg.get("stability_weight", 0.2)
        self._crowding_penalty = cfg.get("crowding_penalty", 0.1)
        self._turnover_penalty = cfg.get("turnover_penalty", 0.05)

    def evaluate_signal(
        self,
        signal_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        existing_signals: Optional[pd.DataFrame] = None,
    ) -> SignalScore:
        """Evaluate a single signal across multiple dates.

        Args:
            signal_values: DataFrame with columns ['date', 'ticker', 'signal']
                or MultiIndex (date, ticker) with signal column.
            forward_returns: DataFrame with same structure but 'return' column.
            existing_signals: DataFrame of existing live signal values
                for crowding check.

        Returns:
            SignalScore with evaluation metrics.
        """
        ic_series = self._compute_daily_ic(signal_values, forward_returns)

        if len(ic_series) < self._min_eval_days:
            return SignalScore(
                signal_name="",
                ic_mean=0.0,
                ic_tstat=0.0,
                ic_std=0.0,
                turnover=0.0,
                correlation_with_existing=0.0,
                composite_score=-999.0,
                passed=False,
            )

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_tstat = (ic_mean / ic_std * np.sqrt(len(ic_series))) if ic_std > 0 else 0.0

        turnover = self._compute_turnover(signal_values)
        crowding_corr = self._compute_crowding(signal_values, existing_signals)

        composite = (
            self._ic_weight * ic_mean
            + self._tstat_weight * (ic_tstat / 10.0)
            + self._stability_weight * (1.0 / (1.0 + ic_std))
            - self._turnover_penalty * turnover
            - self._crowding_penalty * crowding_corr
        )

        passed = ic_mean >= self._min_ic and ic_tstat >= self._min_tstat

        return SignalScore(
            signal_name="",
            ic_mean=ic_mean,
            ic_tstat=ic_tstat,
            ic_std=ic_std,
            turnover=turnover,
            correlation_with_existing=crowding_corr,
            composite_score=composite,
            passed=passed,
        )

    def rank_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        existing_signals: Optional[pd.DataFrame] = None,
    ) -> List[SignalScore]:
        """Evaluate and rank multiple candidate signals.

        Args:
            signals: Dict mapping signal names to signal DataFrames.
            forward_returns: Forward returns for IC computation.
            existing_signals: Existing live signals for crowding check.

        Returns:
            List of SignalScore sorted by composite_score descending.
        """
        scores = []
        for name, sig_df in signals.items():
            score = self.evaluate_signal(sig_df, forward_returns, existing_signals)
            score.signal_name = name
            scores.append(score)

        scores.sort(key=lambda s: s.composite_score, reverse=True)
        return scores

    def combine(
        self,
        signal_scores: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Combine multiple signals into a composite alpha score.

        Args:
            signal_scores: Dict mapping signal names to cross-sectional
                signal Series indexed by ticker.
            weights: Optional signal weights. If None, uses equal weights.

        Returns:
            Combined alpha Series indexed by ticker.
        """
        if not signal_scores:
            return pd.Series(dtype=float)

        if weights is None:
            weights = {name: 1.0 / len(signal_scores) for name in signal_scores}

        total_weight = sum(weights.get(name, 0.0) for name in signal_scores)
        if total_weight == 0:
            return pd.Series(dtype=float)

        all_tickers = set()
        for series in signal_scores.values():
            all_tickers.update(series.index)
        all_tickers = sorted(all_tickers)

        combined = pd.Series(0.0, index=all_tickers)
        for name, series in signal_scores.items():
            w = weights.get(name, 0.0) / total_weight
            aligned = series.reindex(all_tickers, fill_value=0.0)
            combined += w * aligned

        return combined

    def _compute_daily_ic(
        self, signal_values: pd.DataFrame, forward_returns: pd.DataFrame
    ) -> pd.Series:
        """Compute daily rank IC between signal and forward returns."""
        sig = self._to_multiindex(signal_values, "signal")
        ret = self._to_multiindex(forward_returns, "return")

        if sig is None or ret is None:
            return pd.Series(dtype=float)

        merged = sig.join(ret, how="inner")
        if merged.empty:
            return pd.Series(dtype=float)

        dates = merged.index.get_level_values(0).unique()
        ics = {}
        for dt in dates:
            try:
                day_data = merged.loc[dt]
                if len(day_data) < 5:
                    continue
                ic = day_data["signal"].corr(day_data["return"], method="spearman")
                if not np.isnan(ic):
                    ics[dt] = ic
            except Exception:
                continue
        return pd.Series(ics)

    def _to_multiindex(self, df: pd.DataFrame, value_col: str):
        """Ensure DataFrame has MultiIndex (date, ticker) with value_col."""
        if isinstance(df.index, pd.MultiIndex):
            if value_col in df.columns:
                return df[[value_col]]
            if len(df.columns) == 1:
                return df.rename(columns={df.columns[0]: value_col})
            return None
        if "date" in df.columns and "ticker" in df.columns and value_col in df.columns:
            return df.set_index(["date", "ticker"])[[value_col]]
        return None

    def _compute_turnover(self, signal_values: pd.DataFrame) -> float:
        """Estimate signal turnover as mean absolute daily change."""
        sig = self._to_multiindex(signal_values, "signal")
        if sig is None:
            return 0.0
        dates = sig.index.get_level_values(0).unique().sort_values()
        if len(dates) < 2:
            return 0.0

        turnovers = []
        for i in range(1, len(dates)):
            try:
                prev = sig.loc[dates[i - 1]]["signal"]
                curr = sig.loc[dates[i]]["signal"]
                common = prev.index.intersection(curr.index)
                if len(common) > 0:
                    turnover = (curr.loc[common] - prev.loc[common]).abs().mean()
                    turnovers.append(turnover)
            except Exception:
                continue
        return float(np.mean(turnovers)) if turnovers else 0.0

    def _compute_crowding(
        self,
        signal_values: pd.DataFrame,
        existing_signals: Optional[pd.DataFrame],
    ) -> float:
        """Compute max correlation with existing signals."""
        if existing_signals is None or existing_signals.empty:
            return 0.0

        sig = self._to_multiindex(signal_values, "signal")
        if sig is None:
            return 0.0

        max_corr = 0.0
        for col in existing_signals.columns:
            if col in ("date", "ticker"):
                continue
            existing = self._to_multiindex(existing_signals, col)
            if existing is None:
                continue
            merged = sig.join(existing, how="inner")
            if len(merged) > 10:
                corr = abs(merged.iloc[:, 0].corr(merged.iloc[:, 1], method="spearman"))
                max_corr = max(max_corr, corr if not np.isnan(corr) else 0.0)
        return max_corr

    def evaluate_signal_with_correction(
        self,
        signals: Dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        fdr: float = 0.10,
        existing_signals: Optional[pd.DataFrame] = None,
    ) -> List[SignalScore]:
        """Evaluate multiple signals with Benjamini-Hochberg FDR correction.

        Tests each signal's IC against zero, then applies BH correction to
        control the false discovery rate across all tests.

        Args:
            signals: Dict mapping signal names to signal DataFrames.
            forward_returns: Forward returns for IC computation.
            fdr: Target false discovery rate (default 0.10).
            existing_signals: Existing live signals for crowding check.

        Returns:
            List of SignalScore with `passed` reflecting BH correction.
        """
        from scipy import stats as scipy_stats

        scores = []
        p_values = []
        for name, sig_df in signals.items():
            score = self.evaluate_signal(sig_df, forward_returns, existing_signals)
            score.signal_name = name
            scores.append(score)

            # Two-sided p-value from IC t-statistic
            if score.ic_tstat != 0 and not np.isnan(score.ic_tstat):
                ic_series = self._compute_daily_ic(sig_df, forward_returns)
                n = len(ic_series)
                if n > 1:
                    p = 2 * (1 - scipy_stats.t.cdf(abs(score.ic_tstat), df=n - 1))
                else:
                    p = 1.0
            else:
                p = 1.0
            p_values.append(p)

        # Apply BH correction
        significant = _benjamini_hochberg(p_values, fdr=fdr)
        for score, is_sig in zip(scores, significant):
            score.passed = is_sig

        scores.sort(key=lambda s: s.composite_score, reverse=True)
        return scores

    def bootstrap_ic_ci(
        self,
        signal_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> tuple:
        """Bootstrap confidence interval for mean IC.

        Args:
            signal_values: Signal DataFrame.
            forward_returns: Forward returns DataFrame.
            n_bootstrap: Number of bootstrap samples.
            ci: Confidence level.
            seed: Random seed.

        Returns:
            (lower, upper) confidence interval bounds.
        """
        ic_series = self._compute_daily_ic(signal_values, forward_returns)
        if len(ic_series) < 10:
            return (0.0, 0.0)

        rng = np.random.default_rng(seed)
        n = len(ic_series)
        boot_means = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_means.append(ic_series.iloc[idx].mean())

        boot_means.sort()
        alpha = (1 - ci) / 2
        lower = boot_means[int(alpha * len(boot_means))]
        upper = boot_means[int((1 - alpha) * len(boot_means))]
        return (float(lower), float(upper))
