"""Signal decay detector measuring IC half-life.

Computes the time for IC to decay to half its initial value.
Flags signals for review or retirement based on decay speed.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DecayAction(str, Enum):
    NONE = "none"
    REVIEW = "review"
    RETIRE = "retire"


@dataclass
class DecayResult:
    """Result of IC decay analysis for a signal."""

    signal_name: str
    ic_half_life_days: float
    current_ic: float
    peak_ic: float
    action: DecayAction
    message: str


class SignalDecayDetector:
    """Detects signal decay by analysing IC half-life.

    IC half-life < 20 trading days -> review flag
    IC half-life < 10 trading days -> retirement recommendation
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._review_threshold = cfg.get("decay_review_threshold", 20)
        self._retire_threshold = cfg.get("decay_retire_threshold", 10)
        self._min_history = cfg.get("decay_min_history", 60)
        self._estimation_window = cfg.get("decay_estimation_window", 120)
        self._ic_history: Dict[str, List[float]] = {}
        self._date_history: Dict[str, List[pd.Timestamp]] = {}

    def update(
        self, signal_name: str, ic_value: float, date: pd.Timestamp
    ) -> None:
        """Record a daily IC observation."""
        if signal_name not in self._ic_history:
            self._ic_history[signal_name] = []
            self._date_history[signal_name] = []
        self._ic_history[signal_name].append(ic_value)
        self._date_history[signal_name].append(date)

    def detect(self, signal_name: str) -> Optional[DecayResult]:
        """Analyse IC decay for a signal.

        Returns:
            DecayResult if sufficient history, None otherwise.
        """
        if signal_name not in self._ic_history:
            return None

        ic_vals = self._ic_history[signal_name]
        if len(ic_vals) < self._min_history:
            return None

        recent = ic_vals[-self._estimation_window :]
        half_life = self._estimate_half_life(recent)

        peak_ic = max(ic_vals)
        current_ic = np.mean(ic_vals[-20:]) if len(ic_vals) >= 20 else ic_vals[-1]

        if half_life < self._retire_threshold:
            action = DecayAction.RETIRE
            message = (
                f"IC half-life = {half_life:.1f} days < {self._retire_threshold}: "
                f"recommend retirement"
            )
        elif half_life < self._review_threshold:
            action = DecayAction.REVIEW
            message = (
                f"IC half-life = {half_life:.1f} days < {self._review_threshold}: "
                f"flagged for review"
            )
        else:
            action = DecayAction.NONE
            message = f"IC half-life = {half_life:.1f} days: healthy"

        return DecayResult(
            signal_name=signal_name,
            ic_half_life_days=half_life,
            current_ic=current_ic,
            peak_ic=peak_ic,
            action=action,
            message=message,
        )

    def detect_all(self) -> List[DecayResult]:
        """Run decay detection on all tracked signals."""
        results = []
        for name in self._ic_history:
            result = self.detect(name)
            if result is not None:
                results.append(result)
        return results

    def _estimate_half_life(self, ic_series: List[float]) -> float:
        """Estimate IC half-life using AR(1) model.

        Half-life = -log(2) / log(phi) where phi is the AR(1) coefficient.
        """
        if len(ic_series) < 10:
            return float("inf")

        y = np.array(ic_series)
        y_lag = y[:-1]
        y_curr = y[1:]

        valid = np.isfinite(y_lag) & np.isfinite(y_curr)
        y_lag = y_lag[valid]
        y_curr = y_curr[valid]

        if len(y_lag) < 10:
            return float("inf")

        # OLS: y_curr = phi * y_lag + epsilon
        mean_lag = y_lag.mean()
        mean_curr = y_curr.mean()
        cov = ((y_lag - mean_lag) * (y_curr - mean_curr)).mean()
        var = ((y_lag - mean_lag) ** 2).mean()

        if var < 1e-10:
            return float("inf")

        phi = cov / var

        if phi <= 0 or phi >= 1:
            return float("inf")

        half_life = -np.log(2) / np.log(phi)
        return max(0.0, half_life)
