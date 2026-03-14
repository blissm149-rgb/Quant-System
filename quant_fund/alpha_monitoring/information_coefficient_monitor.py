"""Information coefficient monitor for live signal health.

Tracks IC on a daily basis and raises alerts when signal performance
degrades below configured thresholds.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class SignalDegradationAlert:
    """Alert raised when a signal's IC degrades."""

    signal_name: str
    alert_level: AlertLevel
    message: str
    rolling_ic: float
    ic_tstat: float
    date: pd.Timestamp


class InformationCoefficientMonitor:
    """Monitors daily IC for each active signal.

    Raises SignalDegradationAlert when:
    - Rolling 20-day IC falls below 0.0 (flat or reversed)
    - IC t-statistic falls below 1.5 over trailing 60 days
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._short_window = cfg.get("ic_short_window", 20)
        self._long_window = cfg.get("ic_long_window", 60)
        self._ic_floor = cfg.get("ic_floor", 0.0)
        self._tstat_floor = cfg.get("ic_tstat_floor", 1.5)
        self._ic_history: Dict[str, List[float]] = {}
        self._date_history: Dict[str, List[pd.Timestamp]] = {}

    def update(
        self,
        signal_name: str,
        signal_values: pd.Series,
        forward_returns: pd.Series,
        date: pd.Timestamp,
    ) -> List[SignalDegradationAlert]:
        """Record daily IC and check for degradation.

        Args:
            signal_name: Identifier for the signal.
            signal_values: Cross-sectional signal values indexed by ticker.
            forward_returns: Realised returns indexed by ticker.
            date: Date of this observation.

        Returns:
            List of alerts raised (may be empty).
        """
        common = signal_values.index.intersection(forward_returns.index)
        if len(common) < 5:
            return []

        ic = signal_values.loc[common].corr(
            forward_returns.loc[common], method="spearman"
        )
        if np.isnan(ic):
            return []

        if signal_name not in self._ic_history:
            self._ic_history[signal_name] = []
            self._date_history[signal_name] = []

        self._ic_history[signal_name].append(ic)
        self._date_history[signal_name].append(date)

        return self._check_alerts(signal_name, date)

    def _check_alerts(
        self, signal_name: str, date: pd.Timestamp
    ) -> List[SignalDegradationAlert]:
        """Check for IC degradation against thresholds."""
        alerts = []
        ic_vals = self._ic_history[signal_name]

        # Short-window rolling IC check
        if len(ic_vals) >= self._short_window:
            rolling_ic = np.mean(ic_vals[-self._short_window :])
            if rolling_ic < self._ic_floor:
                alerts.append(
                    SignalDegradationAlert(
                        signal_name=signal_name,
                        alert_level=AlertLevel.WARNING,
                        message=(
                            f"Rolling {self._short_window}-day IC = {rolling_ic:.4f} "
                            f"below floor {self._ic_floor}"
                        ),
                        rolling_ic=rolling_ic,
                        ic_tstat=0.0,
                        date=date,
                    )
                )

        # Long-window t-stat check
        if len(ic_vals) >= self._long_window:
            recent = ic_vals[-self._long_window :]
            ic_mean = np.mean(recent)
            ic_std = np.std(recent, ddof=1)
            tstat = (ic_mean / ic_std * np.sqrt(len(recent))) if ic_std > 0 else 0.0
            if tstat < self._tstat_floor:
                alerts.append(
                    SignalDegradationAlert(
                        signal_name=signal_name,
                        alert_level=AlertLevel.CRITICAL,
                        message=(
                            f"IC t-stat = {tstat:.2f} below threshold "
                            f"{self._tstat_floor} over {self._long_window} days"
                        ),
                        rolling_ic=ic_mean,
                        ic_tstat=tstat,
                        date=date,
                    )
                )

        return alerts

    def get_rolling_ic(
        self, signal_name: str, window: Optional[int] = None
    ) -> pd.Series:
        """Get the rolling IC series for a signal."""
        if signal_name not in self._ic_history:
            return pd.Series(dtype=float)

        window = window or self._short_window
        ic_series = pd.Series(
            self._ic_history[signal_name],
            index=pd.DatetimeIndex(self._date_history[signal_name]),
        )
        return ic_series.rolling(window).mean()

    def get_current_ic(self, signal_name: str) -> Optional[float]:
        """Get the most recent IC value for a signal."""
        if signal_name not in self._ic_history or not self._ic_history[signal_name]:
            return None
        return self._ic_history[signal_name][-1]
