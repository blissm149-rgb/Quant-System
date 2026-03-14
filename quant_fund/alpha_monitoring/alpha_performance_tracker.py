"""Alpha performance tracker for rolling per-signal metrics.

Tracks realised IC, information ratio, and cumulative PnL attribution
from each signal on a rolling basis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalPerformance:
    """Performance summary for a single signal."""

    signal_name: str
    realised_ic: float
    ic_ir: float  # annualised IC information ratio (IC_mean / IC_std * sqrt(252))
    cumulative_pnl: float
    n_days: int
    rolling_ic_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


class AlphaPerformanceTracker:
    """Tracks per-signal realised performance metrics on a rolling basis.

    Maintains a history of daily ICs and PnL contributions for each signal.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._rolling_window = cfg.get("rolling_window", 60)
        self._annualisation_factor = np.sqrt(252)
        self._signal_history: Dict[str, Dict] = {}

    def update(
        self,
        signal_name: str,
        signal_values: pd.Series,
        forward_returns: pd.Series,
        date: pd.Timestamp,
    ) -> None:
        """Record a daily observation for a signal.

        Args:
            signal_name: Identifier for the signal.
            signal_values: Cross-sectional signal values indexed by ticker.
            forward_returns: Realised returns indexed by ticker.
            date: Date of this observation.
        """
        if signal_name not in self._signal_history:
            self._signal_history[signal_name] = {
                "ic_values": [],
                "dates": [],
                "pnl_contributions": [],
            }

        common = signal_values.index.intersection(forward_returns.index)
        if len(common) < 5:
            return

        sig = signal_values.loc[common]
        ret = forward_returns.loc[common]

        ic = sig.corr(ret, method="spearman")
        if np.isnan(ic):
            return

        # PnL contribution: signal-weighted return
        weights = sig / sig.abs().sum() if sig.abs().sum() > 0 else sig * 0
        pnl = (weights * ret).sum()

        history = self._signal_history[signal_name]
        history["ic_values"].append(ic)
        history["dates"].append(date)
        history["pnl_contributions"].append(pnl)

    def get_performance(self, signal_name: str) -> Optional[SignalPerformance]:
        """Get current performance summary for a signal."""
        if signal_name not in self._signal_history:
            return None

        history = self._signal_history[signal_name]
        if not history["ic_values"]:
            return None

        ic_series = pd.Series(
            history["ic_values"], index=pd.DatetimeIndex(history["dates"])
        )
        pnl_series = pd.Series(
            history["pnl_contributions"], index=pd.DatetimeIndex(history["dates"])
        )

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = (ic_mean / ic_std * self._annualisation_factor) if ic_std > 0 else 0.0

        return SignalPerformance(
            signal_name=signal_name,
            realised_ic=ic_mean,
            ic_ir=ic_ir,
            cumulative_pnl=pnl_series.sum(),
            n_days=len(ic_series),
            rolling_ic_series=ic_series.rolling(self._rolling_window).mean(),
        )

    def get_all_performances(self) -> List[SignalPerformance]:
        """Get performance summaries for all tracked signals."""
        return [
            perf
            for name in self._signal_history
            if (perf := self.get_performance(name)) is not None
        ]

    def get_signal_names(self) -> List[str]:
        """Return names of all tracked signals."""
        return list(self._signal_history.keys())
