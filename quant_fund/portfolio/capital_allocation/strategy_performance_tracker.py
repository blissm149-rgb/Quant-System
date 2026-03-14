"""Strategy performance tracker for capital allocation decisions.

Tracks trailing Sharpe ratio, returns, and volatility for each active
strategy to inform dynamic capital allocation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy."""

    strategy_name: str
    trailing_sharpe: float
    trailing_return: float
    trailing_volatility: float
    n_days: int


class StrategyPerformanceTracker:
    """Tracks per-strategy daily returns for allocation decisions.

    Maintains a rolling history of daily returns per strategy and computes
    trailing Sharpe ratio used by dynamic_strategy_allocator.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._min_window = cfg.get("min_sharpe_window", 60)
        self._annualisation = np.sqrt(252)
        self._returns: Dict[str, List[float]] = {}
        self._dates: Dict[str, List[pd.Timestamp]] = {}

    def update(
        self, strategy_name: str, daily_return: float, date: pd.Timestamp
    ) -> None:
        """Record a daily return for a strategy."""
        if strategy_name not in self._returns:
            self._returns[strategy_name] = []
            self._dates[strategy_name] = []
        self._returns[strategy_name].append(daily_return)
        self._dates[strategy_name].append(date)

    def get_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """Get current performance metrics for a strategy."""
        if strategy_name not in self._returns:
            return None
        rets = np.array(self._returns[strategy_name])
        if len(rets) < self._min_window:
            return StrategyMetrics(
                strategy_name=strategy_name,
                trailing_sharpe=0.0,
                trailing_return=float(rets.mean()) * 252 if len(rets) > 0 else 0.0,
                trailing_volatility=float(rets.std()) * self._annualisation if len(rets) > 1 else 0.0,
                n_days=len(rets),
            )

        recent = rets[-self._min_window :]
        mean_ret = recent.mean()
        std_ret = recent.std()
        sharpe = (mean_ret / std_ret * self._annualisation) if std_ret > 0 else 0.0

        return StrategyMetrics(
            strategy_name=strategy_name,
            trailing_sharpe=sharpe,
            trailing_return=mean_ret * 252,
            trailing_volatility=std_ret * self._annualisation,
            n_days=len(rets),
        )

    def get_all_metrics(self) -> List[StrategyMetrics]:
        """Get metrics for all tracked strategies."""
        return [
            m for name in self._returns
            if (m := self.get_metrics(name)) is not None
        ]

    def get_returns_series(self, strategy_name: str) -> pd.Series:
        """Get full return series for a strategy."""
        if strategy_name not in self._returns:
            return pd.Series(dtype=float)
        return pd.Series(
            self._returns[strategy_name],
            index=pd.DatetimeIndex(self._dates[strategy_name]),
        )
