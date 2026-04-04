"""P&L dashboard — real-time profit and loss tracking.

Tracks total NAV, daily returns, P&L attribution by strategy
and sector, realised vs unrealised P&L, and drawdown from peak.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PnLSnapshot:
    """Point-in-time P&L snapshot."""

    timestamp: pd.Timestamp
    nav: float
    daily_return: float
    daily_pnl: float
    cumulative_return: float
    drawdown: float
    peak_nav: float
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0
    strategy_pnl: Dict[str, float] = field(default_factory=dict)
    sector_pnl: Dict[str, float] = field(default_factory=dict)


class PnLDashboard:
    """Real-time P&L tracking and attribution.

    Features:
    - Total NAV and daily return
    - P&L attribution by strategy and sector
    - Realised vs unrealised P&L
    - Drawdown from peak NAV
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._initial_nav = cfg.get("initial_nav", 1_000_000.0)
        self._history: deque = deque(maxlen=50_000)
        self._peak_nav = self._initial_nav
        self._prev_nav = self._initial_nav
        self._cumulative_realised = 0.0

    def update(
        self,
        nav: float,
        timestamp: Optional[pd.Timestamp] = None,
        strategy_pnl: Optional[Dict[str, float]] = None,
        sector_pnl: Optional[Dict[str, float]] = None,
        realised_pnl_today: float = 0.0,
    ) -> PnLSnapshot:
        """Record a new P&L observation.

        Parameters
        ----------
        nav : float
            Current net asset value.
        timestamp : pd.Timestamp, optional
            Observation time. Defaults to now.
        strategy_pnl : dict, optional
            P&L by strategy_id.
        sector_pnl : dict, optional
            P&L by sector.
        realised_pnl_today : float
            Realised P&L from today's fills.

        Returns
        -------
        PnLSnapshot
        """
        ts = timestamp or pd.Timestamp.now()

        daily_pnl = nav - self._prev_nav
        daily_return = daily_pnl / self._prev_nav if self._prev_nav > 0 else 0.0
        cumulative_return = (nav / self._initial_nav - 1.0) if self._initial_nav > 0 else 0.0

        self._peak_nav = max(self._peak_nav, nav)
        drawdown = (self._peak_nav - nav) / self._peak_nav if self._peak_nav > 0 else 0.0

        self._cumulative_realised += realised_pnl_today
        unrealised = (nav - self._initial_nav) - self._cumulative_realised

        snapshot = PnLSnapshot(
            timestamp=ts,
            nav=nav,
            daily_return=daily_return,
            daily_pnl=daily_pnl,
            cumulative_return=cumulative_return,
            drawdown=drawdown,
            peak_nav=self._peak_nav,
            realised_pnl=self._cumulative_realised,
            unrealised_pnl=unrealised,
            strategy_pnl=strategy_pnl or {},
            sector_pnl=sector_pnl or {},
        )

        self._history.append(snapshot)
        self._prev_nav = nav
        return snapshot

    @property
    def latest(self) -> Optional[PnLSnapshot]:
        return self._history[-1] if self._history else None

    @property
    def current_drawdown(self) -> float:
        if not self._history:
            return 0.0
        return self._history[-1].drawdown

    def get_return_series(self) -> pd.Series:
        """Get daily returns as a time series."""
        if not self._history:
            return pd.Series(dtype=float)
        data = {s.timestamp: s.daily_return for s in self._history}
        return pd.Series(data)

    def get_nav_series(self) -> pd.Series:
        """Get NAV time series."""
        if not self._history:
            return pd.Series(dtype=float)
        data = {s.timestamp: s.nav for s in self._history}
        return pd.Series(data)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        returns = self.get_return_series()
        if len(returns) == 0:
            return {}
        return {
            "total_return": float((returns + 1).prod() - 1),
            "annualised_return": float(np.mean(returns) * 252),
            "annualised_vol": float(np.std(returns, ddof=1) * np.sqrt(252)),
            "sharpe_ratio": float(
                np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
            ) if np.std(returns) > 0 else 0.0,
            "max_drawdown": float(max(s.drawdown for s in self._history)),
            "current_nav": self._history[-1].nav,
            "peak_nav": self._peak_nav,
            "num_days": len(self._history),
        }
