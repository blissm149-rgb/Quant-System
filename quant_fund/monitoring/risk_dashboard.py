"""Risk dashboard — real-time risk metrics display.

Shows current leverage, sector/factor exposures vs limits,
VaR estimates, and top risk contributors.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskSnapshot:
    """Point-in-time risk metrics."""

    timestamp: pd.Timestamp
    gross_exposure: float
    net_exposure: float
    leverage: float
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    var_95: float = 0.0
    var_99: float = 0.0
    top_contributors: List[dict] = field(default_factory=list)


class RiskDashboard:
    """Real-time risk metrics tracking.

    Features:
    - Current leverage and gross/net exposure
    - Per-sector exposure vs limits
    - Per-factor exposure vs limits
    - VaR (historical simulation, 95% and 99%)
    - Top 10 risk contributors
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._max_leverage = cfg.get("max_leverage", 2.0)
        self._max_sector_exposure = cfg.get("max_sector_exposure", 0.20)
        self._var_lookback = cfg.get("var_lookback_days", 252)
        self._history: List[RiskSnapshot] = []

    def compute_snapshot(
        self,
        weights: pd.Series,
        sector_map: Optional[Dict[str, str]] = None,
        factor_exposures: Optional[pd.DataFrame] = None,
        return_history: Optional[pd.DataFrame] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> RiskSnapshot:
        """Compute current risk snapshot.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.
        sector_map : dict, optional
            Mapping ticker → sector.
        factor_exposures : pd.DataFrame, optional
            Factor exposures indexed by ticker, columns = factors.
        return_history : pd.DataFrame, optional
            Historical returns (rows=dates, columns=tickers) for VaR.
        timestamp : pd.Timestamp, optional

        Returns
        -------
        RiskSnapshot
        """
        ts = timestamp or pd.Timestamp.now()

        gross = float(np.abs(weights).sum())
        net = float(weights.sum())
        leverage = gross

        # Sector exposures
        sector_exp = {}
        if sector_map:
            for ticker, w in weights.items():
                sector = sector_map.get(str(ticker), "Unknown")
                sector_exp[sector] = sector_exp.get(sector, 0.0) + w

        # Factor exposures (portfolio-level)
        factor_exp = {}
        if factor_exposures is not None and not factor_exposures.empty:
            common = weights.index.intersection(factor_exposures.index)
            if len(common) > 0:
                w_aligned = weights.loc[common].values
                for col in factor_exposures.columns:
                    exp = float(w_aligned @ factor_exposures.loc[common, col].values)
                    factor_exp[col] = exp

        # VaR
        var_95, var_99 = 0.0, 0.0
        if return_history is not None and not return_history.empty:
            var_95, var_99 = self._compute_var(weights, return_history)

        # Top risk contributors (by |weight|)
        top = weights.abs().nlargest(10)
        top_contributors = [
            {"ticker": str(t), "weight": float(weights[t]), "abs_weight": float(w)}
            for t, w in top.items()
        ]

        snapshot = RiskSnapshot(
            timestamp=ts,
            gross_exposure=gross,
            net_exposure=net,
            leverage=leverage,
            sector_exposures=sector_exp,
            factor_exposures=factor_exp,
            var_95=var_95,
            var_99=var_99,
            top_contributors=top_contributors,
        )
        self._history.append(snapshot)
        return snapshot

    def check_limits(
        self, snapshot: RiskSnapshot
    ) -> List[str]:
        """Check risk limits and return list of breaches."""
        breaches = []
        if snapshot.leverage > self._max_leverage:
            breaches.append(
                f"Leverage {snapshot.leverage:.2f} > limit {self._max_leverage}"
            )
        for sector, exp in snapshot.sector_exposures.items():
            if abs(exp) > self._max_sector_exposure:
                breaches.append(
                    f"Sector {sector} exposure {exp:.3f} > limit {self._max_sector_exposure}"
                )
        return breaches

    @property
    def latest(self) -> Optional[RiskSnapshot]:
        return self._history[-1] if self._history else None

    def _compute_var(
        self,
        weights: pd.Series,
        return_history: pd.DataFrame,
    ) -> tuple:
        """Compute VaR via historical simulation."""
        common = weights.index.intersection(return_history.columns)
        if len(common) == 0:
            return 0.0, 0.0
        w = weights.loc[common].values
        R = return_history[common].dropna().values
        if len(R) == 0:
            return 0.0, 0.0
        portfolio_returns = R @ w
        var_95 = float(-np.percentile(portfolio_returns, 5))
        var_99 = float(-np.percentile(portfolio_returns, 1))
        return var_95, var_99
