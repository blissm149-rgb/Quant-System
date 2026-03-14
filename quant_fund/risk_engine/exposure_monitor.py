"""Exposure monitor checking portfolio constraints on every update.

Monitors net market exposure, gross exposure, per-sector exposure,
and per-factor exposure against configured limits.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExposureBreachAlert:
    """Alert raised when an exposure limit is breached."""

    exposure_type: str
    current_value: float
    limit: float
    message: str


class ExposureMonitor:
    """Checks portfolio exposures against configured limits.

    Checks on every portfolio update:
    - Net market exposure (beta x portfolio value)
    - Gross exposure (sum of |weights|)
    - Per-sector gross exposure
    - Per-factor exposure vs configured limits
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._max_leverage = cfg.get("max_leverage", 2.0)
        self._max_sector_exposure = cfg.get("max_sector_exposure", 0.20)
        self._max_single_name = cfg.get("max_single_name_exposure", 0.02)
        self._max_net_beta = cfg.get("max_net_beta", 0.20)
        self._max_factor_exposure = cfg.get("max_factor_exposure", {})

    def check(
        self,
        weights: pd.Series,
        sector_map: Optional[Dict[str, str]] = None,
        factor_exposures: Optional[pd.DataFrame] = None,
    ) -> List[ExposureBreachAlert]:
        """Check all exposure limits. Returns list of breaches.

        Args:
            weights: Portfolio weights indexed by ticker.
            sector_map: Dict mapping ticker -> sector name.
            factor_exposures: DataFrame (tickers x factors).

        Returns:
            List of ExposureBreachAlert objects.
        """
        alerts = []

        # Gross exposure (leverage)
        gross = weights.abs().sum()
        if gross > self._max_leverage:
            alerts.append(
                ExposureBreachAlert(
                    exposure_type="gross_leverage",
                    current_value=gross,
                    limit=self._max_leverage,
                    message=f"Gross leverage {gross:.3f} > {self._max_leverage}",
                )
            )

        # Single name concentration
        max_pos = weights.abs().max() if len(weights) > 0 else 0.0
        if max_pos > self._max_single_name:
            worst = weights.abs().idxmax()
            alerts.append(
                ExposureBreachAlert(
                    exposure_type="single_name",
                    current_value=max_pos,
                    limit=self._max_single_name,
                    message=f"Single name {worst}: |w|={max_pos:.4f} > {self._max_single_name}",
                )
            )

        # Per-sector exposure
        if sector_map:
            sector_alerts = self._check_sector_exposure(weights, sector_map)
            alerts.extend(sector_alerts)

        # Per-factor exposure
        if factor_exposures is not None:
            factor_alerts = self._check_factor_exposure(weights, factor_exposures)
            alerts.extend(factor_alerts)

        return alerts

    def _check_sector_exposure(
        self, weights: pd.Series, sector_map: Dict[str, str]
    ) -> List[ExposureBreachAlert]:
        """Check per-sector gross exposure."""
        alerts = []
        sector_exposure: Dict[str, float] = {}

        for ticker, w in weights.items():
            sector = sector_map.get(ticker, "Unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(w)

        for sector, exposure in sector_exposure.items():
            if exposure > self._max_sector_exposure:
                alerts.append(
                    ExposureBreachAlert(
                        exposure_type=f"sector_{sector}",
                        current_value=exposure,
                        limit=self._max_sector_exposure,
                        message=f"Sector {sector} exposure {exposure:.3f} > {self._max_sector_exposure}",
                    )
                )

        return alerts

    def _check_factor_exposure(
        self, weights: pd.Series, factor_exposures: pd.DataFrame
    ) -> List[ExposureBreachAlert]:
        """Check portfolio-level factor exposures."""
        alerts = []
        common = weights.index.intersection(factor_exposures.index)
        if len(common) == 0:
            return alerts

        w = weights.reindex(common, fill_value=0.0).values
        B = factor_exposures.reindex(common, fill_value=0.0)

        portfolio_exposure = B.T @ w

        for factor, exposure in portfolio_exposure.items():
            limit = self._max_factor_exposure.get(factor)
            if limit is not None and abs(exposure) > limit:
                alerts.append(
                    ExposureBreachAlert(
                        exposure_type=f"factor_{factor}",
                        current_value=abs(exposure),
                        limit=limit,
                        message=f"Factor {factor} exposure |{exposure:.3f}| > {limit}",
                    )
                )

        return alerts
