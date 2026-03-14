"""Macro regime classifier.

Classifies the current macro regime based on growth and inflation trajectories:
- risk_on_growth: expanding growth, falling inflation (goldilocks)
- stagflation: contracting growth, rising inflation
- goldilocks: expanding growth, stable inflation
- deflation: contracting growth, falling inflation
"""

from enum import Enum
from typing import Optional

import pandas as pd


class MacroRegime(Enum):
    GOLDILOCKS = "goldilocks"
    RISK_ON_GROWTH = "risk_on_growth"
    STAGFLATION = "stagflation"
    DEFLATION = "deflation"


class MacroRegimeClassifier:
    """Classifies the macro regime from growth and inflation signals.

    These regime labels gate strategy weights in dynamic_strategy_allocator.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._growth_threshold = cfg.get("growth_threshold", 0.0)
        self._inflation_threshold = cfg.get("inflation_threshold", 0.025)

    def classify(
        self,
        gdp_growth: Optional[float] = None,
        inflation_rate: Optional[float] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> MacroRegime:
        """Classify the current macro regime.

        Args:
            gdp_growth: Latest GDP growth rate (annualised).
            inflation_rate: Latest CPI year-over-year change.
            as_of: Point-in-time (for logging only).

        Returns:
            Current MacroRegime.
        """
        if gdp_growth is None and inflation_rate is None:
            return MacroRegime.GOLDILOCKS  # default neutral

        growth_expanding = (
            gdp_growth is not None and gdp_growth > self._growth_threshold
        )
        inflation_rising = (
            inflation_rate is not None and inflation_rate > self._inflation_threshold
        )

        if growth_expanding and not inflation_rising:
            return MacroRegime.GOLDILOCKS
        elif growth_expanding and inflation_rising:
            return MacroRegime.RISK_ON_GROWTH
        elif not growth_expanding and inflation_rising:
            return MacroRegime.STAGFLATION
        else:
            return MacroRegime.DEFLATION

    def get_regime_strategy_adjustments(self, regime: MacroRegime) -> dict:
        """Return strategy weight adjustments for a given macro regime."""
        adjustments = {
            MacroRegime.GOLDILOCKS: {
                "momentum": 1.2,
                "value": 1.0,
                "quality": 0.8,
                "low_volatility": 0.8,
                "mean_reversion": 1.0,
            },
            MacroRegime.RISK_ON_GROWTH: {
                "momentum": 1.0,
                "value": 0.8,
                "quality": 1.0,
                "low_volatility": 0.6,
                "mean_reversion": 0.8,
            },
            MacroRegime.STAGFLATION: {
                "momentum": 0.6,
                "value": 1.2,
                "quality": 1.4,
                "low_volatility": 1.2,
                "mean_reversion": 0.8,
            },
            MacroRegime.DEFLATION: {
                "momentum": 0.8,
                "value": 0.6,
                "quality": 1.4,
                "low_volatility": 1.4,
                "mean_reversion": 1.2,
            },
        }
        return adjustments.get(regime, {})
