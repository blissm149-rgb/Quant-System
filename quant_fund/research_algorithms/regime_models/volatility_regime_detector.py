"""Volatility regime detector.

Classifies current volatility environment as low, normal, or high
based on realised volatility percentile relative to trailing history.
"""

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class VolatilityRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class VolatilityRegimeDetector:
    """Classifies current volatility regime using realised vol percentiles.

    Low vol regime: factor strategies favoured.
    High vol regime: defensive tilt.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._vol_window = cfg.get("vol_regime_window", 20)
        self._lookback_days = cfg.get("vol_regime_lookback", 252)
        self._low_percentile = cfg.get("vol_regime_low_pct", 25)
        self._high_percentile = cfg.get("vol_regime_high_pct", 75)

    def detect(self, market_returns: pd.Series) -> VolatilityRegime:
        """Classify the current volatility regime.

        Args:
            market_returns: Daily market returns series.

        Returns:
            Current VolatilityRegime classification.
        """
        if len(market_returns) < self._vol_window:
            return VolatilityRegime.NORMAL

        returns = market_returns.dropna()
        rolling_vol = returns.rolling(self._vol_window).std().dropna()

        if len(rolling_vol) < 2:
            return VolatilityRegime.NORMAL

        current_vol = rolling_vol.iloc[-1]
        lookback = rolling_vol.iloc[-self._lookback_days:]

        percentile = (lookback < current_vol).mean() * 100

        if percentile <= self._low_percentile:
            return VolatilityRegime.LOW
        elif percentile >= self._high_percentile:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.NORMAL

    def get_vol_percentile(self, market_returns: pd.Series) -> float:
        """Return the current volatility percentile (0-100)."""
        returns = market_returns.dropna()
        rolling_vol = returns.rolling(self._vol_window).std().dropna()

        if len(rolling_vol) < 2:
            return 50.0

        current_vol = rolling_vol.iloc[-1]
        lookback = rolling_vol.iloc[-self._lookback_days:]
        return float((lookback < current_vol).mean() * 100)
