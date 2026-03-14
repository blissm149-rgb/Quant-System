"""Leverage controller enforcing gross exposure limits.

Scales down all positions proportionally if gross exposure would exceed
the configured limit. Applied as a post-optimisation adjustment before
order generation.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class LeverageController:
    """Enforces leverage <= max_leverage by proportional scaling.

    Applied after the portfolio optimizer and before order generation.
    If gross exposure exceeds the limit, all weights are scaled down
    proportionally.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._max_leverage = cfg.get("max_leverage", 2.0)

    def enforce(self, weights: pd.Series) -> pd.Series:
        """Enforce leverage constraint by scaling weights if necessary.

        Args:
            weights: Target portfolio weights indexed by ticker.

        Returns:
            Adjusted weights with gross exposure <= max_leverage.
        """
        gross = weights.abs().sum()
        if gross <= self._max_leverage or gross == 0:
            return weights.copy()

        scale_factor = self._max_leverage / gross
        adjusted = weights * scale_factor
        logger.info(
            "Leverage %.3f exceeded max %.3f. Scaled by %.4f",
            gross,
            self._max_leverage,
            scale_factor,
        )
        return adjusted

    @property
    def max_leverage(self) -> float:
        return self._max_leverage
