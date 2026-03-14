"""Constraint engine — single place where all portfolio constraints live.

Accepts configuration and returns constraint objects for the portfolio
optimizer. No constraint logic should exist anywhere else in the codebase.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConstraintSet:
    """Complete set of portfolio constraints."""

    max_position_size: float = 0.02
    max_sector_exposure: float = 0.20
    max_leverage: float = 2.0
    dollar_neutral: bool = True
    min_position_size: float = 0.0
    max_turnover: Optional[float] = None
    sector_map: Dict[str, str] = field(default_factory=dict)


class ConstraintEngine:
    """Builds and validates portfolio constraints.

    This is the single place where all portfolio constraints are defined.
    The optimizer queries this engine for constraint parameters.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        position_limits = cfg.get("position_limits", {})
        self._max_position_size = position_limits.get("max_position_size", 0.02)
        self._max_sector_exposure = position_limits.get("max_sector_exposure", 0.20)
        self._max_leverage = position_limits.get("max_leverage", 2.0)
        self._dollar_neutral = position_limits.get("dollar_neutral", True)
        self._min_position_size = position_limits.get("min_position_size", 0.0)
        self._max_turnover = position_limits.get("max_turnover")

    def build_constraints(
        self, sector_map: Optional[Dict[str, str]] = None
    ) -> ConstraintSet:
        """Build the complete constraint set.

        Args:
            sector_map: Optional mapping of ticker -> GICS sector.

        Returns:
            ConstraintSet with all configured constraints.
        """
        return ConstraintSet(
            max_position_size=self._max_position_size,
            max_sector_exposure=self._max_sector_exposure,
            max_leverage=self._max_leverage,
            dollar_neutral=self._dollar_neutral,
            min_position_size=self._min_position_size,
            max_turnover=self._max_turnover,
            sector_map=sector_map or {},
        )

    def validate_weights(
        self,
        weights: pd.Series,
        constraints: ConstraintSet,
    ) -> List[str]:
        """Validate that weights satisfy all constraints.

        Args:
            weights: Portfolio weights indexed by ticker.
            constraints: The constraint set to check against.

        Returns:
            List of violation messages (empty if all satisfied).
        """
        violations = []

        # Position size check
        max_pos = weights.abs().max()
        if max_pos > constraints.max_position_size + 1e-6:
            worst = weights.abs().idxmax()
            violations.append(
                f"Position size violation: {worst} has |w|={max_pos:.4f} "
                f"> {constraints.max_position_size}"
            )

        # Leverage check
        gross = weights.abs().sum()
        if gross > constraints.max_leverage + 1e-6:
            violations.append(
                f"Leverage violation: gross={gross:.4f} > {constraints.max_leverage}"
            )

        # Dollar neutral check
        if constraints.dollar_neutral:
            net = weights.sum()
            if abs(net) > 0.01:
                violations.append(
                    f"Dollar neutral violation: net exposure={net:.4f}"
                )

        # Sector exposure check
        if constraints.sector_map:
            sector_exp: Dict[str, float] = {}
            for ticker, w in weights.items():
                sector = constraints.sector_map.get(ticker, "Unknown")
                sector_exp[sector] = sector_exp.get(sector, 0.0) + abs(w)
            for sector, exp in sector_exp.items():
                if exp > constraints.max_sector_exposure + 1e-6:
                    violations.append(
                        f"Sector exposure violation: {sector}={exp:.4f} "
                        f"> {constraints.max_sector_exposure}"
                    )

        return violations
