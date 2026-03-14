"""Dynamic strategy allocator for capital allocation across strategies.

Allocates capital using Sharpe-weighted allocation, penalised for
correlation and adjusted by regime overlay.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DynamicStrategyAllocator:
    """Allocates capital across active strategies.

    Algorithm:
    1. Compute trailing Sharpe for each strategy
    2. Penalise strategies with high pairwise correlation
    3. Apply regime overlay
    4. Normalise allocations to sum to 1.0
    5. Apply min/max allocation bounds
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._min_allocation = cfg.get("min_strategy_allocation", 0.05)
        self._max_allocation = cfg.get("max_strategy_allocation", 0.40)
        self._correlation_penalty_weight = cfg.get("correlation_penalty_weight", 0.3)
        self._min_sharpe_for_allocation = cfg.get("min_sharpe_for_allocation", 0.0)

    def allocate(
        self,
        strategy_sharpes: Dict[str, float],
        correlation_penalties: Optional[pd.Series] = None,
        regime_adjustments: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute capital allocations across strategies.

        Args:
            strategy_sharpes: Dict mapping strategy_name -> trailing Sharpe ratio.
            correlation_penalties: Optional penalty per strategy (0-1).
            regime_adjustments: Optional multiplier per strategy (0-2)
                based on current market regime.

        Returns:
            Dict mapping strategy_name -> allocation weight (sums to 1.0).
        """
        if not strategy_sharpes:
            return {}

        # Filter out strategies below minimum Sharpe
        eligible = {
            name: max(sharpe, 0.0)
            for name, sharpe in strategy_sharpes.items()
            if sharpe >= self._min_sharpe_for_allocation
        }

        if not eligible:
            # Equal weight among all if none meet threshold
            n = len(strategy_sharpes)
            return {name: 1.0 / n for name in strategy_sharpes}

        # Start with Sharpe-proportional weights
        raw_weights = dict(eligible)

        # Apply correlation penalty
        if correlation_penalties is not None:
            for name in raw_weights:
                penalty = correlation_penalties.get(name, 0.0)
                raw_weights[name] *= (1.0 - self._correlation_penalty_weight * penalty)

        # Apply regime adjustments
        if regime_adjustments is not None:
            for name in raw_weights:
                adj = regime_adjustments.get(name, 1.0)
                raw_weights[name] *= adj

        # Ensure non-negative
        raw_weights = {k: max(v, 0.0) for k, v in raw_weights.items()}

        # Normalise
        total = sum(raw_weights.values())
        if total <= 0:
            n = len(strategy_sharpes)
            return {name: 1.0 / n for name in strategy_sharpes}

        allocations = {k: v / total for k, v in raw_weights.items()}

        # Apply min/max bounds and re-normalise
        allocations = self._apply_bounds(allocations)

        # Add zero allocation for ineligible strategies
        for name in strategy_sharpes:
            if name not in allocations:
                allocations[name] = 0.0

        return allocations

    def _apply_bounds(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max allocation bounds and re-normalise."""
        bounded = {}
        for name, alloc in allocations.items():
            if alloc < self._min_allocation:
                bounded[name] = 0.0  # Below floor -> zero out
            else:
                bounded[name] = min(alloc, self._max_allocation)

        total = sum(bounded.values())
        if total <= 0:
            # All got zeroed out, fall back to equal weight
            n = len(allocations)
            return {name: 1.0 / n for name in allocations}

        return {k: v / total for k, v in bounded.items()}
