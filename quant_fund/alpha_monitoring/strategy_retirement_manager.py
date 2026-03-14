"""Strategy retirement manager for signal lifecycle management.

Manages the lifecycle of live signals through review, suspension, and
retirement stages based on IC degradation and decay metrics.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class StrategyStatus(str, Enum):
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    SUSPENDED = "suspended"
    RETIRED = "retired"


@dataclass
class StrategyState:
    """Current state of a strategy in the lifecycle."""

    strategy_name: str
    status: StrategyStatus
    allocation_multiplier: float  # 1.0 = full allocation, 0.0 = no allocation
    status_since: pd.Timestamp
    review_reason: str = ""
    history: List[str] = field(default_factory=list)


class StrategyRetirementManager:
    """Manages the lifecycle of live trading signals.

    Lifecycle transitions:
    - ACTIVE -> UNDER_REVIEW: IC degraded, reduce allocation, notify
    - UNDER_REVIEW -> SUSPENDED: IC persistently negative, reduce to zero
    - SUSPENDED -> RETIRED: manual sign-off required
    - UNDER_REVIEW -> ACTIVE: IC recovers
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._review_allocation = cfg.get("review_allocation_multiplier", 0.5)
        self._review_ic_threshold = cfg.get("review_ic_threshold", 0.01)
        self._suspend_ic_threshold = cfg.get("suspend_ic_threshold", 0.0)
        self._recovery_ic_threshold = cfg.get("recovery_ic_threshold", 0.03)
        self._suspend_days = cfg.get("suspend_after_review_days", 30)
        self._strategies: Dict[str, StrategyState] = {}

    def register_strategy(
        self, strategy_name: str, date: pd.Timestamp
    ) -> StrategyState:
        """Register a new strategy as active."""
        state = StrategyState(
            strategy_name=strategy_name,
            status=StrategyStatus.ACTIVE,
            allocation_multiplier=1.0,
            status_since=date,
            history=[f"{date}: registered as ACTIVE"],
        )
        self._strategies[strategy_name] = state
        return state

    def update(
        self,
        strategy_name: str,
        current_ic: float,
        date: pd.Timestamp,
    ) -> StrategyState:
        """Update strategy status based on current IC performance.

        Args:
            strategy_name: Strategy identifier.
            current_ic: Current rolling IC value.
            date: Current date.

        Returns:
            Updated StrategyState.
        """
        if strategy_name not in self._strategies:
            return self.register_strategy(strategy_name, date)

        state = self._strategies[strategy_name]

        if state.status == StrategyStatus.RETIRED:
            return state

        if state.status == StrategyStatus.ACTIVE:
            if current_ic < self._review_ic_threshold:
                self._transition(
                    state,
                    StrategyStatus.UNDER_REVIEW,
                    self._review_allocation,
                    date,
                    f"IC degraded to {current_ic:.4f}",
                )

        elif state.status == StrategyStatus.UNDER_REVIEW:
            if current_ic >= self._recovery_ic_threshold:
                self._transition(
                    state,
                    StrategyStatus.ACTIVE,
                    1.0,
                    date,
                    f"IC recovered to {current_ic:.4f}",
                )
            elif current_ic < self._suspend_ic_threshold:
                days_in_review = (date - state.status_since).days
                if days_in_review >= self._suspend_days:
                    self._transition(
                        state,
                        StrategyStatus.SUSPENDED,
                        0.0,
                        date,
                        f"IC negative ({current_ic:.4f}) for {days_in_review} days",
                    )

        elif state.status == StrategyStatus.SUSPENDED:
            # Only manual retire is allowed from suspended
            pass

        return state

    def retire_strategy(
        self, strategy_name: str, date: pd.Timestamp, reason: str = ""
    ) -> Optional[StrategyState]:
        """Manually retire a strategy. Requires manual sign-off."""
        if strategy_name not in self._strategies:
            return None

        state = self._strategies[strategy_name]
        self._transition(
            state,
            StrategyStatus.RETIRED,
            0.0,
            date,
            reason or "manual retirement",
        )
        return state

    def get_state(self, strategy_name: str) -> Optional[StrategyState]:
        """Get current state of a strategy."""
        return self._strategies.get(strategy_name)

    def get_active_strategies(self) -> List[StrategyState]:
        """Get all strategies with non-zero allocation."""
        return [
            s for s in self._strategies.values()
            if s.allocation_multiplier > 0
        ]

    def get_all_states(self) -> Dict[str, StrategyState]:
        """Get all strategy states."""
        return dict(self._strategies)

    def get_allocation_multipliers(self) -> Dict[str, float]:
        """Get current allocation multipliers for all strategies."""
        return {
            name: state.allocation_multiplier
            for name, state in self._strategies.items()
        }

    def _transition(
        self,
        state: StrategyState,
        new_status: StrategyStatus,
        new_allocation: float,
        date: pd.Timestamp,
        reason: str,
    ) -> None:
        """Apply a status transition."""
        old_status = state.status
        state.status = new_status
        state.allocation_multiplier = new_allocation
        state.status_since = date
        state.review_reason = reason
        msg = f"{date}: {old_status.value} -> {new_status.value}: {reason}"
        state.history.append(msg)
        logger.info("Strategy %s: %s", state.strategy_name, msg)
