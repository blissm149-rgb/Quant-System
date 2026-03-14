"""Approval workflow — deployment gate for strategies.

Enforces the deployment gate via a state machine:
SUBMITTED → UNDER_REVIEW → APPROVED → DEPLOYED
                         → REJECTED
DEPLOYED  → SUSPENDED (on kill switch or IC degradation)
          → RETIRED (manual)

Requires risk_team_approved = True before DEPLOYED transition.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ApprovalState(str, Enum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    SUSPENDED = "suspended"
    RETIRED = "retired"


# Valid state transitions
_VALID_TRANSITIONS = {
    ApprovalState.SUBMITTED: {ApprovalState.UNDER_REVIEW},
    ApprovalState.UNDER_REVIEW: {ApprovalState.APPROVED, ApprovalState.REJECTED},
    ApprovalState.APPROVED: {ApprovalState.DEPLOYED, ApprovalState.REJECTED},
    ApprovalState.REJECTED: {ApprovalState.SUBMITTED},  # can resubmit
    ApprovalState.DEPLOYED: {ApprovalState.SUSPENDED, ApprovalState.RETIRED},
    ApprovalState.SUSPENDED: {ApprovalState.DEPLOYED, ApprovalState.RETIRED},
    ApprovalState.RETIRED: set(),  # terminal
}


@dataclass
class ApprovalRecord:
    """Record of a state transition in the approval workflow."""

    from_state: ApprovalState
    to_state: ApprovalState
    timestamp: pd.Timestamp
    actor: str = "system"
    reason: str = ""


@dataclass
class StrategyApproval:
    """Approval state for a strategy."""

    strategy_id: str
    state: ApprovalState = ApprovalState.SUBMITTED
    risk_team_approved: bool = False
    paper_trading_days: int = 0
    min_paper_trading_days: int = 126  # ~6 months
    review_result_passed: bool = False
    history: List[ApprovalRecord] = field(default_factory=list)
    created_at: Optional[pd.Timestamp] = None


class ApprovalWorkflow:
    """Enforces the strategy deployment gate.

    No strategy reaches live trading without passing through
    this workflow. Requires:
    - Review pipeline pass
    - Minimum 6 months paper trading
    - Risk team sign-off
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._min_paper_days = cfg.get("min_paper_trading_days", 126)
        self._strategies: Dict[str, StrategyApproval] = {}

    def submit(self, strategy_id: str) -> StrategyApproval:
        """Submit a strategy for approval."""
        approval = StrategyApproval(
            strategy_id=strategy_id,
            state=ApprovalState.SUBMITTED,
            min_paper_trading_days=self._min_paper_days,
            created_at=pd.Timestamp.now(),
        )
        self._strategies[strategy_id] = approval
        logger.info("Strategy %s submitted for approval", strategy_id)
        return approval

    def transition(
        self,
        strategy_id: str,
        to_state: ApprovalState,
        actor: str = "system",
        reason: str = "",
    ) -> bool:
        """Attempt a state transition.

        Returns True if transition succeeded, False otherwise.
        """
        approval = self._strategies.get(strategy_id)
        if approval is None:
            logger.warning("Strategy %s not found", strategy_id)
            return False

        current = approval.state
        valid_next = _VALID_TRANSITIONS.get(current, set())

        if to_state not in valid_next:
            logger.warning(
                "Invalid transition %s → %s for %s",
                current.value, to_state.value, strategy_id,
            )
            return False

        # Deployment gate checks
        if to_state == ApprovalState.DEPLOYED:
            if not approval.risk_team_approved:
                logger.warning(
                    "Cannot deploy %s: risk team approval required",
                    strategy_id,
                )
                return False
            if approval.paper_trading_days < approval.min_paper_trading_days:
                logger.warning(
                    "Cannot deploy %s: need %d paper trading days (have %d)",
                    strategy_id,
                    approval.min_paper_trading_days,
                    approval.paper_trading_days,
                )
                return False
            if not approval.review_result_passed:
                logger.warning(
                    "Cannot deploy %s: review pipeline not passed",
                    strategy_id,
                )
                return False

        record = ApprovalRecord(
            from_state=current,
            to_state=to_state,
            timestamp=pd.Timestamp.now(),
            actor=actor,
            reason=reason,
        )
        approval.history.append(record)
        approval.state = to_state

        logger.info(
            "Strategy %s: %s → %s (by %s)",
            strategy_id, current.value, to_state.value, actor,
        )
        return True

    def set_risk_approval(
        self, strategy_id: str, approved: bool, actor: str = "risk_team"
    ) -> bool:
        """Set risk team approval flag."""
        approval = self._strategies.get(strategy_id)
        if approval is None:
            return False
        approval.risk_team_approved = approved
        logger.info(
            "Strategy %s: risk_team_approved = %s (by %s)",
            strategy_id, approved, actor,
        )
        return True

    def set_review_result(
        self, strategy_id: str, passed: bool
    ) -> bool:
        """Record the review pipeline result."""
        approval = self._strategies.get(strategy_id)
        if approval is None:
            return False
        approval.review_result_passed = passed
        return True

    def update_paper_trading_days(
        self, strategy_id: str, days: int
    ) -> bool:
        """Update paper trading day count."""
        approval = self._strategies.get(strategy_id)
        if approval is None:
            return False
        approval.paper_trading_days = days
        return True

    def get_state(self, strategy_id: str) -> Optional[ApprovalState]:
        """Get current state for a strategy."""
        approval = self._strategies.get(strategy_id)
        return approval.state if approval else None

    def get_approval(self, strategy_id: str) -> Optional[StrategyApproval]:
        """Get full approval record."""
        return self._strategies.get(strategy_id)

    def get_deployed_strategies(self) -> List[str]:
        """Get all currently deployed strategies."""
        return [
            sid for sid, a in self._strategies.items()
            if a.state == ApprovalState.DEPLOYED
        ]

    def can_deploy(self, strategy_id: str) -> tuple:
        """Check if a strategy can be deployed, with reasons if not.

        Returns (bool, list of reasons).
        """
        approval = self._strategies.get(strategy_id)
        if approval is None:
            return False, ["Strategy not found"]

        reasons = []
        if approval.state not in (ApprovalState.APPROVED, ApprovalState.SUSPENDED):
            reasons.append(f"Current state is {approval.state.value}, need APPROVED")
        if not approval.risk_team_approved:
            reasons.append("Risk team approval required")
        if approval.paper_trading_days < approval.min_paper_trading_days:
            reasons.append(
                f"Need {approval.min_paper_trading_days} paper days "
                f"(have {approval.paper_trading_days})"
            )
        if not approval.review_result_passed:
            reasons.append("Review pipeline not passed")

        return len(reasons) == 0, reasons
