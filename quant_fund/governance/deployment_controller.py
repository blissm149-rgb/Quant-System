"""Deployment controller — the only module that flips paper to live.

Reads the approval workflow state and kill switch status before
enabling live orders. Maintains a registry of deployed strategies
and their allocation weights.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from quant_fund.governance.approval_workflow import (
    ApprovalState,
    ApprovalWorkflow,
)

logger = logging.getLogger(__name__)


@dataclass
class DeployedStrategy:
    """A strategy that is deployed for live trading."""

    strategy_id: str
    allocation_weight: float
    deployed_at: pd.Timestamp
    mode: str = "paper"  # "paper" or "live"


class DeploymentController:
    """Controls the transition from paper to live trading.

    This is the ONLY module that can flip a strategy from paper
    to live. It checks:
    - Approval workflow state == DEPLOYED
    - Kill switch is not active
    - Strategy allocation weight is set

    Maintains a registry of all deployed strategies.
    """

    def __init__(
        self,
        approval_workflow: ApprovalWorkflow,
        config: Optional[dict] = None,
    ):
        self._workflow = approval_workflow
        cfg = config or {}
        self._registry: Dict[str, DeployedStrategy] = {}
        self._kill_switch_active = False

    def deploy(
        self,
        strategy_id: str,
        allocation_weight: float = 0.0,
        mode: str = "paper",
    ) -> bool:
        """Deploy a strategy.

        Parameters
        ----------
        strategy_id : str
            Strategy to deploy.
        allocation_weight : float
            Allocation weight for this strategy.
        mode : str
            "paper" or "live". Live requires full approval.

        Returns
        -------
        bool
            True if deployment succeeded.
        """
        if self._kill_switch_active:
            logger.warning("Cannot deploy: kill switch is active")
            return False

        if mode == "live":
            state = self._workflow.get_state(strategy_id)
            if state != ApprovalState.DEPLOYED:
                logger.warning(
                    "Cannot go live: %s approval state is %s, need DEPLOYED",
                    strategy_id, state,
                )
                return False

        entry = DeployedStrategy(
            strategy_id=strategy_id,
            allocation_weight=allocation_weight,
            deployed_at=pd.Timestamp.now(),
            mode=mode,
        )
        self._registry[strategy_id] = entry
        logger.info(
            "Deployed %s in %s mode (weight=%.3f)",
            strategy_id, mode, allocation_weight,
        )
        return True

    def promote_to_live(self, strategy_id: str) -> bool:
        """Promote a paper-traded strategy to live."""
        entry = self._registry.get(strategy_id)
        if entry is None:
            logger.warning("Strategy %s not in registry", strategy_id)
            return False

        if self._kill_switch_active:
            logger.warning("Cannot promote to live: kill switch active")
            return False

        state = self._workflow.get_state(strategy_id)
        if state != ApprovalState.DEPLOYED:
            logger.warning(
                "Cannot promote %s: approval state is %s",
                strategy_id, state,
            )
            return False

        entry.mode = "live"
        logger.info("Promoted %s to live trading", strategy_id)
        return True

    def suspend(self, strategy_id: str, reason: str = "") -> bool:
        """Suspend a deployed strategy."""
        entry = self._registry.get(strategy_id)
        if entry is None:
            return False
        entry.mode = "suspended"
        self._workflow.transition(
            strategy_id, ApprovalState.SUSPENDED,
            actor="deployment_controller", reason=reason,
        )
        logger.info("Suspended %s: %s", strategy_id, reason)
        return True

    def remove(self, strategy_id: str) -> bool:
        """Remove a strategy from the deployment registry."""
        if strategy_id in self._registry:
            del self._registry[strategy_id]
            logger.info("Removed %s from deployment registry", strategy_id)
            return True
        return False

    def set_kill_switch(self, active: bool) -> None:
        """Set kill switch state. When active, suspends all live strategies."""
        self._kill_switch_active = active
        if active:
            for sid, entry in self._registry.items():
                if entry.mode == "live":
                    entry.mode = "suspended"
                    logger.warning("Kill switch: suspended %s", sid)

    def set_allocation_weight(
        self, strategy_id: str, weight: float
    ) -> bool:
        """Update allocation weight for a deployed strategy."""
        entry = self._registry.get(strategy_id)
        if entry is None:
            return False
        entry.allocation_weight = weight
        return True

    def get_live_strategies(self) -> List[DeployedStrategy]:
        """Get all strategies currently in live mode."""
        return [e for e in self._registry.values() if e.mode == "live"]

    def get_paper_strategies(self) -> List[DeployedStrategy]:
        """Get all strategies in paper trading mode."""
        return [e for e in self._registry.values() if e.mode == "paper"]

    def get_all_allocations(self) -> Dict[str, float]:
        """Get allocation weights for all active (non-suspended) strategies."""
        return {
            sid: e.allocation_weight
            for sid, e in self._registry.items()
            if e.mode in ("paper", "live")
        }

    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch_active
