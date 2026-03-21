"""Unit tests for governance subsystem.

TESTING_PLAN.md Section 3.14 — ApprovalWorkflow, DeploymentController,
StrategyReviewPipeline (CRITICAL: deployment gate).
"""

import pytest

from quant_fund.governance.approval_workflow import (
    ApprovalState,
    ApprovalWorkflow,
)
from quant_fund.governance.deployment_controller import DeploymentController
from quant_fund.governance.strategy_review_pipeline import (
    StrategyReviewPipeline,
    ReviewResult,
)


# ── Approval Workflow ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier1
class TestApprovalWorkflow:
    """ApprovalWorkflow — CRITICAL: deployment gate state machine."""

    @pytest.fixture
    def wf(self):
        return ApprovalWorkflow(config={"min_paper_trading_days": 126})

    def test_submit_creates_strategy(self, wf):
        approval = wf.submit("strat_1")
        assert approval.state == ApprovalState.SUBMITTED

    def test_valid_transition_submitted_to_review(self, wf):
        wf.submit("strat_1")
        assert wf.transition("strat_1", ApprovalState.UNDER_REVIEW) is True
        assert wf.get_state("strat_1") == ApprovalState.UNDER_REVIEW

    def test_invalid_transition_submitted_to_deployed(self, wf):
        """Cannot jump directly to DEPLOYED."""
        wf.submit("strat_1")
        assert wf.transition("strat_1", ApprovalState.DEPLOYED) is False

    def test_deploy_requires_risk_approval(self, wf):
        """DEPLOYED requires risk_team_approved = True."""
        wf.submit("strat_1")
        wf.transition("strat_1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat_1", ApprovalState.APPROVED)
        # Try to deploy without risk approval
        wf.set_review_result("strat_1", True)
        wf.update_paper_trading_days("strat_1", 200)
        assert wf.transition("strat_1", ApprovalState.DEPLOYED) is False

    def test_deploy_requires_min_paper_days(self, wf):
        """DEPLOYED requires minimum paper trading days."""
        wf.submit("strat_1")
        wf.transition("strat_1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat_1", ApprovalState.APPROVED)
        wf.set_risk_approval("strat_1", True)
        wf.set_review_result("strat_1", True)
        wf.update_paper_trading_days("strat_1", 10)  # too few
        assert wf.transition("strat_1", ApprovalState.DEPLOYED) is False

    def test_full_deployment_lifecycle(self, wf):
        """Complete workflow: SUBMITTED → UNDER_REVIEW → APPROVED → DEPLOYED."""
        wf.submit("strat_1")
        wf.transition("strat_1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat_1", ApprovalState.APPROVED)
        wf.set_risk_approval("strat_1", True)
        wf.set_review_result("strat_1", True)
        wf.update_paper_trading_days("strat_1", 200)
        assert wf.transition("strat_1", ApprovalState.DEPLOYED) is True
        assert wf.get_state("strat_1") == ApprovalState.DEPLOYED

    def test_deployed_to_suspended(self, wf):
        """DEPLOYED → SUSPENDED on kill switch."""
        wf.submit("strat_1")
        wf.transition("strat_1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat_1", ApprovalState.APPROVED)
        wf.set_risk_approval("strat_1", True)
        wf.set_review_result("strat_1", True)
        wf.update_paper_trading_days("strat_1", 200)
        wf.transition("strat_1", ApprovalState.DEPLOYED)
        assert wf.transition("strat_1", ApprovalState.SUSPENDED) is True

    def test_retired_is_terminal(self, wf):
        """RETIRED has no outbound transitions."""
        wf.submit("strat_1")
        wf.transition("strat_1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat_1", ApprovalState.APPROVED)
        wf.set_risk_approval("strat_1", True)
        wf.set_review_result("strat_1", True)
        wf.update_paper_trading_days("strat_1", 200)
        wf.transition("strat_1", ApprovalState.DEPLOYED)
        wf.transition("strat_1", ApprovalState.RETIRED)
        assert wf.transition("strat_1", ApprovalState.DEPLOYED) is False

    def test_can_deploy_returns_reasons(self, wf):
        """can_deploy explains why deployment is blocked."""
        wf.submit("strat_1")
        ok, reasons = wf.can_deploy("strat_1")
        assert ok is False
        assert len(reasons) > 0

    def test_get_deployed_strategies(self, wf):
        """get_deployed_strategies returns deployed strategy IDs."""
        wf.submit("strat_1")
        wf.transition("strat_1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat_1", ApprovalState.APPROVED)
        wf.set_risk_approval("strat_1", True)
        wf.set_review_result("strat_1", True)
        wf.update_paper_trading_days("strat_1", 200)
        wf.transition("strat_1", ApprovalState.DEPLOYED)
        assert "strat_1" in wf.get_deployed_strategies()


# ── Deployment Controller ──────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier1
class TestDeploymentController:
    """DeploymentController — CRITICAL: paper → live gate."""

    @pytest.fixture
    def workflow(self):
        return ApprovalWorkflow()

    @pytest.fixture
    def controller(self, workflow):
        return DeploymentController(workflow)

    def _approve_strategy(self, wf, sid):
        """Helper to fully approve a strategy."""
        wf.submit(sid)
        wf.transition(sid, ApprovalState.UNDER_REVIEW)
        wf.transition(sid, ApprovalState.APPROVED)
        wf.set_risk_approval(sid, True)
        wf.set_review_result(sid, True)
        wf.update_paper_trading_days(sid, 200)
        wf.transition(sid, ApprovalState.DEPLOYED)

    def test_deploy_paper_succeeds(self, controller, workflow):
        """Paper deploy doesn't need full approval."""
        assert controller.deploy("strat_1", allocation_weight=0.5, mode="paper") is True

    def test_deploy_live_requires_approval(self, controller, workflow):
        """Live deploy requires DEPLOYED approval state."""
        workflow.submit("strat_1")
        assert controller.deploy("strat_1", mode="live") is False

    def test_deploy_live_with_approval(self, controller, workflow):
        """Live deploy succeeds with full approval."""
        self._approve_strategy(workflow, "strat_1")
        assert controller.deploy("strat_1", allocation_weight=0.5, mode="live") is True

    def test_kill_switch_blocks_deploy(self, controller):
        """Kill switch prevents all deployment."""
        controller.set_kill_switch(True)
        assert controller.deploy("strat_1", mode="paper") is False

    def test_kill_switch_suspends_live(self, controller, workflow):
        """Kill switch suspends all live strategies."""
        self._approve_strategy(workflow, "strat_1")
        controller.deploy("strat_1", allocation_weight=0.5, mode="live")
        controller.set_kill_switch(True)
        assert len(controller.get_live_strategies()) == 0

    def test_remove_strategy(self, controller):
        """Remove strategy from registry."""
        controller.deploy("strat_1", mode="paper")
        assert controller.remove("strat_1") is True
        assert controller.remove("strat_1") is False  # already removed

    def test_get_all_allocations(self, controller):
        """Allocations for active strategies."""
        controller.deploy("strat_1", allocation_weight=0.6, mode="paper")
        controller.deploy("strat_2", allocation_weight=0.4, mode="paper")
        allocs = controller.get_all_allocations()
        assert allocs["strat_1"] == pytest.approx(0.6)
        assert allocs["strat_2"] == pytest.approx(0.4)


# ── Strategy Review Pipeline ──────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier1
class TestStrategyReviewPipeline:
    """StrategyReviewPipeline — pre-deployment checklist."""

    @pytest.fixture
    def pipeline(self):
        return StrategyReviewPipeline()

    def test_passing_review(self, pipeline):
        result = pipeline.review(
            strategy_id="strat_1",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            num_eval_days=300,
            transaction_costs_included=True,
            look_ahead_bias_flags=0,
            capacity_impact_pct=0.05,
            max_factor_exposure=1.0,
        )
        assert isinstance(result, ReviewResult)
        assert result.passed is True

    def test_failing_sharpe(self, pipeline):
        result = pipeline.review(
            strategy_id="strat_1",
            sharpe_ratio=0.5,  # below 1.0 threshold
            max_drawdown=0.15,
            num_eval_days=300,
            transaction_costs_included=True,
        )
        assert result.passed is False
        assert any(c.check_name == "sharpe_ratio" for c in result.failed_checks)

    def test_failing_drawdown(self, pipeline):
        result = pipeline.review(
            strategy_id="strat_1",
            sharpe_ratio=1.5,
            max_drawdown=0.40,  # above 0.25 threshold
            num_eval_days=300,
            transaction_costs_included=True,
        )
        assert result.passed is False
        assert any(c.check_name == "max_drawdown" for c in result.failed_checks)

    def test_insufficient_eval_period(self, pipeline):
        result = pipeline.review(
            strategy_id="strat_1",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            num_eval_days=100,  # below 252 threshold
            transaction_costs_included=True,
        )
        assert result.passed is False

    def test_look_ahead_bias_flags(self, pipeline):
        result = pipeline.review(
            strategy_id="strat_1",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            num_eval_days=300,
            transaction_costs_included=True,
            look_ahead_bias_flags=2,
        )
        assert result.passed is False

    def test_costs_not_included_fails(self, pipeline):
        result = pipeline.review(
            strategy_id="strat_1",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            num_eval_days=300,
            transaction_costs_included=False,
        )
        assert result.passed is False
