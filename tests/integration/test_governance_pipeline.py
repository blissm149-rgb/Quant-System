"""Integration Chain 4: Governance Pipeline

Tests: strategy_review_pipeline → approval_workflow → deployment_controller
→ paper/live mode transitions
"""

import pandas as pd
import pytest


@pytest.mark.integration
@pytest.mark.tier3
class TestReviewToApproval:
    """Strategy review results feed into approval workflow."""

    def test_good_strategy_passes_review_and_deploys(self):
        """Strategy meeting all criteria passes review → approval → deployment."""
        from quant_fund.governance.strategy_review_pipeline import StrategyReviewPipeline
        from quant_fund.governance.approval_workflow import ApprovalWorkflow, ApprovalState
        from quant_fund.governance.deployment_controller import DeploymentController

        # Step 1: Review passes
        review = StrategyReviewPipeline()
        result = review.review(
            strategy_id="strat_001",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            num_eval_days=300,
            transaction_costs_included=True,
            look_ahead_bias_flags=0,
            capacity_impact_pct=0.05,
            max_factor_exposure=1.0,
        )
        assert result.passed is True

        # Step 2: Approval workflow
        workflow = ApprovalWorkflow({"min_paper_trading_days": 126})
        workflow.submit("strat_001")
        workflow.transition("strat_001", ApprovalState.UNDER_REVIEW, actor="system")
        workflow.set_review_result("strat_001", passed=True)
        workflow.transition("strat_001", ApprovalState.APPROVED, actor="system")
        workflow.set_risk_approval("strat_001", approved=True, actor="risk_team")
        workflow.update_paper_trading_days("strat_001", 130)

        # Step 3: Deploy
        can, reasons = workflow.can_deploy("strat_001")
        assert can is True, f"Cannot deploy: {reasons}"

        workflow.transition("strat_001", ApprovalState.DEPLOYED, actor="system")
        assert workflow.get_state("strat_001") == ApprovalState.DEPLOYED

        # Step 4: DeploymentController
        controller = DeploymentController(workflow)
        deployed = controller.deploy("strat_001", allocation_weight=0.10, mode="paper")
        assert deployed is True

    def test_bad_sharpe_rejected_at_review(self):
        """Strategy with Sharpe < 1.0 fails review."""
        from quant_fund.governance.strategy_review_pipeline import StrategyReviewPipeline

        review = StrategyReviewPipeline()
        result = review.review(
            strategy_id="bad_strat",
            sharpe_ratio=0.5,  # below 1.0 threshold
            max_drawdown=0.15,
            num_eval_days=300,
            transaction_costs_included=True,
        )
        assert result.passed is False
        failed_names = [c.check_name for c in result.failed_checks]
        assert "sharpe_ratio" in failed_names

    def test_high_drawdown_rejected_at_review(self):
        """Strategy with max drawdown > 25% fails review."""
        from quant_fund.governance.strategy_review_pipeline import StrategyReviewPipeline

        review = StrategyReviewPipeline()
        result = review.review(
            strategy_id="dd_strat",
            sharpe_ratio=1.5,
            max_drawdown=0.30,  # above 0.25 threshold
            num_eval_days=300,
            transaction_costs_included=True,
        )
        assert result.passed is False
        failed_names = [c.check_name for c in result.failed_checks]
        assert "max_drawdown" in failed_names

    def test_insufficient_eval_days_rejected(self):
        """Strategy with < 252 eval days fails review."""
        from quant_fund.governance.strategy_review_pipeline import StrategyReviewPipeline

        review = StrategyReviewPipeline()
        result = review.review(
            strategy_id="short_strat",
            sharpe_ratio=2.0,
            max_drawdown=0.10,
            num_eval_days=100,  # below 252
            transaction_costs_included=True,
        )
        assert result.passed is False


@pytest.mark.integration
@pytest.mark.tier3
class TestDeploymentGating:
    """Deployment requires all governance gates to be satisfied."""

    def test_deploy_without_risk_approval_blocked(self):
        """Cannot deploy to DEPLOYED without risk_team_approved."""
        from quant_fund.governance.approval_workflow import ApprovalWorkflow, ApprovalState

        workflow = ApprovalWorkflow()
        workflow.submit("strat_002")
        workflow.transition("strat_002", ApprovalState.UNDER_REVIEW)
        workflow.set_review_result("strat_002", passed=True)
        workflow.transition("strat_002", ApprovalState.APPROVED)
        # Missing: risk approval and paper trading days

        can, reasons = workflow.can_deploy("strat_002")
        assert can is False
        assert len(reasons) > 0

    def test_deploy_without_paper_trading_blocked(self):
        """Cannot deploy without sufficient paper trading days."""
        from quant_fund.governance.approval_workflow import ApprovalWorkflow, ApprovalState

        workflow = ApprovalWorkflow({"min_paper_trading_days": 126})
        workflow.submit("strat_003")
        workflow.transition("strat_003", ApprovalState.UNDER_REVIEW)
        workflow.set_review_result("strat_003", passed=True)
        workflow.transition("strat_003", ApprovalState.APPROVED)
        workflow.set_risk_approval("strat_003", approved=True)
        workflow.update_paper_trading_days("strat_003", 50)  # only 50 days

        can, reasons = workflow.can_deploy("strat_003")
        assert can is False

    def test_paper_to_live_requires_deployed_state(self):
        """Promoting paper → live requires strategy in DEPLOYED state."""
        from quant_fund.governance.approval_workflow import ApprovalWorkflow, ApprovalState
        from quant_fund.governance.deployment_controller import DeploymentController

        workflow = ApprovalWorkflow()
        workflow.submit("strat_004")
        workflow.transition("strat_004", ApprovalState.UNDER_REVIEW)
        workflow.set_review_result("strat_004", passed=True)
        workflow.transition("strat_004", ApprovalState.APPROVED)
        workflow.set_risk_approval("strat_004", approved=True)
        workflow.update_paper_trading_days("strat_004", 130)
        workflow.transition("strat_004", ApprovalState.DEPLOYED)

        controller = DeploymentController(workflow)
        controller.deploy("strat_004", mode="paper")

        # Promote to live
        promoted = controller.promote_to_live("strat_004")
        assert promoted is True

    def test_kill_switch_suspends_live_strategies(self):
        """Setting kill switch suspends all live strategies."""
        from quant_fund.governance.approval_workflow import ApprovalWorkflow, ApprovalState
        from quant_fund.governance.deployment_controller import DeploymentController

        workflow = ApprovalWorkflow()
        workflow.submit("strat_005")
        workflow.transition("strat_005", ApprovalState.UNDER_REVIEW)
        workflow.set_review_result("strat_005", passed=True)
        workflow.transition("strat_005", ApprovalState.APPROVED)
        workflow.set_risk_approval("strat_005", approved=True)
        workflow.update_paper_trading_days("strat_005", 130)
        workflow.transition("strat_005", ApprovalState.DEPLOYED)

        controller = DeploymentController(workflow)
        controller.deploy("strat_005", mode="paper")
        controller.promote_to_live("strat_005")

        live_before = controller.get_live_strategies()
        assert len(live_before) > 0

        # Activate kill switch
        controller.set_kill_switch(True)
        assert controller.is_kill_switch_active

        live_after = controller.get_live_strategies()
        assert len(live_after) == 0
