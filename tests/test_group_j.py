"""Tests for Group J — Monitoring and Governance.

Covers: PnL dashboard, risk dashboard, execution quality monitor,
alerting system, system health monitor, strategy review pipeline,
approval workflow, and deployment controller.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.monitoring.risk_dashboard import RiskDashboard
from quant_fund.monitoring.execution_quality_monitor import ExecutionQualityMonitor
from quant_fund.monitoring.alerting_system import AlertingSystem, AlertLevel
from quant_fund.monitoring.system_health_monitor import (

    HealthCheck,
    SystemHealthMonitor,
)
from quant_fund.governance.strategy_review_pipeline import StrategyReviewPipeline
from quant_fund.governance.approval_workflow import (
    ApprovalState,
    ApprovalWorkflow,
)

pytestmark = [pytest.mark.tier2]
from quant_fund.governance.deployment_controller import DeploymentController


# ── PnL Dashboard ───────────────────────────────────────────────────

class TestPnLDashboard:
    def test_initial_state(self):
        dash = PnLDashboard({"initial_nav": 1_000_000})
        assert dash.latest is None
        assert dash.current_drawdown == 0.0

    def test_update_and_track(self):
        dash = PnLDashboard({"initial_nav": 1_000_000})
        snap = dash.update(1_010_000, timestamp=pd.Timestamp("2024-01-02"))
        assert snap.nav == 1_010_000
        assert snap.daily_pnl == 10_000
        assert snap.drawdown == 0.0  # at peak

    def test_drawdown_tracking(self):
        dash = PnLDashboard({"initial_nav": 1_000_000})
        dash.update(1_100_000, timestamp=pd.Timestamp("2024-01-02"))
        dash.update(1_000_000, timestamp=pd.Timestamp("2024-01-03"))
        assert dash.current_drawdown == pytest.approx(100_000 / 1_100_000, rel=0.01)

    def test_return_series(self):
        dash = PnLDashboard({"initial_nav": 1_000_000})
        dash.update(1_010_000, timestamp=pd.Timestamp("2024-01-02"))
        dash.update(1_020_000, timestamp=pd.Timestamp("2024-01-03"))
        series = dash.get_return_series()
        assert len(series) == 2

    def test_summary(self):
        dash = PnLDashboard({"initial_nav": 1_000_000})
        for i in range(10):
            nav = 1_000_000 + (i + 1) * 1000
            dash.update(nav, timestamp=pd.Timestamp(f"2024-01-{i+2:02d}"))
        summary = dash.get_summary()
        assert summary["current_nav"] == 1_010_000
        assert summary["num_days"] == 10

    def test_realised_unrealised_split(self):
        dash = PnLDashboard({"initial_nav": 1_000_000})
        dash.update(1_050_000, realised_pnl_today=20_000,
                    timestamp=pd.Timestamp("2024-01-02"))
        snap = dash.latest
        assert snap.realised_pnl == 20_000
        assert snap.unrealised_pnl == pytest.approx(30_000, rel=0.01)


# ── Risk Dashboard ──────────────────────────────────────────────────

class TestRiskDashboard:
    def test_basic_snapshot(self):
        dash = RiskDashboard()
        weights = pd.Series({"AAPL": 0.01, "MSFT": -0.005, "GOOG": 0.008})
        snap = dash.compute_snapshot(weights)
        assert snap.gross_exposure == pytest.approx(0.023, rel=0.01)
        assert snap.net_exposure == pytest.approx(0.013, rel=0.01)

    def test_sector_exposures(self):
        dash = RiskDashboard()
        weights = pd.Series({"AAPL": 0.01, "MSFT": 0.01, "XOM": -0.005})
        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy"}
        snap = dash.compute_snapshot(weights, sector_map=sector_map)
        assert "Tech" in snap.sector_exposures
        assert snap.sector_exposures["Tech"] == pytest.approx(0.02)

    def test_var_computation(self):
        dash = RiskDashboard()
        weights = pd.Series({"A": 0.5, "B": 0.5})
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0, 0.02, (252, 2)),
            columns=["A", "B"],
        )
        snap = dash.compute_snapshot(weights, return_history=returns)
        assert snap.var_95 > 0
        assert snap.var_99 > snap.var_95

    def test_check_limits(self):
        dash = RiskDashboard({"max_leverage": 2.0, "max_sector_exposure": 0.20})
        weights = pd.Series({"A": 1.5, "B": 1.0})
        sector_map = {"A": "Tech", "B": "Tech"}
        snap = dash.compute_snapshot(weights, sector_map=sector_map)
        breaches = dash.check_limits(snap)
        assert len(breaches) >= 1  # leverage breach

    def test_top_contributors(self):
        dash = RiskDashboard()
        weights = pd.Series({f"T{i}": 0.01 * (i + 1) for i in range(15)})
        snap = dash.compute_snapshot(weights)
        assert len(snap.top_contributors) == 10


# ── Execution Quality Monitor ───────────────────────────────────────

class TestExecutionQualityMonitor:
    def test_record_buy_execution(self):
        mon = ExecutionQualityMonitor()
        rec = mon.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=100, filled_qty=100,
            decision_price=150.0, fill_price=150.15,
        )
        assert rec.implementation_shortfall_bps == pytest.approx(10.0, rel=0.1)
        assert rec.fill_rate == 1.0

    def test_record_sell_execution(self):
        mon = ExecutionQualityMonitor()
        rec = mon.record_execution(
            order_id="O2", ticker="MSFT", side="sell",
            target_qty=50, filled_qty=50,
            decision_price=300.0, fill_price=299.70,
        )
        assert rec.implementation_shortfall_bps == pytest.approx(10.0, rel=0.1)

    def test_partial_fill(self):
        mon = ExecutionQualityMonitor()
        rec = mon.record_execution(
            order_id="O3", ticker="GOOG", side="buy",
            target_qty=100, filled_qty=60,
            decision_price=140.0, fill_price=140.0,
        )
        assert rec.fill_rate == pytest.approx(0.6)

    def test_summary(self):
        mon = ExecutionQualityMonitor()
        for i in range(5):
            mon.record_execution(
                order_id=f"O{i}", ticker="AAPL", side="buy",
                target_qty=100, filled_qty=100,
                decision_price=150.0, fill_price=150.0 + i * 0.01,
            )
        summary = mon.get_summary()
        assert summary.num_orders == 5
        assert summary.avg_fill_rate == 1.0

    def test_adverse_selection_detection(self):
        mon = ExecutionQualityMonitor({"adverse_threshold_bps": 10.0})
        mon.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=100, filled_qty=100,
            decision_price=150.0, fill_price=150.30,  # 20 bps
        )
        flagged = mon.detect_adverse_selection()
        assert len(flagged) == 1


# ── Alerting System ─────────────────────────────────────────────────

class TestAlertingSystem:
    def test_send_alert(self):
        system = AlertingSystem({"suppress_duplicates_seconds": 0})
        alert = system.send(AlertLevel.WARNING, "test", "Test alert")
        assert alert.level == AlertLevel.WARNING
        assert alert.alert_id.startswith("ALERT-")

    def test_critical_count(self):
        system = AlertingSystem({"suppress_duplicates_seconds": 0})
        system.send(AlertLevel.CRITICAL, "kill_switch", "Drawdown breach")
        system.send(AlertLevel.WARNING, "exposure", "Sector near limit")
        assert system.critical_count == 1

    def test_acknowledge(self):
        system = AlertingSystem({"suppress_duplicates_seconds": 0})
        alert = system.send(AlertLevel.CRITICAL, "test", "Critical issue")
        assert system.critical_count == 1
        system.acknowledge(alert.alert_id)
        assert system.critical_count == 0

    def test_handler_dispatch(self):
        system = AlertingSystem({"suppress_duplicates_seconds": 0})
        received = []
        system.register_handler(AlertLevel.CRITICAL, lambda a: received.append(a))
        system.send(AlertLevel.CRITICAL, "test", "Fire!")
        assert len(received) == 1

    def test_get_history(self):
        system = AlertingSystem({"suppress_duplicates_seconds": 0})
        system.send(AlertLevel.INFO, "a", "msg1")
        system.send(AlertLevel.WARNING, "b", "msg2")
        history = system.get_history(level=AlertLevel.WARNING)
        assert len(history) == 1


# ── System Health Monitor ───────────────────────────────────────────

class TestSystemHealthMonitor:
    def test_healthy_data_feed(self):
        mon = SystemHealthMonitor({"max_data_lag_s": 300})
        mon.record_data_timestamp("prices", pd.Timestamp.now())
        checks = mon.check_data_feed_health()
        assert len(checks) == 1
        assert checks[0].status == "healthy"

    def test_stale_data_feed(self):
        mon = SystemHealthMonitor({"max_data_lag_s": 60})
        mon.record_data_timestamp(
            "prices", pd.Timestamp.now() - pd.Timedelta(seconds=120)
        )
        checks = mon.check_data_feed_health()
        assert checks[0].status == "degraded"

    def test_full_health_check(self):
        mon = SystemHealthMonitor()
        mon.record_data_timestamp("prices", pd.Timestamp.now())
        snap = mon.run_health_check()
        assert snap.overall_status == "healthy"
        assert snap.uptime_s > 0

    def test_uptime(self):
        mon = SystemHealthMonitor()
        assert mon.uptime_seconds > 0


# ── Strategy Review Pipeline ────────────────────────────────────────

class TestStrategyReviewPipeline:
    def test_passing_review(self):
        pipeline = StrategyReviewPipeline()
        result = pipeline.review(
            strategy_id="mom_v1",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            num_eval_days=300,
            transaction_costs_included=True,
            look_ahead_bias_flags=0,
            capacity_impact_pct=0.05,
            max_factor_exposure=1.0,
        )
        assert result.passed is True
        assert len(result.failed_checks) == 0

    def test_failing_sharpe(self):
        pipeline = StrategyReviewPipeline()
        result = pipeline.review(
            strategy_id="bad_strat",
            sharpe_ratio=0.5,
            max_drawdown=0.15,
            num_eval_days=300,
            transaction_costs_included=True,
        )
        assert result.passed is False
        failed_names = [c.check_name for c in result.failed_checks]
        assert "sharpe_ratio" in failed_names

    def test_failing_drawdown(self):
        pipeline = StrategyReviewPipeline()
        result = pipeline.review(
            strategy_id="risky",
            sharpe_ratio=1.5,
            max_drawdown=0.30,
            num_eval_days=300,
            transaction_costs_included=True,
        )
        assert result.passed is False
        failed_names = [c.check_name for c in result.failed_checks]
        assert "max_drawdown" in failed_names

    def test_insufficient_eval_period(self):
        pipeline = StrategyReviewPipeline()
        result = pipeline.review(
            strategy_id="short",
            sharpe_ratio=2.0,
            max_drawdown=0.10,
            num_eval_days=100,
            transaction_costs_included=True,
        )
        assert result.passed is False

    def test_look_ahead_bias_flag(self):
        pipeline = StrategyReviewPipeline()
        result = pipeline.review(
            strategy_id="biased",
            sharpe_ratio=2.0,
            max_drawdown=0.10,
            num_eval_days=300,
            transaction_costs_included=True,
            look_ahead_bias_flags=3,
        )
        assert result.passed is False


# ── Approval Workflow ────────────────────────────────────────────────

class TestApprovalWorkflow:
    def test_submit_and_approve(self):
        wf = ApprovalWorkflow()
        wf.submit("strat1")
        assert wf.get_state("strat1") == ApprovalState.SUBMITTED
        assert wf.transition("strat1", ApprovalState.UNDER_REVIEW)
        assert wf.transition("strat1", ApprovalState.APPROVED)
        assert wf.get_state("strat1") == ApprovalState.APPROVED

    def test_invalid_transition_rejected(self):
        wf = ApprovalWorkflow()
        wf.submit("strat1")
        # Can't go directly from SUBMITTED to APPROVED
        assert wf.transition("strat1", ApprovalState.APPROVED) is False

    def test_deployment_requires_risk_approval(self):
        wf = ApprovalWorkflow({"min_paper_trading_days": 0})
        wf.submit("strat1")
        wf.transition("strat1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat1", ApprovalState.APPROVED)
        wf.set_review_result("strat1", True)
        # Missing risk team approval
        assert wf.transition("strat1", ApprovalState.DEPLOYED) is False
        wf.set_risk_approval("strat1", True)
        assert wf.transition("strat1", ApprovalState.DEPLOYED) is True

    def test_deployment_requires_paper_trading(self):
        wf = ApprovalWorkflow({"min_paper_trading_days": 126})
        wf.submit("strat1")
        wf.transition("strat1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat1", ApprovalState.APPROVED)
        wf.set_risk_approval("strat1", True)
        wf.set_review_result("strat1", True)
        # Not enough paper trading days
        assert wf.transition("strat1", ApprovalState.DEPLOYED) is False
        wf.update_paper_trading_days("strat1", 130)
        assert wf.transition("strat1", ApprovalState.DEPLOYED) is True

    def test_can_deploy_check(self):
        wf = ApprovalWorkflow()
        wf.submit("strat1")
        can, reasons = wf.can_deploy("strat1")
        assert can is False
        assert len(reasons) > 0

    def test_deployed_strategies_list(self):
        wf = ApprovalWorkflow({"min_paper_trading_days": 0})
        wf.submit("strat1")
        wf.transition("strat1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat1", ApprovalState.APPROVED)
        wf.set_risk_approval("strat1", True)
        wf.set_review_result("strat1", True)
        wf.transition("strat1", ApprovalState.DEPLOYED)
        assert "strat1" in wf.get_deployed_strategies()

    def test_suspend_and_retire(self):
        wf = ApprovalWorkflow({"min_paper_trading_days": 0})
        wf.submit("strat1")
        wf.transition("strat1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat1", ApprovalState.APPROVED)
        wf.set_risk_approval("strat1", True)
        wf.set_review_result("strat1", True)
        wf.transition("strat1", ApprovalState.DEPLOYED)
        wf.transition("strat1", ApprovalState.SUSPENDED, reason="IC degradation")
        assert wf.get_state("strat1") == ApprovalState.SUSPENDED
        wf.transition("strat1", ApprovalState.RETIRED)
        assert wf.get_state("strat1") == ApprovalState.RETIRED


# ── Deployment Controller ───────────────────────────────────────────

class TestDeploymentController:
    def _setup_approved_strategy(self, wf, sid="strat1"):
        wf.submit(sid)
        wf.transition(sid, ApprovalState.UNDER_REVIEW)
        wf.transition(sid, ApprovalState.APPROVED)
        wf.set_risk_approval(sid, True)
        wf.set_review_result(sid, True)
        wf.transition(sid, ApprovalState.DEPLOYED)

    def test_deploy_paper(self):
        wf = ApprovalWorkflow({"min_paper_trading_days": 0})
        ctrl = DeploymentController(wf)
        assert ctrl.deploy("strat1", mode="paper") is True

    def test_deploy_live_requires_approval(self):
        wf = ApprovalWorkflow({"min_paper_trading_days": 0})
        ctrl = DeploymentController(wf)
        # Not approved yet
        assert ctrl.deploy("strat1", mode="live") is False
        self._setup_approved_strategy(wf)
        assert ctrl.deploy("strat1", mode="live") is True

    def test_kill_switch_suspends_live(self):
        wf = ApprovalWorkflow({"min_paper_trading_days": 0})
        ctrl = DeploymentController(wf)
        self._setup_approved_strategy(wf)
        ctrl.deploy("strat1", mode="live")
        assert len(ctrl.get_live_strategies()) == 1
        ctrl.set_kill_switch(True)
        assert len(ctrl.get_live_strategies()) == 0
        assert ctrl.is_kill_switch_active

    def test_kill_switch_prevents_deployment(self):
        wf = ApprovalWorkflow({"min_paper_trading_days": 0})
        ctrl = DeploymentController(wf)
        ctrl.set_kill_switch(True)
        assert ctrl.deploy("strat1", mode="paper") is False

    def test_get_allocations(self):
        wf = ApprovalWorkflow({"min_paper_trading_days": 0})
        ctrl = DeploymentController(wf)
        ctrl.deploy("s1", allocation_weight=0.3, mode="paper")
        ctrl.deploy("s2", allocation_weight=0.7, mode="paper")
        allocs = ctrl.get_all_allocations()
        assert allocs["s1"] == pytest.approx(0.3)
        assert allocs["s2"] == pytest.approx(0.7)

    def test_promote_to_live(self):
        wf = ApprovalWorkflow({"min_paper_trading_days": 0})
        ctrl = DeploymentController(wf)
        self._setup_approved_strategy(wf)
        ctrl.deploy("strat1", mode="paper")
        assert ctrl.promote_to_live("strat1") is True
        assert len(ctrl.get_live_strategies()) == 1
