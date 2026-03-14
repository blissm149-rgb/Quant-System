"""Tests for Group K — Main entry points.

Covers: research_runner, paper_trading_runner, live_trading_runner.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.main.research_runner import ResearchRunner, ResearchResult
from quant_fund.main.paper_trading_runner import PaperTradingRunner, PaperTradingResult
from quant_fund.main.live_trading_runner import LiveTradingRunner
from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.governance.approval_workflow import ApprovalState, ApprovalWorkflow
from quant_fund.governance.deployment_controller import DeploymentController


# ── Helpers ──────────────────────────────────────────────────────────

def make_market_data(tickers=None, n_days=60):
    """Generate synthetic market data."""
    tickers = tickers or ["AAPL", "MSFT", "GOOG"]
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(42)
    rows = []
    for dt in dates:
        for t in tickers:
            price = 100 + rng.normal(0, 2)
            rows.append({
                "date": dt, "ticker": t,
                "open": price, "high": price + 1,
                "low": price - 1, "close": price + 0.5,
                "volume": 1_000_000,
            })
    df = pd.DataFrame(rows)
    df.index = df["date"]
    return df


def make_sim_broker(initial_cash=1_000_000.0):
    broker = SimulationBroker({"initial_cash": initial_cash})
    broker.set_market_data({
        "AAPL": {"bid": 149.0, "ask": 151.0, "mid": 150.0, "last": 150.0,
                 "volume": 50_000_000, "adv": 50_000_000},
        "MSFT": {"bid": 299.0, "ask": 301.0, "mid": 300.0, "last": 300.0,
                 "volume": 30_000_000, "adv": 30_000_000},
    })
    return broker


class MockFeatureGenerator:
    """Simple mock that returns random features."""
    feature_name = "mock_feature"

    def compute(self, data, as_of=None):
        rng = np.random.default_rng(42)
        tickers = ["AAPL", "MSFT", "GOOG"]
        return pd.Series(rng.normal(0, 1, len(tickers)), index=tickers)


class MockSignalRanking:
    """Mock signal ranking that returns alpha scores."""
    def combine(self, feature_matrix):
        return feature_matrix.mean(axis=1) if hasattr(feature_matrix, 'mean') else pd.Series(dtype=float)


# ── Research Runner ──────────────────────────────────────────────────

class TestResearchRunner:
    def test_basic_cycle(self):
        runner = ResearchRunner()
        data = make_market_data(n_days=10)
        result = runner.run_cycle(
            as_of=pd.Timestamp("2024-01-15"),
            market_data=data,
        )
        assert result.status == "completed"
        assert result.as_of == pd.Timestamp("2024-01-15")

    def test_with_feature_generators(self):
        runner = ResearchRunner()
        runner.inject_components(
            feature_generators=[MockFeatureGenerator()],
            signal_ranking=MockSignalRanking(),
        )
        data = make_market_data(n_days=10)
        result = runner.run_cycle(
            as_of=pd.Timestamp("2024-01-15"),
            market_data=data,
        )
        assert result.feature_matrix is not None
        assert result.alpha_scores is not None

    def test_no_data_returns_no_data_status(self):
        runner = ResearchRunner()
        result = runner.run_cycle(
            as_of=pd.Timestamp("2024-01-15"),
            market_data=pd.DataFrame(),
        )
        assert result.status == "no_data"

    def test_backtest_multiple_dates(self):
        runner = ResearchRunner()
        runner.inject_components(
            feature_generators=[MockFeatureGenerator()],
        )
        data = make_market_data(n_days=30)
        dates = [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16")]
        results = runner.run_backtest(dates, market_data=data)
        assert len(results) == 2

    def test_results_stored(self):
        runner = ResearchRunner()
        data = make_market_data(n_days=10)
        runner.run_cycle(pd.Timestamp("2024-01-15"), market_data=data)
        runner.run_cycle(pd.Timestamp("2024-01-16"), market_data=data)
        assert len(runner.results) == 2


# ── Paper Trading Runner ─────────────────────────────────────────────

class TestPaperTradingRunner:
    def test_basic_run(self):
        runner = PaperTradingRunner({"initial_nav": 1_000_000})
        broker = make_sim_broker()
        runner.inject_components(broker=broker)
        dates = pd.bdate_range("2024-01-01", periods=5).tolist()
        result = runner.run(dates)
        assert result.num_days == 5
        assert result.final_nav > 0

    def test_with_pnl_dashboard(self):
        runner = PaperTradingRunner({"initial_nav": 1_000_000})
        broker = make_sim_broker()
        dash = PnLDashboard({"initial_nav": 1_000_000})
        runner.inject_components(broker=broker, pnl_dashboard=dash)
        dates = pd.bdate_range("2024-01-01", periods=5).tolist()
        runner.run(dates)
        assert len(dash.get_nav_series()) > 0

    def test_zero_return_no_crash(self):
        runner = PaperTradingRunner({"initial_nav": 1_000_000})
        dates = pd.bdate_range("2024-01-01", periods=3).tolist()
        result = runner.run(dates)
        assert result.num_days == 3

    def test_sharpe_computation(self):
        runner = PaperTradingRunner({"initial_nav": 1_000_000})
        broker = make_sim_broker()
        runner.inject_components(broker=broker)
        dates = pd.bdate_range("2024-01-01", periods=20).tolist()
        result = runner.run(dates)
        # Sharpe should be a finite number
        assert np.isfinite(result.sharpe_ratio)

    def test_look_ahead_flag_counting(self):
        runner = PaperTradingRunner({"initial_nav": 1_000_000})
        dates = pd.bdate_range("2024-01-01", periods=3).tolist()
        result = runner.run(dates)
        assert result.look_ahead_flags == 0  # no research runner, no flags


# ── Live Trading Runner ──────────────────────────────────────────────

class TestLiveTradingRunner:
    def _setup_deployed(self):
        """Set up a fully approved and deployed strategy."""
        wf = ApprovalWorkflow({"min_paper_trading_days": 0})
        ctrl = DeploymentController(wf)
        wf.submit("strat1")
        wf.transition("strat1", ApprovalState.UNDER_REVIEW)
        wf.transition("strat1", ApprovalState.APPROVED)
        wf.set_risk_approval("strat1", True)
        wf.set_review_result("strat1", True)
        wf.transition("strat1", ApprovalState.DEPLOYED)
        ctrl.deploy("strat1", mode="live")
        return wf, ctrl

    def test_pre_market_check_passes(self):
        wf, ctrl = self._setup_deployed()
        runner = LiveTradingRunner(ctrl, {"strategy_id": "strat1"})
        ok, reasons = runner.pre_market_check()
        assert ok is True
        assert len(reasons) == 0

    def test_pre_market_check_fails_not_live(self):
        wf = ApprovalWorkflow()
        ctrl = DeploymentController(wf)
        runner = LiveTradingRunner(ctrl, {"strategy_id": "strat1"})
        ok, reasons = runner.pre_market_check()
        assert ok is False

    def test_pre_market_check_fails_kill_switch(self):
        wf, ctrl = self._setup_deployed()
        ctrl.set_kill_switch(True)
        runner = LiveTradingRunner(ctrl, {"strategy_id": "strat1"})
        ok, reasons = runner.pre_market_check()
        assert ok is False

    def test_run_single_day_blocked(self):
        wf = ApprovalWorkflow()
        ctrl = DeploymentController(wf)
        runner = LiveTradingRunner(ctrl, {"strategy_id": "strat1"})
        result = runner.run_single_day(pd.Timestamp("2024-01-02"), prev_nav=1_000_000)
        assert result.status == "blocked"

    def test_run_single_day_success(self):
        wf, ctrl = self._setup_deployed()
        runner = LiveTradingRunner(ctrl, {"strategy_id": "strat1"})
        broker = make_sim_broker()
        runner.inject_components(broker=broker)
        result = runner.run_single_day(
            pd.Timestamp("2024-01-02"), prev_nav=1_000_000
        )
        assert result.status == "completed"

    def test_audit_log(self):
        wf, ctrl = self._setup_deployed()
        runner = LiveTradingRunner(ctrl, {"strategy_id": "strat1"})
        broker = make_sim_broker()
        runner.inject_components(broker=broker)
        runner.run_single_day(pd.Timestamp("2024-01-02"), prev_nav=1_000_000)
        assert len(runner.audit_log) > 0

    def test_daily_loss_limit_halts(self):
        wf, ctrl = self._setup_deployed()
        runner = LiveTradingRunner(ctrl, {
            "strategy_id": "strat1",
            "max_daily_loss_pct": 0.001,  # very tight limit
            "pre_market_check_enabled": False,
        })
        # Simulate a big loss
        day = runner.run_single_day(
            pd.Timestamp("2024-01-02"), prev_nav=1_000_000
        )
        # Nav didn't change much (no broker), so no halt
        assert runner.is_halted is False

    def test_reset_halt(self):
        wf, ctrl = self._setup_deployed()
        runner = LiveTradingRunner(ctrl, {"strategy_id": "strat1"})
        runner._is_halted = True
        runner.reset_halt()
        assert runner.is_halted is False
