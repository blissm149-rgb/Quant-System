"""Unit tests for monitoring subsystem.

TESTING_PLAN.md Section 3.13 — PnLDashboard, AlertingSystem,
SystemHealthMonitor, ExecutionQualityMonitor.
"""

import pandas as pd
import pytest

from quant_fund.monitoring.pnl_dashboard import PnLDashboard, PnLSnapshot
from quant_fund.monitoring.alerting_system import (
    Alert,
    AlertingSystem,
    AlertLevel,
)
from quant_fund.monitoring.system_health_monitor import (
    HealthCheck,
    SystemHealthMonitor,
    SystemHealthSnapshot,
)
from quant_fund.monitoring.execution_quality_monitor import (
    ExecutionQualityMonitor,
    ExecutionSummary,
)


# ── PnL Dashboard ──────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier2
class TestPnLDashboard:
    """PnLDashboard — NAV, returns, drawdown tracking."""

    @pytest.fixture
    def dash(self):
        return PnLDashboard(config={"initial_nav": 1_000_000})

    def test_update_returns_snapshot(self, dash):
        snap = dash.update(1_010_000)
        assert isinstance(snap, PnLSnapshot)
        assert snap.nav == 1_010_000

    def test_daily_return_calculated(self, dash):
        snap = dash.update(1_010_000)
        assert snap.daily_return == pytest.approx(0.01, abs=1e-6)

    def test_drawdown_from_peak(self, dash):
        dash.update(1_100_000)
        snap = dash.update(1_050_000)
        expected_dd = (1_100_000 - 1_050_000) / 1_100_000
        assert snap.drawdown == pytest.approx(expected_dd, abs=1e-6)

    def test_cumulative_return(self, dash):
        dash.update(1_100_000)
        snap = dash.update(1_200_000)
        assert snap.cumulative_return == pytest.approx(0.20, abs=1e-6)

    def test_get_nav_series(self, dash):
        dash.update(1_010_000)
        dash.update(1_020_000)
        series = dash.get_nav_series()
        assert len(series) == 2

    def test_get_summary(self, dash):
        dash.update(1_010_000)
        dash.update(1_020_000)
        summary = dash.get_summary()
        assert "total_return" in summary
        assert "sharpe_ratio" in summary

    def test_strategy_pnl_attribution(self, dash):
        snap = dash.update(1_010_000, strategy_pnl={"alpha_1": 5000, "alpha_2": 5000})
        assert snap.strategy_pnl == {"alpha_1": 5000, "alpha_2": 5000}

    def test_latest_property(self, dash):
        assert dash.latest is None
        dash.update(1_010_000)
        assert dash.latest is not None


# ── Alerting System ────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier1
class TestAlertingSystem:
    """AlertingSystem — alert dispatch, acknowledge, history."""

    @pytest.fixture
    def alerting(self):
        return AlertingSystem(config={"suppress_duplicates_seconds": 0})

    def test_send_returns_alert(self, alerting):
        alert = alerting.send(AlertLevel.INFO, "test", "Test alert")
        assert isinstance(alert, Alert)
        assert alert.level == AlertLevel.INFO

    def test_handler_receives_alert(self, alerting):
        received = []
        alerting.register_handler(AlertLevel.INFO, lambda a: received.append(a))
        alerting.send(AlertLevel.INFO, "test", "Hello")
        assert len(received) == 1

    def test_critical_alert_reaches_all_levels(self, alerting):
        info_received = []
        alerting.register_handler(AlertLevel.INFO, lambda a: info_received.append(a))
        alerting.send(AlertLevel.CRITICAL, "risk", "Kill switch")
        assert len(info_received) == 1

    def test_acknowledge_alert(self, alerting):
        alert = alerting.send(AlertLevel.WARNING, "test", "Ack me")
        assert alerting.acknowledge(alert.alert_id) is True
        assert len(alerting.get_active_alerts()) == 0

    def test_critical_count(self, alerting):
        alerting.send(AlertLevel.CRITICAL, "risk", "Problem 1")
        alerting.send(AlertLevel.CRITICAL, "risk", "Problem 2")
        assert alerting.critical_count == 2

    def test_history_filtering(self, alerting):
        alerting.send(AlertLevel.INFO, "a", "info msg")
        alerting.send(AlertLevel.CRITICAL, "b", "critical msg")
        crits = alerting.get_history(level=AlertLevel.CRITICAL)
        assert len(crits) == 1


# ── System Health Monitor ──────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier2
class TestSystemHealthMonitor:
    """SystemHealthMonitor — data feed health, uptime."""

    @pytest.fixture
    def monitor(self):
        return SystemHealthMonitor()

    def test_run_health_check_returns_snapshot(self, monitor):
        snap = monitor.run_health_check()
        assert isinstance(snap, SystemHealthSnapshot)

    def test_healthy_when_no_feeds(self, monitor):
        snap = monitor.run_health_check()
        assert snap.overall_status == "unknown"

    def test_healthy_data_feed(self, monitor):
        monitor.record_data_timestamp("prices", pd.Timestamp.now())
        snap = monitor.run_health_check()
        assert snap.overall_status == "healthy"

    def test_degraded_on_stale_feed(self, monitor):
        stale = pd.Timestamp.now() - pd.Timedelta(minutes=10)
        monitor.record_data_timestamp("prices", stale)
        snap = monitor.run_health_check()
        assert snap.overall_status in ("degraded", "down")

    def test_uptime_positive(self, monitor):
        assert monitor.uptime_seconds > 0

    def test_is_healthy_default(self, monitor):
        assert monitor.is_healthy() is True


# ── Execution Quality Monitor ──────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier2
class TestExecutionQualityMonitor:
    """ExecutionQualityMonitor — shortfall, fill rate, adverse selection."""

    @pytest.fixture
    def eqm(self):
        return ExecutionQualityMonitor()

    def test_record_execution(self, eqm):
        rec = eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=100, filled_qty=100,
            decision_price=150.0, fill_price=150.10,
        )
        assert rec.fill_rate == 1.0

    def test_implementation_shortfall_positive_for_buy(self, eqm):
        rec = eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=100, filled_qty=100,
            decision_price=150.0, fill_price=151.0,
        )
        assert rec.implementation_shortfall_bps > 0

    def test_get_summary(self, eqm):
        eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=100, filled_qty=95,
            decision_price=150.0, fill_price=150.10,
        )
        summary = eqm.get_summary()
        assert isinstance(summary, ExecutionSummary)
        assert summary.num_orders == 1
        assert summary.avg_fill_rate == pytest.approx(0.95, abs=0.01)

    def test_adverse_selection_detection(self, eqm):
        eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=100, filled_qty=100,
            decision_price=100.0, fill_price=110.0,  # 1000 bps shortfall
        )
        flagged = eqm.detect_adverse_selection()
        assert len(flagged) == 1

    def test_records_dataframe(self, eqm):
        eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=100, filled_qty=100,
            decision_price=150.0, fill_price=150.10,
        )
        df = eqm.get_records_dataframe()
        assert len(df) == 1
