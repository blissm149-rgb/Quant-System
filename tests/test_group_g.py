"""Tests for Group G — Risk engine.

Validates:
- Kill switch triggers at exactly 20% drawdown
- Kill switch does not trigger before threshold
- Drawdown monitor emits alerts at correct levels
- Exposure monitor catches single-name, leverage, and sector breaches
- Leverage controller scales weights correctly
- Stress test engine runs scenarios and reports results
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from quant_fund.risk_engine.drawdown_monitor import (
    DrawdownMonitor,
    DrawdownAlertLevel,
)
from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
from quant_fund.risk_engine.leverage_controller import LeverageController
from quant_fund.risk_engine.stress_test_engine import StressTestEngine


TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "BAC", "WMT"]


# ---------------------------------------------------------------------------
# Kill Switch
# ---------------------------------------------------------------------------

class TestKillSwitch:

    def test_triggers_at_20pct_drawdown(self):
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        # 20% drawdown exactly
        assert ks.check(800_000) is True

    def test_does_not_trigger_below_threshold(self):
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        assert ks.check(850_000) is False
        assert ks.check(900_000) is False

    def test_peak_tracking(self):
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        ks.update_peak(1_100_000)
        assert ks.peak_nav == 1_100_000
        # 20% of 1.1M = 220K, so trigger at 880K
        assert ks.check(880_000) is True
        assert ks.check(880_001) is False

    def test_halt_state(self):
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        ks.check(800_000)
        assert ks.is_halted is True

    def test_reset(self):
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        ks.check(800_000)
        assert ks.is_halted
        ks.reset(900_000)
        assert ks.is_halted is False
        assert ks.peak_nav == 900_000

    def test_update_peak_ignored_when_halted(self):
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        ks.check(800_000)  # triggers halt
        ks.update_peak(2_000_000)
        # Peak should NOT update while halted
        assert ks.peak_nav == 1_000_000


# ---------------------------------------------------------------------------
# Drawdown Monitor
# ---------------------------------------------------------------------------

class TestDrawdownMonitor:

    def test_warning_at_10pct(self):
        monitor = DrawdownMonitor({"initial_nav": 1_000_000})
        alerts = monitor.update(900_000)
        assert len(alerts) == 1
        assert alerts[0].level == DrawdownAlertLevel.WARNING

    def test_alert_at_15pct(self):
        monitor = DrawdownMonitor({"initial_nav": 1_000_000})
        alerts = monitor.update(850_000)
        assert len(alerts) == 1
        assert alerts[0].level == DrawdownAlertLevel.ALERT

    def test_critical_at_20pct(self):
        monitor = DrawdownMonitor({"initial_nav": 1_000_000})
        alerts = monitor.update(800_000)
        assert len(alerts) == 1
        assert alerts[0].level == DrawdownAlertLevel.CRITICAL

    def test_no_alert_small_drawdown(self):
        monitor = DrawdownMonitor({"initial_nav": 1_000_000})
        alerts = monitor.update(950_000)
        assert len(alerts) == 0

    def test_peak_updates_on_new_high(self):
        monitor = DrawdownMonitor({"initial_nav": 1_000_000})
        monitor.update(1_100_000)
        assert monitor.peak_nav == 1_100_000


# ---------------------------------------------------------------------------
# Exposure Monitor
# ---------------------------------------------------------------------------

class TestExposureMonitor:

    def test_detects_leverage_breach(self):
        monitor = ExposureMonitor({"max_leverage": 2.0, "max_single_name_exposure": 1.0})
        weights = pd.Series([1.5, -1.5], index=["AAPL", "MSFT"])
        alerts = monitor.check(weights)
        assert any(a.exposure_type == "gross_leverage" for a in alerts)

    def test_detects_single_name_breach(self):
        monitor = ExposureMonitor({"max_single_name_exposure": 0.02})
        weights = pd.Series([0.05, -0.05], index=["AAPL", "MSFT"])
        alerts = monitor.check(weights)
        assert any(a.exposure_type == "single_name" for a in alerts)

    def test_detects_sector_breach(self):
        monitor = ExposureMonitor({
            "max_sector_exposure": 0.20,
            "max_single_name_exposure": 1.0,
            "max_leverage": 10.0,
        })
        # 5 tech stocks at 0.05 each = 0.25 > 0.20
        weights = pd.Series(
            [0.05, 0.05, 0.05, 0.01, 0.05, 0.01, 0.05, -0.13, -0.07, -0.07],
            index=TICKERS,
        )
        sector_map = {
            "AAPL": "Tech", "MSFT": "Tech", "GOOG": "Tech",
            "AMZN": "Consumer", "META": "Tech", "TSLA": "Consumer",
            "NVDA": "Tech", "JPM": "Finance", "BAC": "Finance", "WMT": "Consumer",
        }
        alerts = monitor.check(weights, sector_map=sector_map)
        assert any("sector" in a.exposure_type for a in alerts)

    def test_no_alerts_when_compliant(self):
        monitor = ExposureMonitor({
            "max_leverage": 2.0,
            "max_single_name_exposure": 0.02,
            "max_sector_exposure": 0.20,
        })
        weights = pd.Series(
            [0.01, -0.01, 0.01, -0.01, 0.005, -0.005, 0.01, -0.01, 0.005, -0.005],
            index=TICKERS,
        )
        alerts = monitor.check(weights)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Leverage Controller
# ---------------------------------------------------------------------------

class TestLeverageController:

    def test_scales_down_when_over_limit(self):
        controller = LeverageController({"max_leverage": 2.0})
        weights = pd.Series([1.5, -1.5, 0.5, -0.5], index=["A", "B", "C", "D"])
        adjusted = controller.enforce(weights)
        assert adjusted.abs().sum() <= 2.0 + 1e-6

    def test_no_change_when_under_limit(self):
        controller = LeverageController({"max_leverage": 2.0})
        weights = pd.Series([0.5, -0.5], index=["A", "B"])
        adjusted = controller.enforce(weights)
        pd.testing.assert_series_equal(adjusted, weights)

    def test_preserves_relative_proportions(self):
        controller = LeverageController({"max_leverage": 2.0})
        weights = pd.Series([2.0, -2.0], index=["A", "B"])
        adjusted = controller.enforce(weights)
        ratio = adjusted["A"] / adjusted["B"]
        assert ratio == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Stress Test Engine
# ---------------------------------------------------------------------------

class TestStressTestEngine:

    def test_runs_all_scenarios(self):
        rng = np.random.default_rng(42)
        weights = pd.Series(rng.normal(0, 0.01, 10), index=TICKERS)
        weights -= weights.mean()
        exposures = pd.DataFrame(
            rng.normal(0, 0.5, (10, 3)),
            index=TICKERS,
            columns=["market", "momentum", "value"],
        )
        engine = StressTestEngine()
        results = engine.run_all(weights, exposures)
        assert len(results) == 3
        for r in results:
            assert r.scenario_name in (
                "2008_financial_crisis", "2020_covid_crash", "2022_rate_shock"
            )
            assert isinstance(r.portfolio_return, float)
            assert isinstance(r.factor_pnl, dict)

    def test_stress_with_idiosyncratic(self):
        rng = np.random.default_rng(42)
        weights = pd.Series([0.01, -0.01], index=["AAPL", "MSFT"])
        exposures = pd.DataFrame(
            {"market": [1.0, 0.8]}, index=["AAPL", "MSFT"]
        )
        idio = pd.Series({"AAPL": 0.04, "MSFT": 0.03})
        engine = StressTestEngine({"stress_tests": ["2008_financial_crisis"]})
        results = engine.run_all(weights, exposures, idio)
        assert len(results) == 1
        assert "idiosyncratic" in results[0].factor_pnl
