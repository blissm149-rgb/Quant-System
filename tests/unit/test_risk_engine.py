"""Unit tests for risk engine modules.

TESTING_PLAN.md Section 3.9 — drawdown_monitor, exposure_monitor,
leverage_controller, stress_test_engine.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.risk_engine.drawdown_monitor import (
    DrawdownAlertLevel,
    DrawdownMonitor,
)
from quant_fund.risk_engine.exposure_monitor import (
    ExposureBreachAlert,
    ExposureMonitor,
)
from quant_fund.risk_engine.leverage_controller import LeverageController
from quant_fund.risk_engine.stress_test_engine import StressTestEngine, StressTestResult


@pytest.mark.unit
@pytest.mark.tier1
class TestDrawdownMonitor:
    """DrawdownMonitor — tiered drawdown alerts."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor(config={"initial_nav": 1_000_000})

    def test_10_percent_warning(self, monitor):
        """10% drawdown fires WARNING alert."""
        alerts = monitor.update(900_000)
        assert any(a.level == DrawdownAlertLevel.WARNING for a in alerts)

    def test_15_percent_alert(self, monitor):
        """15% drawdown fires ALERT level."""
        alerts = monitor.update(850_000)
        assert any(a.level == DrawdownAlertLevel.ALERT for a in alerts)

    def test_20_percent_critical(self, monitor):
        """20% drawdown fires CRITICAL alert."""
        alerts = monitor.update(800_000)
        assert any(a.level == DrawdownAlertLevel.CRITICAL for a in alerts)

    def test_no_alert_below_10_percent(self, monitor):
        """Drawdown below 10% fires no alerts."""
        alerts = monitor.update(950_000)
        assert len(alerts) == 0

    def test_peak_tracking(self, monitor):
        """Peak NAV updates correctly."""
        monitor.update(1_100_000)
        assert monitor.peak_nav == 1_100_000
        monitor.update(1_050_000)
        assert monitor.peak_nav == 1_100_000  # peak doesn't decrease


@pytest.mark.unit
@pytest.mark.tier1
class TestExposureMonitor:
    """ExposureMonitor — checks exposure limits."""

    @pytest.fixture
    def monitor(self):
        return ExposureMonitor(config={
            "max_leverage": 2.0,
            "max_sector_exposure": 0.20,
            "max_single_name_exposure": 0.02,
        })

    def test_net_exposure_within_limits(self, monitor):
        """Compliant portfolio produces no breaches."""
        weights = pd.Series({
            "AAPL": 0.01, "MSFT": -0.01, "GOOG": 0.005, "JPM": -0.005,
        })
        breaches = monitor.check(weights)
        assert len(breaches) == 0

    def test_gross_exposure_breach(self, monitor):
        """Gross leverage breach detected."""
        weights = pd.Series({"AAPL": 1.5, "MSFT": -1.5})
        breaches = monitor.check(weights)
        assert any("leverage" in b.message.lower() or "gross" in b.message.lower() for b in breaches)

    def test_single_name_breach(self, monitor):
        """Single name exceeding limit detected."""
        weights = pd.Series({"AAPL": 0.05, "MSFT": -0.05})
        breaches = monitor.check(weights)
        assert any("single" in b.message.lower() or "position" in b.message.lower() or "name" in b.message.lower() for b in breaches)

    def test_sector_exposure_breach(self, monitor):
        """Sector exposure exceeding 20% detected."""
        # 11 tickers at 0.02 each = 0.22 sector exposure > 0.20 limit
        tickers = {f"TECH{i}": 0.02 for i in range(11)}
        weights = pd.Series(tickers)
        sector_map = {t: "Technology" for t in weights.index}
        breaches = monitor.check(weights, sector_map=sector_map)
        assert any("sector" in b.exposure_type.lower() for b in breaches)


@pytest.mark.unit
@pytest.mark.tier1
class TestLeverageController:
    """LeverageController — proportional scaling."""

    @pytest.fixture
    def controller(self):
        return LeverageController(config={"max_leverage": 2.0})

    def test_scales_down_excess_leverage(self, controller):
        """Weights exceeding max leverage are scaled down proportionally."""
        weights = pd.Series({"AAPL": 2.0, "MSFT": -2.0})  # gross = 4.0
        enforced = controller.enforce(weights)
        assert enforced.abs().sum() <= 2.0 + 1e-6

    def test_does_not_scale_within_limit(self, controller):
        """Weights within limit are not modified."""
        weights = pd.Series({"AAPL": 0.5, "MSFT": -0.5})  # gross = 1.0
        enforced = controller.enforce(weights)
        pd.testing.assert_series_equal(enforced, weights)

    def test_exactly_at_limit_passes(self, controller):
        """Weights at exactly max leverage pass unchanged."""
        weights = pd.Series({"AAPL": 1.0, "MSFT": -1.0})  # gross = 2.0
        enforced = controller.enforce(weights)
        np.testing.assert_allclose(enforced.abs().sum(), 2.0, atol=1e-6)

    def test_preserves_direction(self, controller):
        """Scaling preserves long/short direction."""
        weights = pd.Series({"AAPL": 2.0, "MSFT": -2.0})
        enforced = controller.enforce(weights)
        assert enforced["AAPL"] > 0
        assert enforced["MSFT"] < 0


@pytest.mark.unit
@pytest.mark.tier2
class TestStressTestEngine:
    """StressTestEngine — historical scenario analysis."""

    @pytest.fixture
    def engine(self):
        return StressTestEngine()

    def test_run_all_returns_results(self, engine):
        """run_all returns list of StressTestResult."""
        weights = pd.Series({"AAPL": 0.01, "MSFT": -0.01})
        factors = ["Market", "Size", "Value"]
        exposures = pd.DataFrame(
            np.random.default_rng(42).standard_normal((2, 3)),
            index=["AAPL", "MSFT"], columns=factors,
        )
        results = engine.run_all(weights, exposures)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, StressTestResult) for r in results)

    def test_2008_crisis_scenario(self, engine):
        """2008 crisis scenario produces negative portfolio return."""
        weights = pd.Series({"AAPL": 0.02, "MSFT": 0.02})
        exposures = pd.DataFrame(
            [[1.0, 0.5, 0.3], [0.8, 0.3, 0.2]],
            index=["AAPL", "MSFT"], columns=["Market", "Size", "Value"],
        )
        scenario = {
            "description": "2008 crisis test",
            "shocks": {"Market": -0.40, "Size": -0.20, "Value": -0.25},
        }
        result = engine.run_scenario("2008_test", scenario, weights, exposures)
        assert result.portfolio_return < 0
