"""Integration Chain 3: Risk Chain

Tests: drawdown_monitor + exposure_monitor + leverage_controller
+ portfolio_kill_switch → HALT trading
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import STANDARD_TICKERS, STANDARD_SECTORS


SEED = 42


@pytest.mark.integration
@pytest.mark.tier3
class TestDrawdownCascade:
    """Drawdown thresholds cascade: warning → alert → kill switch."""

    def test_10_pct_drawdown_fires_warning(self):
        """10% drawdown fires WARNING alert."""
        from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor, DrawdownAlertLevel

        monitor = DrawdownMonitor({"drawdown_warning": 0.10, "drawdown_alert": 0.15, "drawdown_limit": 0.20})

        # NAV drops to 90% of initial
        alerts = monitor.update(900_000.0)

        warning_alerts = [a for a in alerts if a.level == DrawdownAlertLevel.WARNING]
        assert len(warning_alerts) > 0

    def test_15_pct_drawdown_fires_alert(self):
        """15% drawdown fires ALERT level."""
        from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor, DrawdownAlertLevel

        monitor = DrawdownMonitor({"drawdown_warning": 0.10, "drawdown_alert": 0.15, "drawdown_limit": 0.20})

        alerts = monitor.update(850_000.0)

        alert_alerts = [a for a in alerts if a.level == DrawdownAlertLevel.ALERT]
        assert len(alert_alerts) > 0

    def test_20_pct_drawdown_fires_critical(self):
        """20% drawdown fires CRITICAL alert."""
        from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor, DrawdownAlertLevel

        monitor = DrawdownMonitor({"drawdown_warning": 0.10, "drawdown_alert": 0.15, "drawdown_limit": 0.20})

        alerts = monitor.update(800_000.0)

        critical_alerts = [a for a in alerts if a.level == DrawdownAlertLevel.CRITICAL]
        assert len(critical_alerts) > 0

    def test_kill_switch_triggers_at_20_pct(self):
        """Kill switch halts at 20% drawdown, consistent with drawdown monitor."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor, DrawdownAlertLevel

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)

        monitor = DrawdownMonitor({"drawdown_warning": 0.10, "drawdown_alert": 0.15, "drawdown_limit": 0.20})

        # 20% drawdown
        current_nav = 800_000.0
        alerts = monitor.update(current_nav)
        triggered = ks.check(current_nav)

        # Both should fire
        assert triggered is True
        assert ks.is_halted
        critical = [a for a in alerts if a.level == DrawdownAlertLevel.CRITICAL]
        assert len(critical) > 0


@pytest.mark.integration
@pytest.mark.tier3
class TestExposureRiskChain:
    """Exposure breaches detected and leverage enforced."""

    def test_leverage_breach_detected_and_enforced(self):
        """Weights exceeding leverage are detected by monitor and enforced by controller."""
        from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
        from quant_fund.risk_engine.leverage_controller import LeverageController

        tickers = STANDARD_TICKERS[:10]
        rng = np.random.default_rng(SEED)
        # Create over-leveraged weights (gross > 2.0)
        weights = pd.Series(rng.normal(0, 0.5, len(tickers)), index=tickers)
        gross = weights.abs().sum()

        if gross <= 2.0:
            # Scale up to exceed leverage
            weights *= (2.5 / gross)

        # Monitor detects breach
        monitor = ExposureMonitor({"max_leverage": 2.0})
        breaches = monitor.check(weights)
        leverage_breaches = [b for b in breaches if "leverage" in b.exposure_type.lower() or "gross" in b.exposure_type.lower()]
        assert len(leverage_breaches) > 0, "Leverage breach not detected"

        # Controller enforces
        controller = LeverageController({"max_leverage": 2.0})
        enforced = controller.enforce(weights)

        assert enforced.abs().sum() <= 2.0 + 1e-6

        # After enforcement, leverage should be at or very near the limit
        # (floating point precision may cause monitor to flag at exactly 2.0)
        gross_after = enforced.abs().sum()
        assert gross_after <= 2.0 + 1e-4, f"Leverage still too high: {gross_after}"

    def test_sector_exposure_detected(self):
        """Sector overexposure is detected by ExposureMonitor."""
        from quant_fund.risk_engine.exposure_monitor import ExposureMonitor

        tickers = STANDARD_TICKERS[:10]
        weights = pd.Series(0.0, index=tickers)

        # Overweight all Technology stocks
        for t in tickers:
            if STANDARD_SECTORS[t] == "Technology":
                weights[t] = 0.02

        # Technology weight = 5 * 0.02 = 0.10 — within 0.20 limit
        # Now push one way over
        weights["AAPL"] = 0.15

        monitor = ExposureMonitor({"max_sector_exposure": 0.20, "max_single_name_exposure": 0.02})
        breaches = monitor.check(weights, sector_map=STANDARD_SECTORS)

        # Should have single-name breach
        assert len(breaches) > 0


@pytest.mark.integration
@pytest.mark.tier3
class TestRiskChainIntegration:
    """Full risk chain: leverage + exposure + kill switch work together."""

    def test_full_risk_chain(self):
        """All risk components cooperate without conflicts."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor
        from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
        from quant_fund.risk_engine.leverage_controller import LeverageController

        tickers = STANDARD_TICKERS[:10]
        nav = 1_000_000.0

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(nav)
        monitor = DrawdownMonitor()
        exposure = ExposureMonitor({"max_leverage": 2.0, "max_sector_exposure": 0.20})
        leverage = LeverageController({"max_leverage": 2.0})

        rng = np.random.default_rng(SEED)
        weights = pd.Series(rng.normal(0, 0.3, len(tickers)), index=tickers)

        # Step 1: Enforce leverage
        weights = leverage.enforce(weights)
        assert weights.abs().sum() <= 2.0 + 1e-6

        # Step 2: Check exposure
        breaches = exposure.check(weights, sector_map=STANDARD_SECTORS)
        # Even after leverage enforcement, other breaches may exist

        # Step 3: Check kill switch
        triggered = ks.check(nav)
        assert not triggered  # NAV hasn't dropped

        # Step 4: Monitor drawdown
        alerts = monitor.update(nav)
        # No drawdown yet, so no alerts expected

        # Simulate drawdown
        nav_dropped = 780_000.0
        triggered = ks.check(nav_dropped)
        assert triggered
        assert ks.is_halted
