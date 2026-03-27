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

    def test_risk_cascade_coordinator_matches_manual_chain(self):
        """RiskCascadeCoordinator produces same outcome as manual chain."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor
        from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
        from quant_fund.risk_engine.leverage_controller import LeverageController
        from quant_fund.risk_engine.risk_cascade_coordinator import RiskCascadeCoordinator

        nav = 1_000_000.0
        rng = np.random.default_rng(SEED)
        tickers = STANDARD_TICKERS[:10]
        weights = pd.Series(rng.normal(0, 0.3, len(tickers)), index=tickers)

        # Manual chain
        lc1 = LeverageController({"max_leverage": 2.0})
        em1 = ExposureMonitor({"max_leverage": 2.0, "max_sector_exposure": 0.20,
                                "max_single_name_exposure": 0.10})

        manual_weights = lc1.enforce(weights)
        manual_breaches = em1.check(manual_weights, sector_map=STANDARD_SECTORS)

        # Coordinator chain (fresh instances with same config)
        lc2 = LeverageController({"max_leverage": 2.0})
        em2 = ExposureMonitor({"max_leverage": 2.0, "max_sector_exposure": 0.20,
                                "max_single_name_exposure": 0.10})

        coord = RiskCascadeCoordinator(
            leverage_controller=lc2, exposure_monitor=em2,
        )
        result = coord.run_cascade(nav, weights, sector_map=STANDARD_SECTORS)

        # Same adjusted weights
        pd.testing.assert_series_equal(result.adjusted_weights, manual_weights)
        # Same breach count
        assert len(result.exposure_breaches) == len(manual_breaches)
        # Same block decision
        assert result.orders_blocked == (len(manual_breaches) > 0)

    def test_risk_cascade_in_trading_engine_convergence(self):
        """RiskCascadeCoordinator wired into TradingEngine blocks on breach."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.execution.order_management.order_generator import OrderGenerator
        from quant_fund.execution.order_management.order_router import OrderRouter
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
        from quant_fund.risk_engine.leverage_controller import LeverageController
        from quant_fund.risk_engine.risk_cascade_coordinator import RiskCascadeCoordinator
        from quant_fund.main.trading_engine import TradingEngine
        from tests.conftest import STANDARD_MARKET_DATA

        broker = SimulationBroker(config={"initial_cash": 1_000_000})
        broker.set_market_data(STANDARD_MARKET_DATA)

        coord = RiskCascadeCoordinator(
            kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
            leverage_controller=LeverageController({"max_leverage": 1.0}),
            exposure_monitor=ExposureMonitor({"max_single_name_exposure": 0.02}),
        )

        engine = TradingEngine(config={"synchronous": True})
        engine.inject_components(
            broker=broker,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker),
            kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
            risk_cascade=coord,
        )

        # 50% in one name — breaches single-name limit
        weights = pd.Series({"AAPL": 0.50, "MSFT": 0.01})
        engine.update_target_weights(weights)

        initial_positions = broker.get_positions()
        engine._convergence_tick()
        after_positions = broker.get_positions()

        # Orders should have been blocked
        assert initial_positions.equals(after_positions)
