"""Test drawdown monitoring alerting system.

Validates that the drawdown monitor correctly tracks peak NAV,
computes drawdowns, and fires alerts at the right thresholds.
"""

import pytest
import numpy as np

from quant_fund.risk_engine.drawdown_monitor import (
    DrawdownMonitor,
    DrawdownAlertLevel,
)


pytestmark = [pytest.mark.validation]


class TestDrawdownMonitoring:
    """Verify drawdown monitor threshold alerting."""

    def test_no_alert_below_warning_threshold(self):
        """No alerts should fire when drawdown is below 10%."""
        monitor = DrawdownMonitor(config={"initial_nav": 1_000_000})

        # Drop 5% — below WARNING threshold
        alerts = monitor.update(950_000)
        assert len(alerts) == 0, (
            f"5% drawdown should not trigger alert, got {len(alerts)} alerts"
        )

    def test_warning_alert_at_ten_percent(self):
        """WARNING alert should fire when drawdown hits 10%."""
        monitor = DrawdownMonitor(config={"initial_nav": 1_000_000})

        alerts = monitor.update(900_000)
        assert len(alerts) == 1
        assert alerts[0].level == DrawdownAlertLevel.WARNING

    def test_critical_alert_at_twenty_percent(self):
        """CRITICAL alert should fire when drawdown hits 20%
        (kill switch trigger)."""
        monitor = DrawdownMonitor(config={"initial_nav": 1_000_000})

        alerts = monitor.update(800_000)
        assert len(alerts) == 1
        assert alerts[0].level == DrawdownAlertLevel.CRITICAL
        assert alerts[0].drawdown_pct >= 0.20

    def test_peak_nav_updates_on_new_high(self):
        """Peak NAV should reset when portfolio reaches new highs,
        and drawdown should be measured from the new peak."""
        monitor = DrawdownMonitor(config={"initial_nav": 1_000_000})

        # New high
        monitor.update(1_100_000)
        assert monitor.peak_nav == 1_100_000

        # 10% drawdown from new peak (1.1M -> 990K)
        alerts = monitor.update(990_000)
        assert len(alerts) == 1
        assert alerts[0].level == DrawdownAlertLevel.WARNING

        # Verify drawdown is from the new peak, not original
        expected_dd = (1_100_000 - 990_000) / 1_100_000
        assert abs(monitor.current_drawdown - expected_dd) < 1e-10
