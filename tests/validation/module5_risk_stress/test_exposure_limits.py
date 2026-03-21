"""Test exposure monitor limit enforcement.

Validates that the exposure monitor correctly detects leverage,
sector concentration, single-name, and factor exposure breaches.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.risk_engine.exposure_monitor import ExposureMonitor


pytestmark = [pytest.mark.validation]


class TestExposureLimits:
    """Verify exposure monitor catches all constraint violations."""

    def test_detects_leverage_breach(self):
        """Gross leverage > 2.0 should produce a breach alert."""
        monitor = ExposureMonitor()
        # |0.8| + |0.7| + |-0.6| + |-0.5| = 2.6
        weights = pd.Series(
            [0.8, 0.7, -0.6, -0.5], index=["A", "B", "C", "D"]
        )

        alerts = monitor.check(weights)
        leverage_alerts = [a for a in alerts if a.exposure_type == "gross_leverage"]

        assert len(leverage_alerts) > 0, (
            "Gross leverage 2.6 should breach 2.0 limit"
        )
        assert leverage_alerts[0].current_value > 2.0

    def test_no_breach_within_limits(self):
        """A well-constrained portfolio should produce no breach alerts."""
        monitor = ExposureMonitor(config={"max_single_name_exposure": 0.10})
        weights = pd.Series(
            [0.05, 0.04, -0.03, -0.04], index=["A", "B", "C", "D"]
        )

        alerts = monitor.check(weights)
        assert len(alerts) == 0, (
            f"Well-constrained portfolio should have no breaches, got {len(alerts)}"
        )

    def test_detects_single_name_concentration(self):
        """Single position > 2% (default) should trigger alert."""
        monitor = ExposureMonitor()
        weights = pd.Series(
            [0.05, 0.01, -0.01, 0.01], index=["AAPL", "MSFT", "GOOG", "AMZN"]
        )

        alerts = monitor.check(weights)
        single_alerts = [a for a in alerts if a.exposure_type == "single_name"]

        assert len(single_alerts) > 0, (
            "5% position in AAPL should breach 2% single-name limit"
        )

    def test_detects_sector_concentration(self):
        """Sector exposure > 20% (default) should trigger alert."""
        monitor = ExposureMonitor()
        weights = pd.Series(
            [0.15, 0.10, 0.01, 0.01],
            index=["AAPL", "MSFT", "JPM", "XOM"],
        )
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JPM": "Financials",
            "XOM": "Energy",
        }

        alerts = monitor.check(weights, sector_map=sector_map)
        sector_alerts = [a for a in alerts if "sector" in a.exposure_type]

        assert len(sector_alerts) > 0, (
            "25% Technology exposure should breach 20% sector limit"
        )
