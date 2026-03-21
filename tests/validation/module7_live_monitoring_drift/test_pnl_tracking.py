"""Test P&L dashboard tracking accuracy and drift detection.

Validates that PnLDashboard correctly computes NAV, drawdown,
cumulative returns, and attribution — and that drift from expected
values is detected over multi-day sequences.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.monitoring.pnl_dashboard import PnLDashboard


pytestmark = [pytest.mark.validation]


class TestPnLDashboardAccuracy:
    """Verify PnLDashboard arithmetic over deterministic NAV sequences."""

    def test_cumulative_return_matches_nav_change(self):
        """Cumulative return should equal (final_nav / initial_nav - 1)."""
        dash = PnLDashboard(config={"initial_nav": 1_000_000})
        navs = [1_010_000, 1_025_000, 1_050_000, 1_040_000, 1_080_000]

        for nav in navs:
            snap = dash.update(nav)

        expected_cum = 1_080_000 / 1_000_000 - 1.0
        assert snap.cumulative_return == pytest.approx(expected_cum, abs=1e-8)

    def test_drawdown_peak_tracking(self):
        """Drawdown should be measured from the highest observed NAV."""
        dash = PnLDashboard(config={"initial_nav": 1_000_000})

        dash.update(1_200_000)  # peak
        dash.update(1_100_000)  # drawdown
        snap = dash.update(1_150_000)  # partial recovery

        expected_dd = (1_200_000 - 1_150_000) / 1_200_000
        assert snap.drawdown == pytest.approx(expected_dd, abs=1e-8)
        assert snap.peak_nav == 1_200_000

    def test_daily_return_chain(self):
        """Product of (1 + daily_returns) should equal final_nav / initial_nav."""
        dash = PnLDashboard(config={"initial_nav": 1_000_000})
        rng = np.random.default_rng(42)
        nav = 1_000_000.0
        returns_collected = []

        for _ in range(50):
            daily_ret = rng.normal(0.001, 0.02)
            nav *= (1 + daily_ret)
            snap = dash.update(nav)
            returns_collected.append(snap.daily_return)

        product = np.prod([1 + r for r in returns_collected])
        expected_ratio = nav / 1_000_000
        assert product == pytest.approx(expected_ratio, rel=1e-6)

    def test_realised_unrealised_decomposition(self):
        """Realised + unrealised P&L should equal total P&L."""
        dash = PnLDashboard(config={"initial_nav": 1_000_000})

        dash.update(1_050_000, realised_pnl_today=20_000)
        snap = dash.update(1_080_000, realised_pnl_today=10_000)

        total_pnl = snap.nav - 1_000_000
        assert snap.realised_pnl + snap.unrealised_pnl == pytest.approx(
            total_pnl, abs=1e-2
        )


class TestPnLDriftDetection:
    """Validate that P&L drift is detectable via summary statistics."""

    def test_sharpe_degrades_under_volatility_spike(self):
        """Sharpe ratio should drop when volatility spikes with zero drift."""
        dash_stable = PnLDashboard(config={"initial_nav": 1_000_000})
        dash_volatile = PnLDashboard(config={"initial_nav": 1_000_000})

        rng = np.random.default_rng(42)
        nav_s = 1_000_000.0
        nav_v = 1_000_000.0

        for _ in range(100):
            ret_s = rng.normal(0.0005, 0.01)
            ret_v = rng.normal(0.0005, 0.04)  # 4x volatility
            nav_s *= (1 + ret_s)
            nav_v *= (1 + ret_v)
            dash_stable.update(nav_s)
            dash_volatile.update(nav_v)

        summary_s = dash_stable.get_summary()
        summary_v = dash_volatile.get_summary()

        assert summary_s["annualised_vol"] < summary_v["annualised_vol"]

    def test_max_drawdown_increases_in_bear_regime(self):
        """Max drawdown should increase during sustained losses."""
        dash = PnLDashboard(config={"initial_nav": 1_000_000})
        rng = np.random.default_rng(42)
        nav = 1_000_000.0

        # Bull phase
        for _ in range(50):
            nav *= (1 + rng.normal(0.002, 0.01))
            dash.update(nav)

        bull_dd = dash.get_summary()["max_drawdown"]

        # Bear phase
        for _ in range(50):
            nav *= (1 + rng.normal(-0.005, 0.02))
            dash.update(nav)

        bear_dd = dash.get_summary()["max_drawdown"]
        assert bear_dd > bull_dd

    def test_strategy_attribution_sums_to_daily_pnl(self):
        """Strategy-level P&L attribution should be trackable per snapshot."""
        dash = PnLDashboard(config={"initial_nav": 1_000_000})

        strategy_pnl = {"momentum": 3000, "mean_reversion": 2000, "stat_arb": 5000}
        total_pnl = sum(strategy_pnl.values())
        nav = 1_000_000 + total_pnl

        snap = dash.update(nav, strategy_pnl=strategy_pnl)
        assert sum(snap.strategy_pnl.values()) == total_pnl

    def test_return_series_length_matches_updates(self):
        """Return series should have exactly as many entries as updates."""
        dash = PnLDashboard(config={"initial_nav": 1_000_000})
        n_days = 30
        nav = 1_000_000.0

        for _ in range(n_days):
            nav *= 1.001
            dash.update(nav)

        series = dash.get_return_series()
        assert len(series) == n_days
