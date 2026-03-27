"""Unit tests for RiskCascadeCoordinator.

Module D: Verifies unified risk cascade with kill switch, drawdown,
leverage, and exposure checks in defined order.
"""

import pandas as pd
import pytest

from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor
from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
from quant_fund.risk_engine.leverage_controller import LeverageController
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from quant_fund.risk_engine.risk_cascade_coordinator import (
    RiskCascadeCoordinator,
    RiskCascadeResult,
)


@pytest.mark.unit
@pytest.mark.tier2
class TestRiskCascadeCoordinator:
    """RiskCascadeCoordinator — unified risk pipeline."""

    @pytest.fixture
    def clean_weights(self):
        return pd.Series({
            "AAPL": 0.01, "MSFT": 0.01, "GOOG": 0.01,
            "JPM": 0.01, "BAC": 0.01,
        })

    def test_cascade_kill_switch_stops_everything(self):
        """Kill switch triggered → orders blocked, kill_switch_triggered=True."""
        ks = KillSwitch(config={"drawdown_limit": 0.10, "initial_nav": 1_000_000})

        coord = RiskCascadeCoordinator(kill_switch=ks)
        weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02})

        # NAV dropped 15% from peak — exceeds 10% limit
        result = coord.run_cascade(850_000, weights)

        assert result.kill_switch_triggered is True
        assert result.orders_blocked is True

    def test_cascade_leverage_enforced(self):
        """Over-leveraged weights are scaled down."""
        lc = LeverageController(config={"max_leverage": 1.0})
        coord = RiskCascadeCoordinator(leverage_controller=lc)

        weights = pd.Series({"AAPL": 1.0, "MSFT": 1.0, "GOOG": 1.0})
        result = coord.run_cascade(1_000_000, weights)

        assert result.leverage_enforced is True
        assert result.original_gross == pytest.approx(3.0)
        assert result.adjusted_gross <= 1.0 + 1e-9
        assert result.orders_blocked is False

    def test_cascade_exposure_breach_blocks_orders(self):
        """Sector overconcentration → orders blocked."""
        em = ExposureMonitor(config={"max_single_name_exposure": 0.02})
        coord = RiskCascadeCoordinator(exposure_monitor=em)

        weights = pd.Series({"AAPL": 0.50, "MSFT": 0.01})
        result = coord.run_cascade(1_000_000, weights)

        assert result.orders_blocked is True
        assert len(result.exposure_breaches) > 0

    def test_cascade_clean_weights_pass_through(self, clean_weights):
        """Clean weights → no blocks, weights unchanged."""
        lc = LeverageController(config={"max_leverage": 2.0})
        em = ExposureMonitor(config={
            "max_single_name_exposure": 0.10,
            "max_leverage": 2.0,
        })
        coord = RiskCascadeCoordinator(
            leverage_controller=lc, exposure_monitor=em,
        )

        result = coord.run_cascade(1_000_000, clean_weights)

        assert result.orders_blocked is False
        assert result.leverage_enforced is False
        assert result.adjusted_weights is not None
        pd.testing.assert_series_equal(result.adjusted_weights, clean_weights)

    def test_cascade_order_kill_switch_before_leverage(self):
        """Kill switch fires → leverage_controller is NOT called (short-circuit)."""
        ks = KillSwitch(config={"drawdown_limit": 0.10, "initial_nav": 1_000_000})

        lc = LeverageController(config={"max_leverage": 1.0})
        coord = RiskCascadeCoordinator(
            kill_switch=ks, leverage_controller=lc,
        )

        weights = pd.Series({"AAPL": 2.0, "MSFT": 2.0})
        result = coord.run_cascade(850_000, weights)

        # Kill switch triggered — leverage was NOT enforced
        assert result.kill_switch_triggered is True
        assert result.leverage_enforced is False

    def test_cascade_drawdown_alerts_propagated(self):
        """Drawdown alerts are captured in the result."""
        dm = DrawdownMonitor(config={
            "initial_nav": 1_000_000,
            "drawdown_warning": 0.05,
        })
        # Push peak up then drop
        dm.update(1_000_000)

        coord = RiskCascadeCoordinator(drawdown_monitor=dm)
        weights = pd.Series({"AAPL": 0.02})

        # 8% drawdown from peak (> 5% warning threshold)
        result = coord.run_cascade(920_000, weights)

        assert len(result.drawdown_alerts) > 0
        assert result.orders_blocked is False

    def test_factor_exposure_tracking(self):
        """track_factor_exposure stores snapshots, retrievable as DataFrame."""
        coord = RiskCascadeCoordinator()

        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        factors = pd.DataFrame(
            {"market": [1.0, 1.2], "value": [0.3, -0.1]},
            index=["AAPL", "MSFT"],
        )

        for i in range(10):
            coord.track_factor_exposure(
                pd.Timestamp(f"2024-01-{i + 1:02d}"),
                weights, factors,
            )

        history = coord.get_factor_exposure_history()
        assert len(history) == 10
        assert "market" in history.columns
        assert "value" in history.columns

    def test_cascade_with_none_components(self, clean_weights):
        """All components None → no crash, weights unchanged, not blocked."""
        coord = RiskCascadeCoordinator()
        result = coord.run_cascade(1_000_000, clean_weights)

        assert result.orders_blocked is False
        assert result.kill_switch_triggered is False
        assert result.leverage_enforced is False
        pd.testing.assert_series_equal(result.adjusted_weights, clean_weights)
