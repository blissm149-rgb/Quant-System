"""Failure mode tests — graceful degradation under adverse conditions.

Tests that the system handles NaN injection, missing tickers, broker
rejections, stale data, config corruption, and zero NAV without crashing.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import STANDARD_TICKERS, STANDARD_MARKET_DATA

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.config.config_validator import (
    ConfigValidator,
    ConfigurationError,
)
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.execution.order_management.order_safety_validator import (
    OrderSafetyValidator,
)
from quant_fund.main.paper_trading_runner import PaperTradingRunner
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.risk_engine.leverage_controller import LeverageController
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch


def _broker(cash=1_000_000.0):
    b = SimulationBroker({"initial_cash": cash, "enforce_cash_floor": True})
    b.set_market_data(STANDARD_MARKET_DATA)
    return b


# ── NaN Injection ──────────────────────────────────────────────────────


class TestNaNInjection:
    """Test handling of NaN values at various pipeline stages."""

    def test_leverage_controller_nan_weights(self):
        """Leverage controller with NaN weights should not crash."""
        ctrl = LeverageController({"max_leverage": 2.0})
        weights = pd.Series([0.3, np.nan, 0.2], index=["A", "B", "C"])
        result = ctrl.enforce(weights)
        assert isinstance(result, pd.Series)

    def test_order_generator_nan_price_filtered(self):
        """Order generator should handle pre-filtered NaN prices."""
        gen = OrderGenerator()
        weights = pd.Series([0.3], index=["MSFT"])
        prices = pd.Series([300.0], index=["MSFT"])
        orders = gen.generate_orders(
            target_weights=weights,
            current_positions=pd.Series(dtype=float),
            prices=prices,
            nav=1_000_000,
        )
        assert isinstance(orders, list)
        assert len(orders) > 0

    def test_pnl_dashboard_nan_nav(self):
        """PnL dashboard should handle NaN NAV gracefully."""
        pnl = PnLDashboard({"initial_nav": 1_000_000})
        # Normal update first
        pnl.update(1_000_000, timestamp=pd.Timestamp("2024-01-01"))
        # NaN update — should not crash
        try:
            pnl.update(float("nan"), timestamp=pd.Timestamp("2024-01-02"))
        except (ValueError, TypeError):
            pass  # Acceptable to raise on NaN
        # Dashboard should still be usable
        assert pnl.latest is not None

    def test_kill_switch_nan_nav(self):
        """Kill switch should handle NaN NAV."""
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        # NaN should not trigger (or at least not crash)
        result = ks.check(float("nan"))
        assert isinstance(result, bool)


# ── Missing Tickers ───────────────────────────────────────────────────


class TestMissingTickers:
    """Test handling of unknown or missing tickers."""

    def test_broker_rejects_unknown_ticker(self):
        """Broker should reject orders for tickers without market data."""
        broker = _broker()
        order = Order(
            ticker="ZZZZ", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        ack = broker.submit_order(order)
        assert ack.status == OrderStatus.REJECTED

    def test_order_generator_empty_prices(self):
        """Order generator with empty prices should produce no orders."""
        gen = OrderGenerator()
        orders = gen.generate_orders(
            target_weights=pd.Series([0.5], index=["AAPL"]),
            current_positions=pd.Series(dtype=float),
            prices=pd.Series(dtype=float),
            nav=1_000_000,
        )
        assert len(orders) == 0

    def test_broker_market_data_partial(self):
        """Broker handles request for mix of known/unknown tickers."""
        broker = _broker()
        md = broker.get_market_data(["AAPL", "UNKNOWN"])
        assert "AAPL" in md.index
        assert md.loc["AAPL", "mid"] == 150.0


# ── Broker Rejection ─────────────────────────────────────────────────


class TestBrokerRejection:
    """Test handling when broker rejects all orders."""

    def test_paper_trading_survives_all_rejections(self):
        """Paper trading runner should not crash when all orders fail."""
        runner = PaperTradingRunner({"initial_nav": 100})
        broker = _broker(cash=100)  # Very low cash

        class MockResearch:
            def run_cycle(self, as_of, market_data=None):
                class R:
                    alpha_scores = pd.Series(
                        [0.5, 0.5], index=["AAPL", "MSFT"],
                    )
                    validation_flags = []
                return R()

        class MockOptimizer:
            def optimize(self, alpha_scores, **kwargs):
                return alpha_scores / alpha_scores.abs().sum()

        runner.inject_components(
            broker=broker,
            research_runner=MockResearch(),
            portfolio_optimizer=MockOptimizer(),
            order_generator=OrderGenerator({"min_trade_value": 10}),
            order_router=OrderRouter(broker, {"safety_checks_enabled": False}),
        )

        dates = list(pd.bdate_range("2024-01-01", periods=3))
        result = runner.run(dates)
        # Should complete without crashing
        assert result.num_days == 3
        assert all(d.status != "failed" for d in result.daily_results)


# ── Config Corruption ────────────────────────────────────────────────


class TestConfigCorruption:
    """Test that config validator catches dangerous values."""

    def test_negative_leverage(self):
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate_all(
                {"trading": {"max_leverage": -5}}
            )

    def test_zero_drawdown(self):
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate_all(
                {"risk": {"drawdown_limit": 0}}
            )

    def test_drawdown_over_one(self):
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate_all(
                {"risk": {"drawdown_limit": 2.0}}
            )

    def test_zero_initial_nav(self):
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate_all(
                {"system": {"initial_nav": 0}}
            )

    def test_excessive_position_size(self):
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate_all(
                {"trading": {"max_position_size": 0.8}}
            )

    def test_excessive_participation_rate(self):
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate_all(
                {"execution": {"participation_rate": 0.9}}
            )


# ── Zero NAV ─────────────────────────────────────────────────────────


class TestZeroNAV:
    """Test handling of zero or negative NAV edge cases."""

    def test_pnl_dashboard_zero_initial(self):
        """PnL dashboard with zero initial NAV should not divide by zero."""
        pnl = PnLDashboard({"initial_nav": 0.0})
        snap = pnl.update(100.0, timestamp=pd.Timestamp("2024-01-01"))
        assert snap is not None

    def test_kill_switch_zero_peak(self):
        """Kill switch with zero peak NAV should not divide by zero."""
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 0})
        result = ks.check(100.0)
        assert isinstance(result, bool)

    def test_paper_runner_zero_nav(self):
        """Paper trading runner with zero initial NAV."""
        runner = PaperTradingRunner({"initial_nav": 0})
        broker = _broker()
        runner.inject_components(broker=broker)
        dates = list(pd.bdate_range("2024-01-01", periods=2))
        result = runner.run(dates)
        assert result.num_days == 2


# ── Safety Validator Edge Cases ──────────────────────────────────────


class TestSafetyEdgeCases:
    """Test safety validator boundary conditions."""

    def test_exactly_at_limit(self):
        """Order exactly at the limit should pass."""
        validator = OrderSafetyValidator({"max_shares_per_order": 1000})
        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=1000,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        result = validator.validate(order)
        assert result.passed

    def test_one_over_limit(self):
        """Order one share over the limit should fail."""
        validator = OrderSafetyValidator({"max_shares_per_order": 1000})
        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=1001,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        result = validator.validate(order)
        assert not result.passed

    def test_multiple_rejections_accumulated(self):
        """Order violating multiple checks should list all rejections."""
        validator = OrderSafetyValidator({
            "max_shares_per_order": 100,
            "max_order_pct_nav": 0.01,
        })
        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=200,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        result = validator.validate(order, nav=1000, mid_price=150.0)
        assert not result.passed
        assert len(result.rejections) >= 2  # max_shares + max_pct_nav
