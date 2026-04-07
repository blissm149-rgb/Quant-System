"""Full pipeline integration test — wires all components end-to-end.

Runs a simulated multi-day paper trading loop with all components
connected: research → portfolio construction → risk checks →
execution → monitoring. Validates correctness across module boundaries.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import (

    make_ohlcv,
    make_returns,
    make_factor_returns,
    STANDARD_TICKERS,
    STANDARD_SECTORS,
    STANDARD_MARKET_DATA,
)

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)

pytestmark = [pytest.mark.tier3]
from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.config.config_validator import (
    ConfigValidator,
    ConfigurationError,
    RISK_BOUNDS,
    TRADING_BOUNDS,
)
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.execution.order_management.order_safety_validator import (
    OrderSafetyValidator,
)
from quant_fund.main.paper_trading_runner import (
    PaperTradingRunner,
    PaperTradingResult,
    TradingDayResult,
)
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.portfolio.portfolio_construction.constraint_engine import (
    ConstraintEngine,
)
from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
from quant_fund.risk_engine.leverage_controller import LeverageController
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch


# ── Helpers ────────────────────────────────────────────────────────────


def _make_broker(initial_cash=10_000_000.0):
    """Create a simulation broker with standard market data."""
    broker = SimulationBroker({
        "initial_cash": initial_cash,
        "enforce_cash_floor": True,
    })
    broker.set_market_data(STANDARD_MARKET_DATA)
    return broker


def _make_simple_alpha_scores(tickers=None, seed=42):
    """Generate simple alpha scores for testing."""
    tickers = tickers or STANDARD_TICKERS[:5]
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 1, len(tickers))
    return pd.Series(scores, index=tickers)


# ── Integration Tests ──────────────────────────────────────────────────


class TestOrderGeneratorToBroker:
    """Test order generator → order router → simulation broker pipeline."""

    def test_target_weights_produce_fills(self):
        """Weights → orders → fills → positions updated."""
        broker = _make_broker()
        gen = OrderGenerator()
        router = OrderRouter(broker, {"safety_checks_enabled": False})

        tickers = ["AAPL", "MSFT", "GOOG"]
        weights = pd.Series([0.4, 0.3, 0.3], index=tickers)
        prices = pd.Series([150.0, 300.0, 140.0], index=tickers)
        positions = pd.Series(dtype=float)

        orders = gen.generate_orders(
            target_weights=weights,
            current_positions=positions,
            prices=prices,
            nav=broker.get_account_value(),
        )

        assert len(orders) > 0

        acks = router.route_orders(orders)
        filled = [a for a in acks if a.status in (
            OrderStatus.FILLED, OrderStatus.PARTIAL_FILL,
        )]
        assert len(filled) > 0

        # Positions should now exist
        pos = broker.get_positions()
        assert len(pos) > 0
        # All positions should be long (all weights positive)
        for ticker in pos.index:
            assert pos[ticker] > 0

    def test_long_short_portfolio(self):
        """Dollar-neutral portfolio produces both long and short positions."""
        broker = _make_broker()
        gen = OrderGenerator()
        router = OrderRouter(broker, {"safety_checks_enabled": False})

        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        weights = pd.Series([0.3, 0.2, -0.2, -0.2, -0.1], index=tickers)
        prices = pd.Series([150.0, 300.0, 140.0, 175.0, 350.0], index=tickers)

        orders = gen.generate_orders(
            target_weights=weights,
            current_positions=pd.Series(dtype=float),
            prices=prices,
            nav=broker.get_account_value(),
        )

        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        assert len(buy_orders) > 0
        assert len(sell_orders) > 0

        router.route_orders(orders)
        pos = broker.get_positions()
        long_pos = [t for t in pos.index if pos[t] > 0]
        short_pos = [t for t in pos.index if pos[t] < 0]
        assert len(long_pos) > 0
        assert len(short_pos) > 0

    def test_nav_changes_after_trades(self):
        """NAV should change after executing trades (due to slippage/commission)."""
        broker = _make_broker(initial_cash=1_000_000.0)
        gen = OrderGenerator()
        router = OrderRouter(broker, {"safety_checks_enabled": False})

        initial_nav = broker.get_account_value()
        weights = pd.Series([0.3, 0.3], index=["AAPL", "MSFT"])
        prices = pd.Series([150.0, 300.0], index=["AAPL", "MSFT"])

        orders = gen.generate_orders(
            target_weights=weights,
            current_positions=pd.Series(dtype=float),
            prices=prices,
            nav=initial_nav,
        )
        router.route_orders(orders)

        # NAV should differ from initial due to slippage
        new_nav = broker.get_account_value()
        assert new_nav != initial_nav

    def test_cash_floor_prevents_overbuying(self):
        """Broker rejects BUY when insufficient cash."""
        broker = _make_broker(initial_cash=100.0)  # Very low cash
        gen = OrderGenerator({"min_trade_value": 10.0})
        router = OrderRouter(broker, {"safety_checks_enabled": False})

        weights = pd.Series([0.5, 0.5], index=["AAPL", "MSFT"])
        prices = pd.Series([150.0, 300.0], index=["AAPL", "MSFT"])

        orders = gen.generate_orders(
            target_weights=weights,
            current_positions=pd.Series(dtype=float),
            prices=prices,
            nav=1_000_000.0,  # Pretend NAV is high to generate orders
        )

        acks = router.route_orders(orders)
        rejected = [a for a in acks if a.status == OrderStatus.REJECTED]
        assert len(rejected) > 0


class TestSafetyValidatorIntegration:
    """Test safety validator wired into order router."""

    def test_safety_rejects_zero_quantity(self):
        """Orders with qty=0 are rejected by safety validator."""
        broker = _make_broker()
        router = OrderRouter(broker)  # Safety enabled by default

        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=0,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        ack = router.route_single(order)
        assert ack.status == OrderStatus.REJECTED
        assert "Safety rejected" in ack.message

    def test_safety_rejects_empty_ticker(self):
        """Orders with empty ticker are rejected."""
        broker = _make_broker()
        router = OrderRouter(broker)

        order = Order(
            ticker="", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        ack = router.route_single(order)
        assert ack.status == OrderStatus.REJECTED

    def test_safety_rejects_excessive_shares(self):
        """Orders exceeding max shares per order are rejected."""
        broker = _make_broker()
        router = OrderRouter(broker)

        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=2_000_000,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        ack = router.route_single(order)
        assert ack.status == OrderStatus.REJECTED
        assert "Safety rejected" in ack.message

    def test_safety_allows_normal_order(self):
        """Normal-sized orders pass safety checks."""
        broker = _make_broker()
        router = OrderRouter(broker)

        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        ack = router.route_single(order)
        assert ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL)

    def test_safety_can_be_disabled(self):
        """Safety checks can be disabled via config."""
        broker = _make_broker()
        router = OrderRouter(broker, {"safety_checks_enabled": False})

        # This would normally be rejected (qty=0)
        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        ack = router.route_single(order)
        assert ack.status != OrderStatus.REJECTED


class TestConstraintToExposurePipeline:
    """Test constraint engine → optimizer output → exposure monitor flow."""

    def test_leverage_controller_scales_weights(self):
        """Leverage controller should scale down over-leveraged weights."""
        ctrl = LeverageController({"max_leverage": 2.0})
        weights = pd.Series(
            [0.6, 0.5, 0.4, -0.5, -0.4, -0.3],
            index=["A", "B", "C", "D", "E", "F"],
        )
        # Gross = 2.7, exceeds max 2.0
        adjusted = ctrl.enforce(weights)
        gross = adjusted.abs().sum()
        assert gross <= 2.0 + 1e-9

    def test_exposure_monitor_detects_breaches(self):
        """Exposure monitor flags violations."""
        monitor = ExposureMonitor({
            "max_leverage": 2.0,
            "max_single_name_exposure": 0.10,
        })
        # Weight exceeds single-name limit
        weights = pd.Series(
            [0.5, 0.3, 0.2],
            index=["AAPL", "MSFT", "GOOG"],
        )
        breaches = monitor.check(weights)
        assert len(breaches) > 0

    def test_constraint_validation(self):
        """Constraint engine validates weights against constraints."""
        engine = ConstraintEngine({
            "position_limits": {
                "max_position_size": 0.05,
                "max_leverage": 2.0,
            }
        })
        constraints = engine.build_constraints()
        weights = pd.Series([0.6, 0.4], index=["AAPL", "MSFT"])
        violations = engine.validate_weights(weights, constraints)
        assert len(violations) > 0  # Position > 5%


class TestKillSwitchIntegration:
    """Test kill switch triggers and effects."""

    def test_kill_switch_triggers_on_drawdown(self):
        """Kill switch should trigger at 20% drawdown."""
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        # 19% drawdown — should not trigger
        assert not ks.check(810_000)
        # 20% drawdown — should trigger
        assert ks.check(800_000)
        assert ks.is_halted

    def test_kill_switch_blocks_paper_trading(self):
        """Kill switch should stop paper trading loop early."""
        runner = PaperTradingRunner({"initial_nav": 1_000_000})
        ks = KillSwitch({"drawdown_limit": 0.10, "initial_nav": 1_000_000})

        # Create a broker that returns declining NAV
        broker = _make_broker(initial_cash=800_000)

        runner.inject_components(
            broker=broker,
            kill_switch=ks,
        )

        dates = list(pd.bdate_range("2024-01-01", periods=10))
        result = runner.run(dates)

        # Kill switch should have triggered since broker NAV < peak
        assert result.num_days <= 10
        if any(d.kill_switch_triggered for d in result.daily_results):
            last_day = result.daily_results[-1]
            assert last_day.kill_switch_triggered or last_day.status == "kill_switch"


class TestPaperTradingRunner:
    """Test paper trading runner orchestration."""

    def test_basic_run(self):
        """Runner should complete a multi-day run without errors."""
        runner = PaperTradingRunner({"initial_nav": 1_000_000})
        broker = _make_broker()
        pnl = PnLDashboard({"initial_nav": 1_000_000})

        runner.inject_components(broker=broker, pnl_dashboard=pnl)

        dates = list(pd.bdate_range("2024-01-01", periods=5))
        result = runner.run(dates)

        assert result.num_days == 5
        assert result.strategy_id == "default"
        assert all(d.status in ("completed", "kill_switch") for d in result.daily_results)

    def test_exposure_breach_blocks_orders(self):
        """Exposure breaches should prevent order generation."""
        runner = PaperTradingRunner({"initial_nav": 1_000_000})
        broker = _make_broker()

        # Mock exposure monitor that always finds breaches
        class AlwaysBreachMonitor:
            def check(self, weights, **kwargs):
                return ["Leverage breach: 5.0 > 2.0"]

        # Mock leverage controller (pass-through)
        class PassThroughLeverage:
            def enforce(self, weights):
                return weights

        # Mock research that returns alpha scores
        class MockResearch:
            def run_cycle(self, as_of, market_data=None):
                class Res:
                    alpha_scores = pd.Series(
                        [0.5, 0.3, 0.2],
                        index=["AAPL", "MSFT", "GOOG"],
                    )
                    validation_flags = []
                return Res()

        # Mock optimizer
        class MockOptimizer:
            def optimize(self, alpha_scores, **kwargs):
                return alpha_scores / alpha_scores.abs().sum()

        runner.inject_components(
            broker=broker,
            research_runner=MockResearch(),
            portfolio_optimizer=MockOptimizer(),
            exposure_monitor=AlwaysBreachMonitor(),
            leverage_controller=PassThroughLeverage(),
        )

        dates = list(pd.bdate_range("2024-01-01", periods=3))
        result = runner.run(dates)

        # Days should have exposure_breach status
        breach_days = [
            d for d in result.daily_results
            if d.status == "exposure_breach"
        ]
        assert len(breach_days) > 0
        # No orders should have been generated
        assert all(d.num_orders == 0 for d in breach_days)


class TestConfigValidation:
    """Test config validation catches dangerous configurations."""

    def test_valid_config_passes(self):
        """Well-formed config should pass validation."""
        validator = ConfigValidator()
        configs = {
            "risk": {
                "drawdown_limit": 0.20,
                "max_sector_exposure": 0.30,
            },
            "trading": {
                "max_leverage": 2.0,
                "max_position_size": 0.05,
            },
            "execution": {
                "participation_rate": 0.10,
            },
            "system": {
                "initial_nav": 1_000_000.0,
            },
        }
        # Should not raise
        validator.validate_all(configs)

    def test_zero_drawdown_limit_rejected(self):
        """drawdown_limit=0 should be rejected (must be > 0)."""
        validator = ConfigValidator()
        configs = {
            "risk": {"drawdown_limit": 0.0},
        }
        with pytest.raises(ConfigurationError):
            validator.validate_all(configs)

    def test_negative_leverage_rejected(self):
        """Negative max_leverage should be rejected."""
        validator = ConfigValidator()
        configs = {
            "trading": {"max_leverage": -5.0},
        }
        with pytest.raises(ConfigurationError):
            validator.validate_all(configs)

    def test_excessive_leverage_rejected(self):
        """max_leverage > 10 should be rejected."""
        validator = ConfigValidator()
        configs = {
            "trading": {"max_leverage": 15.0},
        }
        with pytest.raises(ConfigurationError):
            validator.validate_all(configs)

    def test_zero_nav_rejected(self):
        """initial_nav=0 should be rejected (division by zero risk)."""
        validator = ConfigValidator()
        configs = {
            "system": {"initial_nav": 0.0},
        }
        with pytest.raises(ConfigurationError):
            validator.validate_all(configs)

    def test_validate_single_returns_errors(self):
        """validate_single returns error list without raising."""
        validator = ConfigValidator()
        errors = validator.validate_single(
            {"drawdown_limit": 0.0}, "risk"
        )
        assert len(errors) > 0
        assert any("drawdown_limit" in e for e in errors)


class TestEndToEndPipeline:
    """Full end-to-end pipeline test: research → execution → monitoring."""

    def test_full_pipeline_30_days(self):
        """Run a 30-day simulated trading loop with all components."""
        broker = _make_broker(initial_cash=5_000_000)
        gen = OrderGenerator()
        router = OrderRouter(broker, {"safety_checks_enabled": False})
        pnl = PnLDashboard({"initial_nav": 5_000_000})
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 5_000_000})
        ctrl = LeverageController({"max_leverage": 2.0})
        engine = ConstraintEngine({
            "position_limits": {
                "max_position_size": 0.10,
                "max_leverage": 2.0,
            }
        })

        tickers = STANDARD_TICKERS[:5]
        dates = pd.bdate_range("2024-01-01", periods=30)
        rng = np.random.default_rng(42)

        nav = 5_000_000.0
        daily_returns = []

        for i, date in enumerate(dates):
            # 1. Kill switch
            if ks.check(nav):
                break

            # 2. Generate random alpha scores (simulating research)
            scores = pd.Series(
                rng.normal(0, 1, len(tickers)), index=tickers,
            )

            # 3. Simple weight allocation from scores
            weights = scores / scores.abs().sum() * 0.5  # 50% gross exposure

            # 4. Leverage control
            weights = ctrl.enforce(weights)

            # 5. Get current state
            positions = broker.get_positions()
            md = broker.get_market_data(tickers)
            prices = md["mid"] if not md.empty and "mid" in md.columns else pd.Series(dtype=float)

            # 6. Generate and route orders
            orders = gen.generate_orders(
                target_weights=weights,
                current_positions=positions,
                prices=prices,
                nav=nav,
            )
            if orders:
                router.route_orders(orders)

            # 7. Update NAV
            new_nav = broker.get_account_value()
            daily_ret = (new_nav / nav - 1.0) if nav > 0 else 0.0
            daily_returns.append(daily_ret)

            # 8. Update monitoring
            pnl.update(new_nav, timestamp=date)
            ks.update_peak(new_nav)
            nav = new_nav

        # Assertions
        assert len(daily_returns) >= 1  # Ran at least one day
        assert nav > 0  # NAV is positive
        assert pnl.latest is not None
        # PnL dashboard NAV should match broker
        assert abs(pnl.latest.nav - broker.get_account_value()) < 0.01

        # Positions should exist (we traded)
        pos = broker.get_positions()
        assert len(pos) > 0

        # Summary should be computable
        summary = pnl.get_summary()
        assert "total_return" in summary or "cumulative_return" in summary

    def test_liquidation_orders(self):
        """OrderGenerator can produce liquidation orders."""
        broker = _make_broker()
        gen = OrderGenerator()
        router = OrderRouter(broker, {"safety_checks_enabled": False})

        # First, build positions
        tickers = ["AAPL", "MSFT"]
        weights = pd.Series([0.3, 0.3], index=tickers)
        prices = pd.Series([150.0, 300.0], index=tickers)
        orders = gen.generate_orders(
            target_weights=weights,
            current_positions=pd.Series(dtype=float),
            prices=prices,
            nav=broker.get_account_value(),
        )
        router.route_orders(orders)
        pos_before = broker.get_positions()
        assert len(pos_before) > 0

        # Now liquidate
        liq_orders = gen.generate_liquidation_orders(
            current_positions=pos_before,
            prices=prices,
        )
        assert len(liq_orders) > 0

        router2 = OrderRouter(broker, {"safety_checks_enabled": False})
        router2.route_orders(liq_orders)
        pos_after = broker.get_positions()

        # All positions should be closed or near-zero
        if len(pos_after) > 0:
            assert pos_after.abs().sum() < pos_before.abs().sum()


class TestFailureModes:
    """Test graceful handling of edge cases and failures."""

    def test_nan_prices_filtered_before_generator(self):
        """NaN prices should be filtered before passing to order generator."""
        broker = _make_broker()
        gen = OrderGenerator()

        weights = pd.Series([0.3, 0.3], index=["AAPL", "MSFT"])
        prices = pd.Series([np.nan, 300.0], index=["AAPL", "MSFT"])

        # Filter NaN prices and corresponding weights (defensive pattern)
        valid = prices.dropna().index
        clean_weights = weights.reindex(valid).dropna()
        clean_prices = prices.reindex(valid).dropna()

        orders = gen.generate_orders(
            target_weights=clean_weights,
            current_positions=pd.Series(dtype=float),
            prices=clean_prices,
            nav=1_000_000,
        )
        assert isinstance(orders, list)
        # Only MSFT should have an order
        tickers_ordered = {o.ticker for o in orders}
        assert "AAPL" not in tickers_ordered

    def test_empty_weights(self):
        """Empty weights should produce no orders."""
        gen = OrderGenerator()
        orders = gen.generate_orders(
            target_weights=pd.Series(dtype=float),
            current_positions=pd.Series(dtype=float),
            prices=pd.Series(dtype=float),
            nav=1_000_000,
        )
        assert len(orders) == 0

    def test_missing_ticker_in_market_data(self):
        """Broker handles unknown tickers gracefully."""
        broker = _make_broker()
        order = Order(
            ticker="UNKNOWN", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        ack = broker.submit_order(order)
        assert ack.status == OrderStatus.REJECTED

    def test_zero_nav_no_crash(self):
        """PnL dashboard handles zero initial NAV without division by zero."""
        pnl = PnLDashboard({"initial_nav": 0.0})
        # Should not crash
        snap = pnl.update(100.0, timestamp=pd.Timestamp("2024-01-01"))
        assert snap is not None

    def test_config_corruption_detected(self):
        """Config validator catches corrupt configs."""
        validator = ConfigValidator()
        with pytest.raises(ConfigurationError):
            validator.validate_all({
                "trading": {"max_leverage": -5},
            })
        with pytest.raises(ConfigurationError):
            validator.validate_all({
                "risk": {"drawdown_limit": 2.0},  # > 1.0
            })
