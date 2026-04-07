"""Phase-1 safety tests — OrderSafetyValidator, ConfigValidator,
SimulationBroker cash floor, and PaperTradingRunner exposure blocking.

Target: 40+ tests covering all safety checks, config bounds,
broker cash enforcement, and exposure-breach order blocking.
"""

import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quant_fund.broker_interface.broker_abstraction_layer import (

    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.config.config_validator import (
    EXECUTION_BOUNDS,
    RISK_BOUNDS,
    SYSTEM_BOUNDS,
    TRADING_BOUNDS,
    BoundSpec,
    ConfigurationError,
    ConfigValidator,
    validate_config,
)

pytestmark = [pytest.mark.tier1]
from quant_fund.execution.order_management.order_safety_validator import (
    OrderSafetyValidator,
    SafetyRejection,
    SafetyResult,
)
from quant_fund.main.paper_trading_runner import PaperTradingRunner

TS = pd.Timestamp("2024-01-01")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buy_order(ticker="AAPL", qty=100, order_type=OrderType.MARKET,
               limit_price=None):
    return Order(
        ticker=ticker, side=OrderSide.BUY, quantity=qty,
        order_type=order_type, limit_price=limit_price, timestamp=TS,
    )


def _sell_order(ticker="AAPL", qty=100, order_type=OrderType.MARKET,
                limit_price=None):
    return Order(
        ticker=ticker, side=OrderSide.SELL, quantity=qty,
        order_type=order_type, limit_price=limit_price, timestamp=TS,
    )


# ===================================================================
# OrderSafetyValidator tests
# ===================================================================

class TestOrderSafetyValidatorPasses:
    """Orders that should pass all checks."""

    def test_valid_market_buy_passes(self):
        v = OrderSafetyValidator()
        result = v.validate(
            _buy_order(qty=100), nav=1_000_000, mid_price=150.0,
            adv=1_000_000, cash=100_000,
        )
        assert result.passed
        assert result.rejections == []

    def test_valid_market_sell_passes(self):
        v = OrderSafetyValidator()
        result = v.validate(
            _sell_order(qty=50), nav=1_000_000, mid_price=150.0,
            adv=1_000_000, current_position=500,
        )
        assert result.passed

    def test_valid_limit_order_passes(self):
        v = OrderSafetyValidator()
        order = _buy_order(qty=100, order_type=OrderType.LIMIT, limit_price=151.0)
        result = v.validate(order, nav=1_000_000, mid_price=150.0,
                            adv=1_000_000, cash=100_000)
        assert result.passed


class TestQuantityPositive:
    def test_zero_quantity_rejected(self):
        v = OrderSafetyValidator()
        result = v.validate(_buy_order(qty=0))
        assert not result.passed
        names = [r.check_name for r in result.rejections]
        assert "quantity_positive" in names

    def test_negative_quantity_rejected(self):
        v = OrderSafetyValidator()
        result = v.validate(_buy_order(qty=-10))
        assert not result.passed
        names = [r.check_name for r in result.rejections]
        assert "quantity_positive" in names


class TestTickerValid:
    def test_empty_ticker_rejected(self):
        v = OrderSafetyValidator()
        order = _buy_order(ticker="", qty=100)
        result = v.validate(order)
        assert not result.passed
        names = [r.check_name for r in result.rejections]
        assert "ticker_valid" in names

    def test_whitespace_ticker_rejected(self):
        v = OrderSafetyValidator()
        order = _buy_order(ticker="   ", qty=100)
        result = v.validate(order)
        names = [r.check_name for r in result.rejections]
        assert "ticker_valid" in names


class TestMaxShares:
    def test_exceeds_max_shares(self):
        v = OrderSafetyValidator({"max_shares_per_order": 500})
        result = v.validate(_buy_order(qty=501))
        names = [r.check_name for r in result.rejections]
        assert "max_shares" in names

    def test_at_max_shares_passes(self):
        v = OrderSafetyValidator({"max_shares_per_order": 500})
        result = v.validate(_buy_order(qty=500), cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "max_shares" not in names


class TestMaxPctNav:
    def test_exceeds_nav_limit(self):
        v = OrderSafetyValidator({"max_order_pct_nav": 0.05})
        # 1000 shares * $100 = $100k, NAV = $1M => 10% > 5%
        result = v.validate(_buy_order(qty=1000), nav=1_000_000,
                            mid_price=100.0, cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "max_pct_nav" in names

    def test_within_nav_limit(self):
        v = OrderSafetyValidator({"max_order_pct_nav": 0.05})
        # 10 shares * $100 = $1k, NAV = $1M => 0.1% < 5%
        result = v.validate(_buy_order(qty=10), nav=1_000_000,
                            mid_price=100.0, cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "max_pct_nav" not in names

    def test_nav_zero_skips_check(self):
        v = OrderSafetyValidator()
        result = v.validate(_buy_order(qty=999_999), nav=0, mid_price=100.0)
        names = [r.check_name for r in result.rejections]
        assert "max_pct_nav" not in names


class TestMaxPctAdv:
    def test_exceeds_adv_limit(self):
        v = OrderSafetyValidator({"max_order_pct_adv": 0.10})
        # qty 200 > adv 1000 * 0.10 = 100
        result = v.validate(_buy_order(qty=200), adv=1000, cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "max_pct_adv" in names

    def test_within_adv_limit(self):
        v = OrderSafetyValidator({"max_order_pct_adv": 0.10})
        result = v.validate(_buy_order(qty=50), adv=1000, cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "max_pct_adv" not in names


class TestFatFinger:
    def test_fat_finger_triggered(self):
        v = OrderSafetyValidator({"fat_finger_multiplier": 10.0})
        # qty 1100 > position 100 * 10 = 1000
        result = v.validate(_buy_order(qty=1100), current_position=100,
                            cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "fat_finger" in names

    def test_fat_finger_not_triggered(self):
        v = OrderSafetyValidator({"fat_finger_multiplier": 10.0})
        result = v.validate(_buy_order(qty=500), current_position=100,
                            cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "fat_finger" not in names

    def test_fat_finger_skipped_when_no_position(self):
        v = OrderSafetyValidator({"fat_finger_multiplier": 10.0})
        result = v.validate(_buy_order(qty=999_999), current_position=0,
                            cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "fat_finger" not in names


class TestLimitPriceSanity:
    def test_limit_too_far_from_mid(self):
        v = OrderSafetyValidator({"max_limit_deviation_bps": 500})
        # 6% deviation = 600 bps > 500
        order = _buy_order(qty=10, order_type=OrderType.LIMIT, limit_price=106.0)
        result = v.validate(order, mid_price=100.0, cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "limit_price_sanity" in names

    def test_limit_within_tolerance(self):
        v = OrderSafetyValidator({"max_limit_deviation_bps": 500})
        order = _buy_order(qty=10, order_type=OrderType.LIMIT, limit_price=101.0)
        result = v.validate(order, mid_price=100.0, cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "limit_price_sanity" not in names

    def test_market_order_skips_limit_check(self):
        v = OrderSafetyValidator()
        order = _buy_order(qty=10, order_type=OrderType.MARKET)
        result = v.validate(order, mid_price=100.0, cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "limit_price_sanity" not in names


class TestBuyingPower:
    def test_buy_exceeds_cash(self):
        v = OrderSafetyValidator()
        # 100 * 150 = 15000 > 10000
        result = v.validate(_buy_order(qty=100), mid_price=150.0, cash=10_000)
        names = [r.check_name for r in result.rejections]
        assert "buying_power" in names

    def test_buy_within_cash(self):
        v = OrderSafetyValidator()
        result = v.validate(_buy_order(qty=10), mid_price=150.0, cash=100_000)
        names = [r.check_name for r in result.rejections]
        assert "buying_power" not in names

    def test_sell_ignores_buying_power(self):
        v = OrderSafetyValidator()
        result = v.validate(_sell_order(qty=100), mid_price=150.0, cash=0)
        names = [r.check_name for r in result.rejections]
        assert "buying_power" not in names


class TestRateLimit:
    def test_rate_limit_triggered(self):
        v = OrderSafetyValidator({"max_orders_per_second": 2})
        v.validate(_buy_order(qty=1), cash=float("inf"))
        v.validate(_buy_order(qty=2), cash=float("inf"))
        result = v.validate(_buy_order(qty=3), cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "rate_limit" in names

    def test_rate_limit_different_tickers_ok(self):
        v = OrderSafetyValidator({"max_orders_per_second": 2})
        v.validate(_buy_order(ticker="AAPL", qty=1), cash=float("inf"))
        v.validate(_buy_order(ticker="AAPL", qty=2), cash=float("inf"))
        result = v.validate(_buy_order(ticker="GOOG", qty=1), cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "rate_limit" not in names


class TestDuplicate:
    def test_duplicate_detected(self):
        v = OrderSafetyValidator({"dedup_window_s": 60})
        v.validate(_buy_order(qty=100), cash=float("inf"))
        result = v.validate(_buy_order(qty=100), cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "duplicate" in names

    def test_different_qty_not_duplicate(self):
        v = OrderSafetyValidator({"dedup_window_s": 60})
        v.validate(_buy_order(qty=100), cash=float("inf"))
        result = v.validate(_buy_order(qty=200), cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "duplicate" not in names

    def test_different_side_not_duplicate(self):
        v = OrderSafetyValidator({"dedup_window_s": 60})
        v.validate(_buy_order(qty=100), cash=float("inf"))
        result = v.validate(_sell_order(qty=100), cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "duplicate" not in names


class TestClearState:
    def test_clear_resets_rate_limit(self):
        v = OrderSafetyValidator({"max_orders_per_second": 1})
        v.validate(_buy_order(qty=1), cash=float("inf"))
        v.clear_state()
        result = v.validate(_buy_order(qty=2), cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "rate_limit" not in names

    def test_clear_resets_dedup(self):
        v = OrderSafetyValidator({"dedup_window_s": 60})
        v.validate(_buy_order(qty=100), cash=float("inf"))
        v.clear_state()
        result = v.validate(_buy_order(qty=100), cash=float("inf"))
        names = [r.check_name for r in result.rejections]
        assert "duplicate" not in names


class TestSafetyResultDataclass:
    def test_passed_result(self):
        r = SafetyResult(passed=True, rejections=[])
        assert r.passed
        assert r.rejections == []

    def test_rejection_fields(self):
        rej = SafetyRejection(
            check_name="test_check", message="bad order",
            order_ticker="XYZ", order_qty=42,
        )
        assert rej.check_name == "test_check"
        assert rej.order_ticker == "XYZ"
        assert rej.order_qty == 42


# ===================================================================
# ConfigValidator tests
# ===================================================================

class TestConfigValidatorValidConfigs:
    def test_valid_full_config_passes(self):
        cv = ConfigValidator()
        configs = {
            "risk": {"drawdown_limit": 0.2, "max_sector_exposure": 0.3,
                     "max_single_name_exposure": 0.1, "daily_var_limit": 0.05},
            "trading": {"max_leverage": 2.0, "max_position_size": 0.1,
                        "max_sector_exposure": 0.4},
            "execution": {"participation_rate": 0.05, "max_order_size_usd": 1_000_000,
                          "latency_ms": 100},
            "system": {"initial_nav": 1_000_000.0},
        }
        cv.validate_all(configs)  # should not raise

    def test_empty_config_passes(self):
        cv = ConfigValidator()
        cv.validate_all({})  # no keys, no errors


class TestConfigValidatorBoundViolations:
    def test_drawdown_limit_zero_rejected(self):
        errors = validate_config({"drawdown_limit": 0.0}, RISK_BOUNDS, "risk")
        assert any("drawdown_limit" in e for e in errors)

    def test_drawdown_limit_above_max_rejected(self):
        errors = validate_config({"drawdown_limit": 1.5}, RISK_BOUNDS, "risk")
        assert any("drawdown_limit" in e for e in errors)

    def test_max_leverage_zero_rejected(self):
        errors = validate_config({"max_leverage": 0.0}, TRADING_BOUNDS, "trading")
        assert any("max_leverage" in e for e in errors)

    def test_max_leverage_above_10_rejected(self):
        errors = validate_config({"max_leverage": 11.0}, TRADING_BOUNDS, "trading")
        assert any("max_leverage" in e for e in errors)

    def test_max_position_size_above_half_rejected(self):
        errors = validate_config({"max_position_size": 0.6}, TRADING_BOUNDS, "trading")
        assert any("max_position_size" in e for e in errors)

    def test_participation_rate_above_half_rejected(self):
        errors = validate_config({"participation_rate": 0.6}, EXECUTION_BOUNDS, "exec")
        assert any("participation_rate" in e for e in errors)

    def test_initial_nav_zero_rejected(self):
        errors = validate_config({"initial_nav": 0.0}, SYSTEM_BOUNDS, "system")
        assert any("initial_nav" in e for e in errors)

    def test_negative_value_rejected(self):
        errors = validate_config({"max_leverage": -1.0}, TRADING_BOUNDS, "trading")
        assert any("max_leverage" in e for e in errors)


class TestValidateAllRaises:
    def test_raises_configuration_error(self):
        cv = ConfigValidator()
        configs = {
            "risk": {"drawdown_limit": 0.0},  # violates > 0
        }
        with pytest.raises(ConfigurationError):
            cv.validate_all(configs)

    def test_error_message_contains_field_name(self):
        cv = ConfigValidator()
        configs = {"trading": {"max_leverage": 0.0}}
        with pytest.raises(ConfigurationError, match="max_leverage"):
            cv.validate_all(configs)


class TestValidateSingle:
    def test_returns_errors_without_raising(self):
        cv = ConfigValidator()
        errors = cv.validate_single({"drawdown_limit": 0.0}, "risk")
        assert len(errors) > 0
        assert any("drawdown_limit" in e for e in errors)

    def test_valid_returns_empty(self):
        cv = ConfigValidator()
        errors = cv.validate_single({"drawdown_limit": 0.1}, "risk")
        drawdown_errors = [e for e in errors if "drawdown_limit" in e]
        assert drawdown_errors == []

    def test_custom_bounds_via_argument(self):
        cv = ConfigValidator()
        custom = [BoundSpec("my_param", min_val=0, max_val=1, min_exclusive=True)]
        errors = cv.validate_single({"my_param": 0.0}, "custom", bounds=custom)
        assert any("my_param" in e for e in errors)


class TestNestedConfigLookup:
    def test_nested_value_found(self):
        config = {"outer": {"drawdown_limit": 0.5}}
        errors = validate_config(config, RISK_BOUNDS, "risk")
        drawdown_errors = [e for e in errors if "drawdown_limit" in e]
        assert drawdown_errors == []

    def test_deeply_nested_value(self):
        config = {"level1": {"level2": {"participation_rate": 0.05}}}
        errors = validate_config(config, EXECUTION_BOUNDS, "exec")
        pr_errors = [e for e in errors if "participation_rate" in e]
        assert pr_errors == []


class TestAddBounds:
    def test_custom_bounds_validated(self):
        cv = ConfigValidator()
        cv.add_bounds("custom", [
            BoundSpec("threshold", min_val=0, max_val=100, min_exclusive=True),
        ])
        configs = {"custom": {"threshold": 0.0}}
        with pytest.raises(ConfigurationError, match="threshold"):
            cv.validate_all(configs)


# ===================================================================
# SimulationBroker cash-floor tests
# ===================================================================

def _make_broker(initial_cash=100_000, enforce_cash_floor=True):
    broker = SimulationBroker({
        "initial_cash": initial_cash,
        "enforce_cash_floor": enforce_cash_floor,
        "half_spread_bps": 0.0,
        "market_impact_bps": 0.0,
        "commission_per_share": 0.0,
    })
    broker.set_market_data({
        "AAPL": {"mid": 150.0, "last": 150.0, "bid": 149.9,
                 "ask": 150.1, "volume": 1_000_000, "adv": 1_000_000},
    })
    return broker


class TestSimulationBrokerCashFloor:
    def test_buy_rejected_when_cost_exceeds_cash(self):
        broker = _make_broker(initial_cash=1_000)
        # 100 * 150 = 15000 > 1000
        order = _buy_order(qty=100)
        ack = broker.submit_order(order)
        assert ack.status == OrderStatus.REJECTED
        assert "Insufficient cash" in ack.message

    def test_buy_accepted_when_within_cash(self):
        broker = _make_broker(initial_cash=100_000)
        # 10 * 150 = 1500 < 100000
        order = _buy_order(qty=10)
        ack = broker.submit_order(order)
        assert ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL)

    def test_sell_unaffected_by_cash_floor(self):
        broker = _make_broker(initial_cash=0)
        # First seed a position by disabling cash floor temporarily
        broker._positions["AAPL"] = 100
        order = _sell_order(qty=50)
        ack = broker.submit_order(order)
        assert ack.status != OrderStatus.REJECTED

    def test_cash_floor_disabled(self):
        broker = _make_broker(initial_cash=1_000, enforce_cash_floor=False)
        order = _buy_order(qty=100)
        ack = broker.submit_order(order)
        # Should NOT be rejected even though cost > cash
        assert ack.status != OrderStatus.REJECTED

    def test_cash_decreases_after_buy(self):
        broker = _make_broker(initial_cash=100_000)
        order = _buy_order(qty=10)
        broker.submit_order(order)
        assert broker.cash < 100_000


# ===================================================================
# PaperTradingRunner exposure-blocking tests
# ===================================================================

class TestPaperTradingRunnerExposureBlocking:
    def _make_runner_with_exposure_breach(self, breaches):
        """Build a runner where exposure_monitor returns breaches."""
        runner = PaperTradingRunner({"strategy_id": "test", "initial_nav": 1_000_000})

        broker = MagicMock()
        broker.get_account_value.return_value = 1_000_000.0

        exposure_monitor = MagicMock()
        exposure_monitor.check.return_value = breaches

        research_runner = MagicMock()
        research_result = MagicMock()
        research_result.alpha_scores = pd.Series({"AAPL": 0.5})
        research_result.validation_flags = []
        research_runner.run_cycle.return_value = research_result

        optimizer = MagicMock()
        optimizer.optimize.return_value = pd.Series({"AAPL": 0.1})

        order_generator = MagicMock()
        order_generator.generate_orders.return_value = [_buy_order()]

        runner.inject_components(
            broker=broker,
            exposure_monitor=exposure_monitor,
            research_runner=research_runner,
            portfolio_optimizer=optimizer,
            order_generator=order_generator,
        )
        return runner, order_generator

    def test_exposure_breach_sets_status(self):
        runner, _ = self._make_runner_with_exposure_breach(
            ["sector_exposure > 0.4"]
        )
        result = runner.run([TS])
        day = result.daily_results[0]
        assert day.status == "exposure_breach"

    def test_exposure_breach_blocks_order_generation(self):
        runner, order_gen = self._make_runner_with_exposure_breach(
            ["sector_exposure > 0.4"]
        )
        runner.run([TS])
        order_gen.generate_orders.assert_not_called()

    def test_exposure_breach_records_breaches(self):
        runner, _ = self._make_runner_with_exposure_breach(
            ["sector_exposure > 0.4", "single_name > 0.15"]
        )
        result = runner.run([TS])
        day = result.daily_results[0]
        assert len(day.exposure_breaches) == 2

    def test_no_breach_allows_orders(self):
        runner, order_gen = self._make_runner_with_exposure_breach([])
        # No breaches — empty list — orders should proceed
        broker = MagicMock()
        broker.get_account_value.return_value = 1_000_000.0
        broker.get_positions.return_value = pd.Series(dtype=float)
        md_df = pd.DataFrame({"mid": [150.0]}, index=["AAPL"])
        broker.get_market_data.return_value = md_df
        runner.inject_components(broker=broker)

        order_router = MagicMock()
        order_router.route_orders.return_value = []
        runner.inject_components(order_router=order_router)

        runner.run([TS])
        order_gen.generate_orders.assert_called_once()
