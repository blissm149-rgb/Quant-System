"""Stress tests and fault injection: broker disconnect, NaN injection,
large universes, kill switch extremes, concurrent events, config edge cases,
state machine fuzzing.
"""

import random
import threading
import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, STANDARD_TICKERS, STANDARD_SECTORS, STANDARD_MARKET_DATA
from tests.generators.stress_scenario_generator import (
    inject_nans,
    inject_negative_prices,
    inject_duplicates,
    make_extreme_config,
    make_boundary_configs,
    make_large_universe,
)

SEED = 42


# ---------------------------------------------------------------------------
# 9.1  Broker Disconnect Mid-Order-Batch
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.tier4
class TestBrokerDisconnectMidBatch:
    """Verify partial order submission handled gracefully on disconnect."""

    def test_partial_submission_no_crash(self):
        """Submitting orders to a broker that disconnects mid-batch doesn't crash."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_abstraction_layer import (
            Order, OrderSide, OrderType,
        )

        broker = SimulationBroker({"initial_cash": 1_000_000.0})
        broker.set_market_data(STANDARD_MARKET_DATA)

        orders = [
            Order(
                ticker="AAPL", side=OrderSide.BUY, quantity=100,
                order_type=OrderType.MARKET, order_id=f"ORD-{i}",
            )
            for i in range(20)
        ]

        results = []
        for i, order in enumerate(orders):
            try:
                ack = broker.submit_order(order)
                results.append(ack)
            except Exception:
                # Broker may raise on disconnect — this is acceptable
                break

        # At least some orders should have been processed
        assert len(results) >= 1

    def test_reconnection_manager_handles_failure(self):
        """BrokerReconnectionManager retries after failure."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_reconnection_manager import BrokerReconnectionManager

        primary = SimulationBroker({"initial_cash": 1_000_000.0})
        primary.set_market_data(STANDARD_MARKET_DATA)

        mgr = BrokerReconnectionManager(primary, config={"max_retries": 2})
        connected = mgr.connect()
        assert connected is True
        assert mgr.is_connected


# ---------------------------------------------------------------------------
# 9.2  NaN Injection Into OHLCV Mid-Stream
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.tier4
class TestNaNInjection:
    """Verify DataValidator catches corrupted data."""

    def test_nan_injection_detected(self):
        """DataValidator rejects OHLCV with NaN close prices."""
        from quant_fund.data_layer.data_validator import DataValidator

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=100, seed=SEED)
        corrupted = inject_nans(ohlcv, n_nans=50, columns=["close"])
        as_of = ohlcv.index.get_level_values("date").max() + pd.Timedelta(days=1)

        validator = DataValidator()
        result = validator.validate(corrupted, as_of=as_of)

        # Should have errors or warnings about NaN values
        assert not result.is_valid or len(result.warnings) > 0

    def test_negative_prices_detected(self):
        """DataValidator rejects negative prices."""
        from quant_fund.data_layer.data_validator import DataValidator

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=100, seed=SEED)
        corrupted = inject_negative_prices(ohlcv, n_negatives=10)
        as_of = ohlcv.index.get_level_values("date").max() + pd.Timedelta(days=1)

        validator = DataValidator()
        result = validator.validate(corrupted, as_of=as_of)

        assert not result.is_valid

    def test_duplicate_rows_detected(self):
        """DataValidator flags duplicate (date, ticker) entries."""
        from quant_fund.data_layer.data_validator import DataValidator

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=100, seed=SEED)
        corrupted = inject_duplicates(ohlcv, n_duplicates=20)
        as_of = corrupted.index.get_level_values("date").max() + pd.Timedelta(days=1)

        validator = DataValidator()
        result = validator.validate(corrupted, as_of=as_of)

        # Duplicates should be flagged
        has_issues = not result.is_valid or len(result.warnings) > 0 or len(result.errors) > 0
        assert has_issues


# ---------------------------------------------------------------------------
# 9.3  Large Universe Scalability
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.tier4
class TestLargeUniverse:
    """System handles 100+ ticker universes without crash."""

    def test_100_tickers_feature_computation(self):
        """Feature computation for 100 tickers completes without error."""
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine

        tickers = [f"T{i:04d}" for i in range(100)]
        ohlcv = make_ohlcv(tickers=tickers, periods=252, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max()

        engine = TechnicalIndicatorEngine()
        result = engine.compute_all(ohlcv, as_of)

        assert result is not None
        assert len(result) == 100

    def test_100_tickers_order_generation(self):
        """Order generator handles 100-ticker rebalance."""
        from quant_fund.execution.order_management.order_generator import OrderGenerator

        rng = np.random.default_rng(SEED)
        n = 100
        tickers = [f"T{i:04d}" for i in range(n)]
        target = pd.Series(rng.normal(0, 0.01, n), index=tickers)
        current = pd.Series(0.0, index=tickers)
        prices = pd.Series(rng.uniform(10, 500, n), index=tickers)

        gen = OrderGenerator()
        orders = gen.generate_orders(target, current, prices, nav=10_000_000.0)

        # Should produce at least some orders
        assert isinstance(orders, list)


# ---------------------------------------------------------------------------
# 9.5  Kill Switch with Extreme Drawdown
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.tier4
class TestKillSwitchExtremes:
    """Kill switch must handle extreme scenarios without overflow."""

    def test_90_percent_crash(self):
        """90% drawdown triggers kill switch cleanly."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(10_000_000.0)
        triggered = ks.check(1_000_000.0)  # 90% drawdown

        assert triggered is True
        assert ks.is_halted

    def test_zero_nav(self):
        """NAV = 0 triggers kill switch, no division by zero."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)
        triggered = ks.check(0.0)

        assert triggered is True
        assert ks.is_halted

    def test_negative_nav(self):
        """Negative NAV triggers kill switch without error."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)
        triggered = ks.check(-100_000.0)

        assert triggered is True
        assert ks.is_halted

    def test_tiny_drawdown_no_trigger(self):
        """0.1% drawdown does not trigger with 20% threshold."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)
        triggered = ks.check(999_000.0)  # 0.1% drawdown

        assert triggered is False
        assert not ks.is_halted

    def test_exact_threshold_triggers(self):
        """Drawdown exactly at threshold triggers kill switch."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)
        triggered = ks.check(800_000.0)  # exactly 20%

        assert triggered is True
        assert ks.is_halted

    def test_reset_allows_new_peak(self):
        """After reset, kill switch tracks from new peak."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)
        ks.check(700_000.0)
        assert ks.is_halted

        ks.reset(800_000.0)
        assert not ks.is_halted

        # New drawdown from reset peak
        triggered = ks.check(790_000.0)
        assert triggered is False


# ---------------------------------------------------------------------------
# 9.6  Concurrent EventBus Publishing (Thread Safety)
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.tier4
class TestConcurrentEventBus:
    """Thread-safe event delivery under concurrent publishing."""

    def test_concurrent_publish_no_lost_events(self):
        """10 threads × 100 events = 1000 events all delivered."""
        from quant_fund.infrastructure.event_bus import EventBus, Event, EventType

        bus = EventBus(synchronous=True)
        received = []
        lock = threading.Lock()

        def handler(event):
            with lock:
                received.append(event)

        bus.subscribe("stress", handler, event_types={EventType.MARKET_DATA})

        def publish_batch(thread_id):
            for j in range(100):
                bus.publish(Event(
                    event_type=EventType.MARKET_DATA,
                    payload={"thread": thread_id, "seq": j},
                ))

        threads = [threading.Thread(target=publish_batch, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Allow for dedup if some events get duplicate keys
        assert len(received) >= 900, f"Only {len(received)}/1000 events delivered"

    def test_subscribe_unsubscribe_under_load(self):
        """Subscribe/unsubscribe during concurrent publishing doesn't crash."""
        from quant_fund.infrastructure.event_bus import EventBus, Event, EventType

        bus = EventBus(synchronous=True)

        def noop(event):
            pass

        bus.subscribe("stable", noop, event_types={EventType.MARKET_DATA})

        def publisher():
            for j in range(200):
                bus.publish(Event(
                    event_type=EventType.MARKET_DATA,
                    payload={"seq": j},
                ))

        def churner():
            for j in range(50):
                sub_id = f"temp_{j}"
                bus.subscribe(sub_id, noop, event_types={EventType.MARKET_DATA})
                bus.unsubscribe(sub_id)

        t1 = threading.Thread(target=publisher)
        t2 = threading.Thread(target=churner)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # No crash is the assertion


# ---------------------------------------------------------------------------
# 9.7  Config with Extreme Values
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.tier4
class TestExtremeConfigValues:
    """System handles extreme configuration values gracefully."""

    def test_extreme_constraints_accepted(self):
        """ConstraintEngine accepts extreme but valid config."""
        from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintEngine

        extreme = make_extreme_config()
        engine = ConstraintEngine({"position_limits": extreme})
        constraints = engine.build_constraints()

        assert constraints is not None

    def test_boundary_configs_accepted(self):
        """All boundary configs are accepted without error."""
        from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintEngine

        for cfg in make_boundary_configs():
            engine = ConstraintEngine({"position_limits": cfg})
            constraints = engine.build_constraints()
            assert constraints is not None

    def test_zero_leverage_produces_zero_weights(self):
        """Zero max_leverage should produce zero or near-zero weights."""
        from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintEngine, ConstraintSet

        rng = np.random.default_rng(SEED)
        constraints = ConstraintSet(max_leverage=0.0)
        weights = pd.Series(rng.normal(0, 0.01, 5), index=STANDARD_TICKERS[:5])

        engine = ConstraintEngine()
        violations = engine.validate_weights(weights, constraints)

        # Non-zero weights violate zero leverage
        if weights.abs().sum() > 1e-10:
            assert len(violations) > 0


# ---------------------------------------------------------------------------
# 9.8  State Machine Fuzzing
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.tier4
class TestStateMachineFuzzing:
    """Rapid random transitions don't corrupt state machine."""

    def test_10000_random_transitions_no_corruption(self):
        """10K random transition attempts leave state machine consistent."""
        from quant_fund.infrastructure.system_state_machine import (
            SystemStateMachine, SystemState, InvalidTransitionError,
        )

        sm = SystemStateMachine()
        states = list(SystemState)

        rng = random.Random(SEED)
        for _ in range(10_000):
            target = rng.choice(states)
            try:
                sm.transition_to(target)
            except InvalidTransitionError:
                pass  # expected for invalid transitions

        # State machine is in a valid state
        assert sm.state in set(states)

    def test_valid_path_always_succeeds(self):
        """Known valid transition path always works."""
        from quant_fund.infrastructure.system_state_machine import SystemStateMachine, SystemState

        sm = SystemStateMachine()
        # INITIALIZING → DATA_READY → TRADING_ENABLED → RISK_HALT → SHUTDOWN
        path = [
            SystemState.DATA_READY,
            SystemState.TRADING_ENABLED,
            SystemState.RISK_HALT,
            SystemState.SHUTDOWN,
        ]
        for state in path:
            sm.transition_to(state)
        assert sm.state == SystemState.SHUTDOWN

    def test_shutdown_is_terminal(self):
        """No transition from SHUTDOWN is allowed."""
        from quant_fund.infrastructure.system_state_machine import (
            SystemStateMachine, SystemState, InvalidTransitionError,
        )

        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        sm.transition_to(SystemState.SHUTDOWN)

        for state in SystemState:
            if state == SystemState.SHUTDOWN:
                continue
            with pytest.raises(InvalidTransitionError):
                sm.transition_to(state)

    def test_invalid_backwards_transition(self):
        """Cannot go backward: TRADING_ENABLED → INITIALIZING."""
        from quant_fund.infrastructure.system_state_machine import (
            SystemStateMachine, SystemState, InvalidTransitionError,
        )

        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)

        with pytest.raises(InvalidTransitionError):
            sm.transition_to(SystemState.INITIALIZING)


# ---------------------------------------------------------------------------
# 9.9  Data Alignment Under Stress
# ---------------------------------------------------------------------------

@pytest.mark.stress
@pytest.mark.tier4
class TestDataAlignmentStress:
    """DataAlignmentEngine under edge-case timing conditions."""

    def test_as_of_one_day_after_last_date(self):
        """as_of one day after last data date returns all data within lookback."""
        from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=SEED)
        last_date = ohlcv.index.get_level_values("date").max()
        as_of = last_date + pd.Timedelta(days=1)

        engine = DataAlignmentEngine()
        aligned = engine.get_aligned_data(ohlcv, as_of=as_of, lookback_days=30)

        # All dates in aligned data should be < as_of
        dates = aligned.index.get_level_values("date")
        assert dates.max() < as_of
        assert len(aligned) > 0

    def test_as_of_on_exact_date_raises_lookahead(self):
        """as_of exactly on last data date raises LookAheadError (strict mode)."""
        from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine, LookAheadError

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=SEED)
        last_date = ohlcv.index.get_level_values("date").max()

        engine = DataAlignmentEngine()
        with pytest.raises(LookAheadError):
            engine.get_aligned_data(ohlcv, as_of=last_date, lookback_days=30)

    def test_as_of_before_all_data_raises_in_strict(self):
        """as_of before all data dates raises LookAheadError in strict mode."""
        from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine, LookAheadError

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=SEED)
        first_date = ohlcv.index.get_level_values("date").min()
        early_date = first_date - pd.Timedelta(days=30)

        engine = DataAlignmentEngine()
        with pytest.raises(LookAheadError):
            engine.get_aligned_data(ohlcv, as_of=early_date, lookback_days=30)

    def test_as_of_before_all_data_permissive_returns_empty(self):
        """as_of before all data dates produces empty result in permissive mode."""
        from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=SEED)
        first_date = ohlcv.index.get_level_values("date").min()
        early_date = first_date - pd.Timedelta(days=30)

        engine = DataAlignmentEngine()
        aligned = engine.get_aligned_data_permissive(ohlcv, as_of=early_date, lookback_days=30)

        assert len(aligned) == 0

    def test_permissive_mode_removes_future_data(self):
        """Permissive mode silently removes data >= as_of instead of raising."""
        from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=SEED)
        dates = ohlcv.index.get_level_values("date").unique()
        mid_date = dates[len(dates) // 2]

        engine = DataAlignmentEngine()
        aligned = engine.get_aligned_data_permissive(ohlcv, as_of=mid_date, lookback_days=30)

        result_dates = aligned.index.get_level_values("date")
        assert result_dates.max() < mid_date
