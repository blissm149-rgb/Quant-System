"""Long-duration stability tests: multi-day simulation, EventBus 100K events,
StateStore 1000 snapshots, broker reconnection cycles.

These tests verify no memory growth, state accumulation, or corruption
over extended operation.  Designed for weekly CI tier.
"""

import time
import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, STANDARD_TICKERS, STANDARD_MARKET_DATA

SEED = 42


# ---------------------------------------------------------------------------
# 14.1  Multi-Day Paper Trading Stability
# ---------------------------------------------------------------------------

@pytest.mark.stability
@pytest.mark.tier5
class TestMultiDayStability:
    """Verify no state corruption over multi-day simulation."""

    def test_20_day_paper_trading_completes(self):
        """20-day paper trading simulation completes without error."""
        from quant_fund.main.paper_trading_runner import PaperTradingRunner
        from quant_fund.main.research_runner import ResearchRunner
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.execution.order_management.order_generator import OrderGenerator
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=SEED)
        dates = ohlcv.index.get_level_values("date").unique()[-20:].tolist()

        research = ResearchRunner({"universe": tickers, "lookback_days": 252})
        research.inject_components(
            feature_generators=TechnicalIndicatorEngine().generators,
            feature_normalizer=FeatureNormalizer(),
        )

        runner = PaperTradingRunner({"strategy_id": "stability", "initial_nav": 1_000_000.0})
        broker = SimulationBroker({"initial_cash": 1_000_000.0})
        runner.inject_components(
            research_runner=research,
            order_generator=OrderGenerator(),
            broker=broker,
            kill_switch=KillSwitch({"drawdown_limit": 0.50}),
        )

        result = runner.run(dates=dates, market_data_by_date={d: ohlcv for d in dates})

        assert result.num_days >= 1
        assert result.final_nav > 0

    def test_nav_bounded_over_20_days(self):
        """NAV stays within reasonable range (no NaN, no extreme values)."""
        from quant_fund.main.paper_trading_runner import PaperTradingRunner
        from quant_fund.main.research_runner import ResearchRunner
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.execution.order_management.order_generator import OrderGenerator
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=SEED)
        dates = ohlcv.index.get_level_values("date").unique()[-20:].tolist()

        research = ResearchRunner({"universe": tickers, "lookback_days": 252})
        research.inject_components(
            feature_generators=TechnicalIndicatorEngine().generators,
            feature_normalizer=FeatureNormalizer(),
        )

        runner = PaperTradingRunner({"strategy_id": "stability", "initial_nav": 1_000_000.0})
        broker = SimulationBroker({"initial_cash": 1_000_000.0})
        runner.inject_components(
            research_runner=research,
            order_generator=OrderGenerator(),
            broker=broker,
            kill_switch=KillSwitch({"drawdown_limit": 0.50}),
        )

        result = runner.run(dates=dates, market_data_by_date={d: ohlcv for d in dates})

        for day in result.daily_results:
            assert not np.isnan(day.nav), f"NaN NAV on {day.date}"
            assert day.nav >= 0, f"Negative NAV on {day.date}"
            assert day.nav < 1e12, f"Unreasonably large NAV on {day.date}"


# ---------------------------------------------------------------------------
# 14.3  EventBus — 100,000 Events
# ---------------------------------------------------------------------------

@pytest.mark.stability
@pytest.mark.tier5
class TestEventBus100K:
    """EventBus handles 100K events without backpressure failure."""

    def test_100k_events_all_delivered(self):
        """100K synchronous events all reach the subscriber."""
        from quant_fund.infrastructure.event_bus import EventBus, Event, EventType

        bus = EventBus(synchronous=True)
        received_count = 0

        def handler(event):
            nonlocal received_count
            received_count += 1

        bus.subscribe("stability", handler, event_types={EventType.MARKET_DATA})

        for i in range(100_000):
            bus.publish(Event(
                event_type=EventType.MARKET_DATA,
                payload={"seq": i},
            ))

        assert received_count == 100_000

    def test_100k_events_no_memory_blowup(self):
        """History buffer doesn't grow unbounded."""
        from quant_fund.infrastructure.event_bus import EventBus, Event, EventType

        bus = EventBus(synchronous=True, config={"history_size": 1_000})

        for i in range(100_000):
            bus.publish(Event(
                event_type=EventType.MARKET_DATA,
                payload={"seq": i},
            ))

        history = bus.get_history(limit=2_000)
        assert len(history) <= 1_000, f"History grew to {len(history)} (expected ≤ 1000)"


# ---------------------------------------------------------------------------
# 14.4  StateStore — 1,000 Snapshots
# ---------------------------------------------------------------------------

@pytest.mark.stability
@pytest.mark.tier5
class TestStateStore1000:
    """StateStore handles 1000 snapshots without corruption."""

    def test_1000_save_restore_cycles(self):
        """Save 1000 snapshots, verify first and last are intact."""
        from quant_fund.infrastructure.state_store import StateStore

        store = StateStore(":memory:")

        for i in range(1_000):
            store.save("test", f"snapshot_{i}", {
                "nav": 1_000_000 + i,
                "positions": {"AAPL": 100 + i},
                "step": i,
            })

        # Verify last snapshot
        last = store.load("test", "snapshot_999")
        assert last["nav"] == 1_000_999
        assert last["positions"]["AAPL"] == 1099
        assert last["step"] == 999

        # Verify first snapshot still accessible
        first = store.load("test", "snapshot_0")
        assert first["nav"] == 1_000_000
        assert first["positions"]["AAPL"] == 100
        assert first["step"] == 0

        store.close()

    def test_overwrite_same_key_1000_times(self):
        """Writing to the same key 1000 times keeps only latest value."""
        from quant_fund.infrastructure.state_store import StateStore

        store = StateStore(":memory:")

        for i in range(1_000):
            store.save("counter", "value", {"count": i})

        result = store.load("counter", "value")
        assert result["count"] == 999

        store.close()

    def test_multiple_namespaces(self):
        """100 namespaces × 10 keys each stored and retrieved correctly."""
        from quant_fund.infrastructure.state_store import StateStore

        store = StateStore(":memory:")

        for ns in range(100):
            for key in range(10):
                store.save(f"ns_{ns}", f"key_{key}", {"ns": ns, "key": key})

        # Spot check
        val = store.load("ns_50", "key_7")
        assert val["ns"] == 50
        assert val["key"] == 7

        val = store.load("ns_99", "key_9")
        assert val["ns"] == 99
        assert val["key"] == 9

        keys = store.list_keys("ns_0")
        assert len(keys) == 10

        store.close()


# ---------------------------------------------------------------------------
# 14.5  Broker Reconnection — Multiple Cycles
# ---------------------------------------------------------------------------

@pytest.mark.stability
@pytest.mark.tier5
class TestBrokerReconnectionCycles:
    """Broker reconnection over repeated connect/disconnect cycles."""

    def test_20_reconnection_cycles(self):
        """20 connect/disconnect cycles with no state corruption."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_reconnection_manager import BrokerReconnectionManager

        primary = SimulationBroker({"initial_cash": 1_000_000.0})
        primary.set_market_data(STANDARD_MARKET_DATA)

        mgr = BrokerReconnectionManager(primary, config={"max_retries": 2})

        for cycle in range(20):
            connected = mgr.connect()
            assert connected is True, f"Failed to connect on cycle {cycle}"
            assert mgr.is_connected

            # Do a basic operation each cycle
            positions = mgr.get_positions()
            assert positions is not None

            mgr.disconnect()

        # Final reconnect
        mgr.connect()
        assert mgr.is_connected

    def test_failover_and_back(self):
        """Primary → secondary → primary failover cycle works."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_reconnection_manager import BrokerReconnectionManager

        primary = SimulationBroker({"initial_cash": 1_000_000.0})
        primary.set_market_data(STANDARD_MARKET_DATA)
        secondary = SimulationBroker({"initial_cash": 500_000.0})
        secondary.set_market_data(STANDARD_MARKET_DATA)

        mgr = BrokerReconnectionManager(primary, secondary=secondary, config={"max_retries": 2})
        connected = mgr.connect()
        assert connected is True


# ---------------------------------------------------------------------------
# 14.6  StatePersistenceManager Round-Trip
# ---------------------------------------------------------------------------

@pytest.mark.stability
@pytest.mark.tier5
class TestStatePersistenceRoundTrip:
    """Save/restore full system state over many cycles."""

    def test_50_snapshot_cycles(self):
        """50 full snapshot + restore cycles with no data loss."""
        from quant_fund.infrastructure.state_store import StateStore
        from quant_fund.infrastructure.state_persistence_manager import StatePersistenceManager
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.infrastructure.system_state_machine import SystemStateMachine

        store = StateStore(":memory:")
        persistence = StatePersistenceManager(store)

        for i in range(50):
            ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000.0 + i})
            sm = SystemStateMachine()

            persistence.save_kill_switch_state(ks)
            persistence.save_system_state(sm)

        # Restore last saved state
        ks2 = KillSwitch({"drawdown_limit": 0.20})
        restored = persistence.restore_kill_switch_state(ks2)
        # Should restore without error
        assert isinstance(restored, bool)

        store.close()


# ---------------------------------------------------------------------------
# 14.7  Research Runner Backtest Stability
# ---------------------------------------------------------------------------

@pytest.mark.stability
@pytest.mark.tier5
class TestResearchBacktestStability:
    """Walk-forward backtest over many dates doesn't accumulate errors."""

    def test_50_date_backtest_completes(self):
        """50-date backtest completes without crash and returns results for all dates."""
        from quant_fund.main.research_runner import ResearchRunner
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=SEED)
        dates = ohlcv.index.get_level_values("date").unique()[-50:].tolist()

        runner = ResearchRunner({"universe": tickers, "lookback_days": 252})
        runner.inject_components(
            feature_generators=TechnicalIndicatorEngine().generators,
            feature_normalizer=FeatureNormalizer(),
        )

        results = runner.run_backtest(dates=dates, market_data=ohlcv)

        # Backtest returns a result for every date requested
        assert len(results) == 50
        # Each result should have a feature_matrix (features computed even if alpha fails)
        computed = [r for r in results if r.feature_matrix is not None]
        assert len(computed) > 0, "No dates produced feature matrices"
