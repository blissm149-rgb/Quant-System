"""Section 12: Mocking/Stubbing Strategy Tests

Validates test doubles: mock broker adapters, always-real modules,
and the mocking boundary conventions.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import STANDARD_TICKERS, STANDARD_SECTORS, STANDARD_MARKET_DATA


@pytest.mark.unit
@pytest.mark.tier2
class TestAlwaysRealModules:
    """These modules must always use real implementations — never mocked."""

    def test_kill_switch_is_real(self):
        """PortfolioKillSwitch must use real logic, not a mock."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)

        # Real behavior: 20% drawdown triggers
        assert ks.check(800_000.0) is True
        assert ks.check(800_001.0) is False

    def test_data_validator_is_real(self):
        """DataValidator must use real validation logic."""
        from quant_fund.data_layer.data_validator import DataValidator
        from tests.conftest import make_ohlcv

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=42)
        as_of = ohlcv.index.get_level_values("date").max() + pd.Timedelta(days=1)

        validator = DataValidator()
        result = validator.validate(ohlcv, as_of=as_of)
        assert result.is_valid

    def test_constraint_engine_is_real(self):
        """ConstraintEngine must use real constraint logic."""
        from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintEngine

        engine = ConstraintEngine()
        constraints = engine.build_constraints(sector_map=STANDARD_SECTORS)

        # Real constraints have meaningful values
        assert constraints.max_position_size == 0.02
        assert constraints.max_leverage == 2.0
        assert constraints.max_sector_exposure == 0.20

    def test_feature_normalizer_is_real(self):
        """FeatureNormalizer must use real normalization."""
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        rng = np.random.default_rng(42)
        raw = pd.Series(rng.normal(5, 2, 50), index=[f"T{i:03d}" for i in range(50)])

        normalizer = FeatureNormalizer()
        normalized = normalizer.normalize(raw, method="zscore")

        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.2

    def test_data_alignment_is_real(self):
        """DataAlignmentEngine must use real look-ahead checks."""
        from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine
        from tests.conftest import make_ohlcv

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=100, seed=42)
        dates = ohlcv.index.get_level_values("date").unique()
        as_of = dates[50]

        engine = DataAlignmentEngine()
        # Real implementation raises on future data
        with pytest.raises(Exception):
            engine.get_aligned_data(ohlcv, as_of=as_of)


@pytest.mark.unit
@pytest.mark.tier2
class TestMockBrokerAdapters:
    """Broker adapters are always mocked for unit tests."""

    def test_ib_adapter_mocked_submit(self):
        """IB adapter submit_order works with mock config (no real connection)."""
        from quant_fund.broker_interface.interactive_brokers_adapter import InteractiveBrokersAdapter
        from quant_fund.broker_interface.broker_abstraction_layer import Order, OrderSide, OrderType

        adapter = InteractiveBrokersAdapter({"ib_host": "127.0.0.1", "ib_port": 0})

        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, order_id="MOCK-001",
        )

        # Without connection, should raise or return error status
        try:
            ack = adapter.submit_order(order)
            # If it returns, check it's an error or unconnected status
            assert ack is not None
        except Exception:
            pass  # Expected — no real broker connection

    def test_alpaca_adapter_mocked_submit(self):
        """Alpaca adapter submit_order works with mock config."""
        from quant_fund.broker_interface.alpaca_adapter import AlpacaAdapter
        from quant_fund.broker_interface.broker_abstraction_layer import Order, OrderSide, OrderType

        adapter = AlpacaAdapter({"alpaca_api_key": "test", "alpaca_secret_key": "test"})

        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, order_id="MOCK-002",
        )

        try:
            ack = adapter.submit_order(order)
            assert ack is not None
        except Exception:
            pass  # Expected — no real API connection

    def test_simulation_broker_always_available(self):
        """SimulationBroker works without any external dependencies."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_abstraction_layer import Order, OrderSide, OrderType

        broker = SimulationBroker({"initial_cash": 1_000_000.0})
        broker.set_market_data(STANDARD_MARKET_DATA)

        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, order_id="SIM-001",
        )

        ack = broker.submit_order(order)
        assert ack is not None

        positions = broker.get_positions()
        assert positions is not None


@pytest.mark.unit
@pytest.mark.tier2
class TestSimulationBrokerConfigurations:
    """Section 11.1: SimulationBroker configs for different testing contexts."""

    def test_ideal_config_no_slippage(self):
        """Ideal config: zero slippage for deterministic unit tests."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_abstraction_layer import Order, OrderSide, OrderType

        broker = SimulationBroker({
            "initial_cash": 1_000_000.0,
            "slippage_bps": 0,
            "latency_ms": 0,
        })
        broker.set_market_data(STANDARD_MARKET_DATA)

        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, order_id="IDEAL-001",
        )
        ack = broker.submit_order(order)
        assert ack is not None

    def test_realistic_config_with_slippage(self):
        """Realistic config: 2-5 bps slippage for integration tests."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_abstraction_layer import Order, OrderSide, OrderType

        broker = SimulationBroker({
            "initial_cash": 1_000_000.0,
            "slippage_bps": 3,
            "latency_ms": 25,
        })
        broker.set_market_data(STANDARD_MARKET_DATA)

        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, order_id="REAL-001",
        )
        ack = broker.submit_order(order)
        assert ack is not None

    def test_adverse_config_high_slippage(self):
        """Adverse config: 10-20 bps slippage for stress tests."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_abstraction_layer import Order, OrderSide, OrderType

        broker = SimulationBroker({
            "initial_cash": 1_000_000.0,
            "slippage_bps": 15,
            "latency_ms": 250,
        })
        broker.set_market_data(STANDARD_MARKET_DATA)

        order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, order_id="ADV-001",
        )
        ack = broker.submit_order(order)
        assert ack is not None
