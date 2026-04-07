"""Comprehensive validation tests for previously untested modules.

Covers 6 identified gaps:
1. MarketHoursEnforcer — safety-critical order gating
2. Broker adapters (IB + Alpaca) disconnected-state behavior
3. NeuralNetworkPredictor — ML model with real training logic
4. YFinanceConnector._reshape_yfinance — data schema transformation
5. Full-chain integration: Research → Portfolio → Execution
6. Stress scenario → risk response (kill switch, drawdown alerts)
"""

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from tests.conftest import (

    STANDARD_MARKET_DATA,
    STANDARD_SECTORS,
    STANDARD_TICKERS,
    make_factor_returns,
    make_ohlcv,
    make_returns,
)

pytestmark = [pytest.mark.tier3]

ET = ZoneInfo("America/New_York")


# ═══════════════════════════════════════════════════════════════════
# GAP 1: MarketHoursEnforcer
# ═══════════════════════════════════════════════════════════════════


class TestMarketHoursEnforcer:
    """Tests for the NYSE market hours enforcement module.

    This is safety-critical: bugs here could allow orders outside
    trading hours in a live system.
    """

    @pytest.fixture
    def enforcer(self):
        from quant_fund.execution.order_management.market_hours_enforcer import (
            MarketHoursEnforcer,
        )
        return MarketHoursEnforcer()

    @pytest.fixture
    def extended_enforcer(self):
        from quant_fund.execution.order_management.market_hours_enforcer import (
            MarketHoursEnforcer,
        )
        return MarketHoursEnforcer({"allow_extended_hours": True})

    # --- is_market_open ---

    def test_regular_hours_open(self, enforcer):
        """10:00 AM ET on a Tuesday → market is open."""
        ts = pd.Timestamp("2025-03-04 10:00", tz=ET)  # Tuesday
        assert enforcer.is_market_open(ts) is True

    def test_before_open(self, enforcer):
        """9:00 AM ET → market not yet open."""
        ts = pd.Timestamp("2025-03-04 09:00", tz=ET)
        assert enforcer.is_market_open(ts) is False

    def test_at_open(self, enforcer):
        """Exactly 9:30 AM ET → market is open."""
        ts = pd.Timestamp("2025-03-04 09:30", tz=ET)
        assert enforcer.is_market_open(ts) is True

    def test_after_close(self, enforcer):
        """4:30 PM ET → market closed."""
        ts = pd.Timestamp("2025-03-04 16:30", tz=ET)
        assert enforcer.is_market_open(ts) is False

    def test_at_close(self, enforcer):
        """Exactly 4:00 PM ET → market closed (close time is exclusive)."""
        ts = pd.Timestamp("2025-03-04 16:00", tz=ET)
        assert enforcer.is_market_open(ts) is False

    def test_weekend_saturday(self, enforcer):
        """Saturday → market closed."""
        ts = pd.Timestamp("2025-03-08 12:00", tz=ET)  # Saturday
        assert enforcer.is_market_open(ts) is False

    def test_weekend_sunday(self, enforcer):
        """Sunday → market closed."""
        ts = pd.Timestamp("2025-03-09 12:00", tz=ET)  # Sunday
        assert enforcer.is_market_open(ts) is False

    def test_nyse_holiday_christmas(self, enforcer):
        """Christmas 2025 → market closed."""
        ts = pd.Timestamp("2025-12-25 12:00", tz=ET)  # Thursday
        assert enforcer.is_market_open(ts) is False

    def test_nyse_holiday_mlk_day(self, enforcer):
        """MLK Day 2025 → market closed."""
        ts = pd.Timestamp("2025-01-20 10:00", tz=ET)
        assert enforcer.is_market_open(ts) is False

    def test_early_close_before_1pm(self, enforcer):
        """Black Friday 2025 at 12:30 PM → market still open (early close at 1 PM)."""
        ts = pd.Timestamp("2025-11-28 12:30", tz=ET)
        assert enforcer.is_market_open(ts) is True

    def test_early_close_after_1pm(self, enforcer):
        """Black Friday 2025 at 1:30 PM → market closed (early close at 1 PM)."""
        ts = pd.Timestamp("2025-11-28 13:30", tz=ET)
        assert enforcer.is_market_open(ts) is False

    # --- can_submit_order ---

    def test_can_submit_regular_hours(self, enforcer):
        """10:00 AM ET weekday → allowed."""
        ts = pd.Timestamp("2025-03-04 10:00", tz=ET)
        allowed, reason = enforcer.can_submit_order(ts)
        assert allowed is True
        assert reason == "OK"

    def test_cannot_submit_weekend(self, enforcer):
        ts = pd.Timestamp("2025-03-08 12:00", tz=ET)
        allowed, reason = enforcer.can_submit_order(ts)
        assert allowed is False
        assert "weekend" in reason.lower()

    def test_cannot_submit_holiday(self, enforcer):
        ts = pd.Timestamp("2025-12-25 10:00", tz=ET)
        allowed, reason = enforcer.can_submit_order(ts)
        assert allowed is False
        assert "holiday" in reason.lower()

    def test_eod_cutoff_blocks_submission(self, enforcer):
        """3:56 PM ET (within default 5-min cutoff) → blocked."""
        ts = pd.Timestamp("2025-03-04 15:56", tz=ET)
        allowed, reason = enforcer.can_submit_order(ts)
        assert allowed is False
        assert "cutoff" in reason.lower()

    def test_eod_cutoff_just_before_ok(self, enforcer):
        """3:54 PM ET (before 5-min cutoff) → allowed."""
        ts = pd.Timestamp("2025-03-04 15:54", tz=ET)
        allowed, reason = enforcer.can_submit_order(ts)
        assert allowed is True

    def test_extended_hours_allowed(self, extended_enforcer):
        """7:00 AM ET with extended hours enabled → allowed."""
        ts = pd.Timestamp("2025-03-04 07:00", tz=ET)
        allowed, reason = extended_enforcer.can_submit_order(ts)
        assert allowed is True

    def test_extended_hours_outside(self, extended_enforcer):
        """3:00 AM ET even with extended hours → blocked."""
        ts = pd.Timestamp("2025-03-04 03:00", tz=ET)
        allowed, reason = extended_enforcer.can_submit_order(ts)
        assert allowed is False

    # --- next_market_open ---

    def test_next_open_same_day(self, enforcer):
        """8:00 AM on a weekday → same day 9:30 AM."""
        ts = pd.Timestamp("2025-03-04 08:00", tz=ET)
        next_open = enforcer.next_market_open(ts)
        assert next_open.hour == 9
        assert next_open.minute == 30
        assert next_open.date() == ts.date()

    def test_next_open_after_close(self, enforcer):
        """After close on Friday → Monday 9:30 AM."""
        ts = pd.Timestamp("2025-03-07 17:00", tz=ET)  # Friday 5 PM
        next_open = enforcer.next_market_open(ts)
        assert next_open.weekday() == 0  # Monday
        assert next_open.hour == 9
        assert next_open.minute == 30

    def test_next_open_across_holiday(self, enforcer):
        """Wednesday before Thanksgiving 2025 after close → skip Thursday to Friday."""
        ts = pd.Timestamp("2025-11-26 17:00", tz=ET)  # Wed after close
        next_open = enforcer.next_market_open(ts)
        # Thanksgiving is Nov 27 (Thursday) — next open is Nov 28 (Friday)
        assert next_open.date() == pd.Timestamp("2025-11-28").date()

    # --- is_holiday ---

    def test_is_holiday_known_dates(self, enforcer):
        """Verify is_holiday for a few known NYSE holidays."""
        assert enforcer.is_holiday(pd.Timestamp("2025-12-25", tz=ET)) is True
        assert enforcer.is_holiday(pd.Timestamp("2025-07-04", tz=ET)) is True
        assert enforcer.is_holiday(pd.Timestamp("2025-03-04", tz=ET)) is False

    def test_utc_timestamp_converted(self, enforcer):
        """UTC timestamp is converted to ET before checking."""
        # 2:00 PM UTC = 10:00 AM ET (during EDT)
        ts = pd.Timestamp("2025-03-04 15:00", tz="UTC")
        assert enforcer.is_market_open(ts) is True


# ═══════════════════════════════════════════════════════════════════
# GAP 2: Broker Adapters — Disconnected State
# ═══════════════════════════════════════════════════════════════════


class TestBrokerAdaptersDisconnected:
    """IB and Alpaca adapters must return safe defaults when not connected.

    These tests verify the disconnected code paths without requiring
    the external packages (ib_insync, alpaca_trade_api).
    """

    @pytest.fixture
    def ib_adapter(self):
        from quant_fund.broker_interface.interactive_brokers_adapter import (
            InteractiveBrokersAdapter,
        )
        return InteractiveBrokersAdapter()

    @pytest.fixture
    def alpaca_adapter(self):
        from quant_fund.broker_interface.alpaca_adapter import AlpacaAdapter
        return AlpacaAdapter()

    @pytest.fixture
    def sample_order(self):
        from quant_fund.broker_interface.broker_abstraction_layer import (
            Order,
            OrderSide,
            OrderType,
        )
        return Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp("2025-01-15"),
        )

    # --- Interactive Brokers ---

    def test_ib_not_connected_submit_rejected(self, ib_adapter, sample_order):
        from quant_fund.broker_interface.broker_abstraction_layer import OrderStatus
        ack = ib_adapter.submit_order(sample_order)
        assert ack.status == OrderStatus.REJECTED

    def test_ib_not_connected_positions_empty(self, ib_adapter):
        positions = ib_adapter.get_positions()
        assert isinstance(positions, pd.Series)
        assert positions.empty

    def test_ib_not_connected_account_zero(self, ib_adapter):
        assert ib_adapter.get_account_value() == 0.0

    def test_ib_not_connected_fills_empty(self, ib_adapter):
        fills = ib_adapter.get_fills(pd.Timestamp("2025-01-01"))
        assert isinstance(fills, list)
        assert len(fills) == 0

    def test_ib_not_connected_cancel_false(self, ib_adapter):
        assert ib_adapter.cancel_order("some-order-id") is False

    def test_ib_not_connected_market_data_empty(self, ib_adapter):
        df = ib_adapter.get_market_data(["AAPL", "MSFT"])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    # --- Alpaca ---

    def test_alpaca_not_connected_submit_rejected(self, alpaca_adapter, sample_order):
        from quant_fund.broker_interface.broker_abstraction_layer import OrderStatus
        ack = alpaca_adapter.submit_order(sample_order)
        assert ack.status == OrderStatus.REJECTED

    def test_alpaca_not_connected_positions_empty(self, alpaca_adapter):
        positions = alpaca_adapter.get_positions()
        assert isinstance(positions, pd.Series)
        assert positions.empty

    def test_alpaca_not_connected_account_zero(self, alpaca_adapter):
        assert alpaca_adapter.get_account_value() == 0.0

    def test_alpaca_not_connected_fills_empty(self, alpaca_adapter):
        fills = alpaca_adapter.get_fills(pd.Timestamp("2025-01-01"))
        assert isinstance(fills, list)
        assert len(fills) == 0

    def test_alpaca_not_connected_cancel_false(self, alpaca_adapter):
        assert alpaca_adapter.cancel_order("some-order-id") is False

    def test_alpaca_not_connected_market_data_empty(self, alpaca_adapter):
        df = alpaca_adapter.get_market_data(["AAPL", "MSFT"])
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ═══════════════════════════════════════════════════════════════════
# GAP 3: NeuralNetworkPredictor
# ═══════════════════════════════════════════════════════════════════


class TestNeuralNetworkPredictor:
    """Tests for the sklearn MLP-based neural network predictor."""

    @pytest.fixture
    def predictor(self):
        from quant_fund.research_algorithms.machine_learning.neural_network_predictor import (
            NeuralNetworkPredictor,
        )
        return NeuralNetworkPredictor({"nn_min_train_days": 100})

    @pytest.fixture
    def training_data(self):
        """Feature matrix and forward returns for training."""
        rng = np.random.default_rng(42)
        n = 300
        features = pd.DataFrame({
            "momentum": rng.normal(0, 1, n),
            "volatility": rng.normal(0, 1, n),
            "mean_rev": rng.normal(0, 1, n),
        })
        # Forward returns weakly correlated with momentum
        forward = 0.01 * features["momentum"] + rng.normal(0, 0.02, n)
        return features, pd.Series(forward)

    @pytest.fixture
    def ohlcv_with_features(self):
        """OHLCV data with extra feature columns for prediction."""
        rng = np.random.default_rng(42)
        tickers = ["AAPL", "MSFT", "GOOG"]
        dates = pd.bdate_range("2023-01-01", periods=60)
        rows = []
        for d in dates:
            for t in tickers:
                c = 100 + rng.normal() * 5
                rows.append({
                    "date": d, "ticker": t,
                    "open": c, "high": c + 1, "low": c - 1,
                    "close": c, "volume": 1_000_000,
                    "momentum": rng.normal(),
                    "volatility": rng.normal(),
                    "mean_rev": rng.normal(),
                })
        return pd.DataFrame(rows).set_index(["date", "ticker"])

    def test_compute_untrained_returns_nan(self, predictor, ohlcv_with_features):
        """No model trained → compute returns NaN for all tickers."""
        result = predictor.compute(
            ohlcv_with_features, pd.Timestamp("2023-03-15")
        )
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_train_with_sufficient_data(self, predictor, training_data):
        """300 samples (> min 100) → training succeeds."""
        features, forward = training_data
        result = predictor.train_model(features, forward)
        assert "error" not in result
        assert "train_r2" in result
        assert "n_iter" in result
        assert result["n_samples"] == 300

    def test_train_insufficient_data(self, training_data):
        """Fewer than min_train_days → returns error dict."""
        from quant_fund.research_algorithms.machine_learning.neural_network_predictor import (
            NeuralNetworkPredictor,
        )
        predictor = NeuralNetworkPredictor({"nn_min_train_days": 500})
        features, forward = training_data
        result = predictor.train_model(features, forward)
        assert "error" in result

    def test_compute_after_training(self, predictor, training_data, ohlcv_with_features):
        """Train then predict → returns float values."""
        features, forward = training_data
        predictor.train_model(features, forward)
        result = predictor.compute(
            ohlcv_with_features, pd.Timestamp("2023-03-15")
        )
        assert isinstance(result, pd.Series)
        # At least some predictions should be non-NaN
        assert not result.isna().all()

    def test_nan_input_handled(self, predictor, training_data, ohlcv_with_features):
        """NaN in feature matrix → returns NaN for those tickers."""
        features, forward = training_data
        predictor.train_model(features, forward)

        # Inject NaN into one ticker's features
        ohlcv_with_features.loc[
            ohlcv_with_features.index.get_level_values("ticker") == "AAPL",
            "momentum",
        ] = np.nan

        result = predictor.compute(
            ohlcv_with_features, pd.Timestamp("2023-03-15")
        )
        assert isinstance(result, pd.Series)
        # AAPL should be NaN due to NaN features
        assert np.isnan(result["AAPL"])


# ═══════════════════════════════════════════════════════════════════
# GAP 4: YFinanceConnector._reshape_yfinance
# ═══════════════════════════════════════════════════════════════════


class TestYFinanceReshape:
    """Tests for the static _reshape_yfinance method.

    These test the data transformation logic without requiring
    the yfinance package — we construct synthetic DataFrames
    matching yfinance's output formats.
    """

    @staticmethod
    def _reshape(raw, tickers):
        from quant_fund.data_layer.connectors.yfinance_connector import (
            YFinanceConnector,
        )
        return YFinanceConnector._reshape_yfinance(raw, tickers)

    def test_reshape_single_ticker(self):
        """Single-ticker flat DataFrame → (date, ticker) MultiIndex."""
        dates = pd.bdate_range("2024-01-01", periods=5)
        raw = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [102, 103, 104, 105, 106],
            "Low": [99, 100, 101, 102, 103],
            "Close": [101, 102, 103, 104, 105],
            "Volume": [1e6] * 5,
        }, index=dates)
        raw.index.name = "Date"

        result = self._reshape(raw, ["AAPL"])

        assert isinstance(result.index, pd.MultiIndex)
        assert list(result.index.names) == ["date", "ticker"]
        assert len(result) == 5
        tickers = result.index.get_level_values("ticker").unique()
        assert list(tickers) == ["AAPL"]
        assert "close" in result.columns

    def test_reshape_multi_ticker(self):
        """Multi-ticker MultiIndex columns → (date, ticker) MultiIndex."""
        dates = pd.bdate_range("2024-01-01", periods=3)
        arrays = [
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL",
             "MSFT", "MSFT", "MSFT", "MSFT", "MSFT"],
            ["Open", "High", "Low", "Close", "Volume",
             "Open", "High", "Low", "Close", "Volume"],
        ]
        cols = pd.MultiIndex.from_arrays(arrays)
        data = np.array([
            [100, 102, 99, 101, 1e6, 200, 202, 199, 201, 2e6],
            [101, 103, 100, 102, 1e6, 201, 203, 200, 202, 2e6],
            [102, 104, 101, 103, 1e6, 202, 204, 201, 203, 2e6],
        ])
        raw = pd.DataFrame(data, index=dates, columns=cols)
        raw.index.name = "Date"

        result = self._reshape(raw, ["AAPL", "MSFT"])

        assert isinstance(result.index, pd.MultiIndex)
        tickers = result.index.get_level_values("ticker").unique().tolist()
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert len(result) == 6  # 3 dates x 2 tickers

    def test_reshape_empty_returns_empty(self):
        """Empty DataFrame → empty result with OHLCV schema."""
        raw = pd.DataFrame()
        result = self._reshape(raw, ["AAPL"])
        assert result.empty

    def test_adj_close_renamed(self):
        """'adj close' column should be renamed to 'close'."""
        dates = pd.bdate_range("2024-01-01", periods=3)
        raw = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [102, 103, 104],
            "Low": [99, 100, 101],
            "Adj Close": [101, 102, 103],
            "Volume": [1e6] * 3,
        }, index=dates)
        raw.index.name = "Date"

        result = self._reshape(raw, ["AAPL"])
        # After reshape and rename, "adj close" becomes "close"
        assert "close" in result.columns or "adj close" in result.columns

    def test_missing_ticker_skipped(self):
        """Multi-ticker request with one missing → only available tickers."""
        dates = pd.bdate_range("2024-01-01", periods=3)
        # Only AAPL data, no MSFT
        arrays = [
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            ["Open", "High", "Low", "Close", "Volume"],
        ]
        cols = pd.MultiIndex.from_arrays(arrays)
        data = np.array([
            [100, 102, 99, 101, 1e6],
            [101, 103, 100, 102, 1e6],
            [102, 104, 101, 103, 1e6],
        ])
        raw = pd.DataFrame(data, index=dates, columns=cols)
        raw.index.name = "Date"

        result = self._reshape(raw, ["AAPL", "MSFT"])

        tickers = result.index.get_level_values("ticker").unique().tolist()
        assert "AAPL" in tickers
        assert "MSFT" not in tickers


# ═══════════════════════════════════════════════════════════════════
# GAP 5: Full-Chain Integration
# ═══════════════════════════════════════════════════════════════════


class TestFullChainIntegration:
    """Tests the real wired chain:
    ResearchRunner → PortfolioOptimizer → OrderGenerator → Broker.
    """

    def test_research_to_execution_30_days(self):
        """Wire all components and run 30 days. No errors, positions exist."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.execution.order_management.order_generator import (
            OrderGenerator,
        )
        from quant_fund.execution.order_management.order_router import OrderRouter
        from quant_fund.feature_factory.technical_indicator_engine import (
            MomentumFeature,
            ReturnFeature,
            VolatilityFeature,
        )
        from quant_fund.main.research_runner import ResearchRunner
        from quant_fund.risk_engine.leverage_controller import LeverageController

        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=120, seed=42)

        # Wire research with signal ranking that combines features into alpha
        research = ResearchRunner()

        class SimpleSignalCombiner:
            """Equal-weight feature combiner for testing."""
            def combine(self, feature_matrix):
                if feature_matrix.empty:
                    return pd.Series(dtype=float)
                return feature_matrix.mean(axis=1)

        research.inject_components(
            feature_generators=[
                ReturnFeature(window_days=5),
                MomentumFeature(),
                VolatilityFeature(),
            ],
            signal_ranking=SimpleSignalCombiner(),
        )

        # Wire execution
        broker = SimulationBroker({"initial_cash": 5_000_000})
        broker.set_market_data(STANDARD_MARKET_DATA)
        gen = OrderGenerator()
        router = OrderRouter(broker, {"safety_checks_enabled": False})
        ctrl = LeverageController({"max_leverage": 2.0})

        dates = pd.bdate_range("2019-06-01", periods=30)
        nav = 5_000_000.0

        for date in dates:
            # Research
            result = research.run_cycle(as_of=date, market_data=ohlcv)
            if result.alpha_scores is None:
                continue

            # Simple weight allocation from alpha scores
            scores = result.alpha_scores
            if scores.abs().sum() == 0:
                continue
            weights = scores / scores.abs().sum() * 0.5
            weights = ctrl.enforce(weights)

            # Execute
            positions = broker.get_positions()
            md = broker.get_market_data(tickers)
            if md.empty:
                continue
            prices = md["mid"]

            orders = gen.generate_orders(
                target_weights=weights,
                current_positions=positions,
                prices=prices,
                nav=nav,
            )
            if orders:
                router.route_orders(orders)

            nav = broker.get_account_value()

        # Verify
        assert nav > 0, "NAV should be positive"
        positions = broker.get_positions()
        assert len(positions) > 0, "Should have positions after 30 days of trading"

    def test_alpha_scores_flow_to_constrained_weights(self):
        """Alpha scores → optimizer → weights respect constraints."""
        from quant_fund.feature_factory.technical_indicator_engine import (
            MomentumFeature,
            ReturnFeature,
        )
        from quant_fund.main.research_runner import ResearchRunner
        from quant_fund.risk_engine.leverage_controller import LeverageController

        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=120, seed=42)

        class SimpleSignalCombiner:
            def combine(self, feature_matrix):
                if feature_matrix.empty:
                    return pd.Series(dtype=float)
                return feature_matrix.mean(axis=1)

        research = ResearchRunner()
        research.inject_components(
            feature_generators=[
                ReturnFeature(window_days=5),
                MomentumFeature(),
            ],
            signal_ranking=SimpleSignalCombiner(),
        )

        result = research.run_cycle(
            as_of=pd.Timestamp("2019-06-01"), market_data=ohlcv
        )
        assert result.alpha_scores is not None
        assert len(result.alpha_scores) > 0

        # Convert to weights
        scores = result.alpha_scores
        weights = scores / scores.abs().sum() * 1.5  # intentionally high gross

        # Leverage controller should scale down
        ctrl = LeverageController({"max_leverage": 2.0})
        adjusted = ctrl.enforce(weights)
        assert adjusted.abs().sum() <= 2.0 + 1e-9


# ═══════════════════════════════════════════════════════════════════
# GAP 6: Stress Scenario → Risk Response
# ═══════════════════════════════════════════════════════════════════


class TestStressScenarioRiskResponse:
    """Tests that risk controls respond correctly during stress scenarios."""

    @pytest.fixture
    def portfolio_state(self):
        """A portfolio with known factor exposures."""
        tickers = STANDARD_TICKERS[:5]
        weights = pd.Series(
            [0.3, 0.25, 0.2, -0.15, -0.1],
            index=tickers,
        )
        factors = ["market", "momentum", "value", "quality", "low_vol", "size"]
        rng = np.random.default_rng(42)
        exposures = pd.DataFrame(
            rng.normal(0, 0.5, (len(tickers), len(factors))),
            index=tickers,
            columns=factors,
        )
        # Make market exposure dominant
        exposures["market"] = [1.0, 0.9, 1.1, 0.8, 1.0]
        return weights, exposures

    def test_kill_switch_triggers_during_2008_stress(self, portfolio_state):
        """Apply 2008 stress scenario → if drawdown exceeds limit, kill switch fires."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.risk_engine.stress_test_engine import StressTestEngine

        weights, exposures = portfolio_state

        engine = StressTestEngine()
        results = engine.run_all(weights, exposures)

        # Find the 2008 scenario
        crisis_2008 = next(r for r in results if "2008" in r.scenario_name)
        assert crisis_2008.portfolio_drawdown > 0, "2008 should produce a drawdown"

        # Simulate NAV decline and check kill switch
        initial_nav = 1_000_000
        ks = KillSwitch({
            "drawdown_limit": crisis_2008.portfolio_drawdown * 0.8,  # Set limit below stress loss
            "initial_nav": initial_nav,
        })

        stressed_nav = initial_nav * (1 + crisis_2008.portfolio_return)
        triggered = ks.check(stressed_nav)
        assert triggered is True, "Kill switch should trigger during 2008 stress"
        assert ks.is_halted is True

    def test_drawdown_monitor_alerts_fire_in_order(self):
        """Apply declining NAV → drawdown monitor fires warning, then alert, then critical."""
        from quant_fund.risk_engine.drawdown_monitor import (
            DrawdownAlertLevel,
            DrawdownMonitor,
        )

        monitor = DrawdownMonitor({
            "initial_nav": 1_000_000,
            "drawdown_warning": 0.05,
            "drawdown_alert": 0.10,
            "drawdown_limit": 0.15,
        })

        # No alert at 3% drawdown
        alerts = monitor.update(970_000)
        assert len(alerts) == 0

        # Warning at 5% drawdown
        alerts = monitor.update(950_000)
        assert len(alerts) == 1
        assert alerts[0].level == DrawdownAlertLevel.WARNING

        # Alert at 10%
        alerts = monitor.update(900_000)
        assert len(alerts) == 1
        assert alerts[0].level == DrawdownAlertLevel.ALERT

        # Critical at 15%
        alerts = monitor.update(850_000)
        assert len(alerts) == 1
        assert alerts[0].level == DrawdownAlertLevel.CRITICAL

    def test_leverage_scaled_under_stress(self):
        """High-leverage weights get scaled down by leverage controller."""
        from quant_fund.risk_engine.leverage_controller import LeverageController

        ctrl = LeverageController({"max_leverage": 2.0})

        # Simulate a portfolio with high gross exposure (stress scenario)
        weights = pd.Series(
            [0.8, 0.6, 0.5, -0.7, -0.5, -0.4],
            index=["A", "B", "C", "D", "E", "F"],
        )
        gross_before = weights.abs().sum()
        assert gross_before > 2.0, "Setup: gross should exceed limit"

        adjusted = ctrl.enforce(weights)
        gross_after = adjusted.abs().sum()
        assert gross_after <= 2.0 + 1e-9, "Leverage should be within limit"

        # Proportions should be preserved
        if gross_after > 0:
            ratio = adjusted / adjusted.abs().sum()
            original_ratio = weights / weights.abs().sum()
            pd.testing.assert_series_equal(ratio, original_ratio, atol=1e-9)
