"""Tests for Group H — Execution + Broker Interface.

Covers: broker abstraction, simulation broker, microstructure models,
execution algorithms (VWAP/TWAP/liquidity-seeking), order generator,
and order router.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.broker_interface.broker_abstraction_layer import (
    Fill,
    Order,
    OrderAcknowledgement,
    OrderSide,
    OrderStatus,
    OrderType,
)
from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.microstructure_models.bid_ask_spread_model import (
    BidAskSpreadModel,
)
from quant_fund.execution.microstructure_models.order_book_liquidity_model import (
    OrderBookLevel,
    OrderBookLiquidityModel,
)
from quant_fund.execution.microstructure_models.adverse_selection_model import (
    AdverseSelectionModel,
)
from quant_fund.execution.microstructure_models.fill_probability_model import (
    FillProbabilityModel,
)
from quant_fund.execution.microstructure_models.queue_position_estimator import (
    QueuePositionEstimator,
)
from quant_fund.execution.execution_algorithms.vwap_execution import VWAPExecution
from quant_fund.execution.execution_algorithms.twap_execution import TWAPExecution
from quant_fund.execution.execution_algorithms.liquidity_seeking_execution import (
    LiquiditySeekingExecution,
    LiquidityState,
    Urgency,
)
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter


# ── Helpers ──────────────────────────────────────────────────────────

def make_sim_broker(initial_cash=1_000_000.0):
    broker = SimulationBroker({"initial_cash": initial_cash})
    broker.set_market_data({
        "AAPL": {"bid": 149.0, "ask": 151.0, "mid": 150.0, "last": 150.0, "volume": 50_000_000, "adv": 50_000_000},
        "MSFT": {"bid": 299.0, "ask": 301.0, "mid": 300.0, "last": 300.0, "volume": 30_000_000, "adv": 30_000_000},
        "GOOG": {"bid": 139.0, "ask": 141.0, "mid": 140.0, "last": 140.0, "volume": 20_000_000, "adv": 20_000_000},
    })
    return broker


# ── Simulation Broker ───────────────────────────────────────────────

class TestSimulationBroker:
    def test_submit_market_order_buy(self):
        broker = make_sim_broker()
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=100,
                      timestamp=pd.Timestamp("2024-01-01"))
        ack = broker.submit_order(order)
        assert ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL)
        assert ack.order_id.startswith("SIM-")
        positions = broker.get_positions()
        assert positions.get("AAPL", 0) > 0

    def test_submit_market_order_sell(self):
        broker = make_sim_broker()
        # Buy first
        broker.submit_order(Order(ticker="AAPL", side=OrderSide.BUY, quantity=100,
                                  timestamp=pd.Timestamp("2024-01-01")))
        # Sell
        ack = broker.submit_order(Order(ticker="AAPL", side=OrderSide.SELL, quantity=50,
                                        timestamp=pd.Timestamp("2024-01-01")))
        assert ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL)
        positions = broker.get_positions()
        assert positions.get("AAPL", 0) > 0  # still have some

    def test_no_market_data_rejects(self):
        broker = make_sim_broker()
        order = Order(ticker="UNKNOWN", side=OrderSide.BUY, quantity=100,
                      timestamp=pd.Timestamp("2024-01-01"))
        ack = broker.submit_order(order)
        assert ack.status == OrderStatus.REJECTED

    def test_cash_decreases_on_buy(self):
        broker = make_sim_broker(initial_cash=1_000_000.0)
        initial = broker.cash
        broker.submit_order(Order(ticker="AAPL", side=OrderSide.BUY, quantity=100,
                                  timestamp=pd.Timestamp("2024-01-01")))
        assert broker.cash < initial

    def test_get_fills_since(self):
        broker = make_sim_broker()
        t0 = pd.Timestamp("2024-01-01")
        broker.submit_order(Order(ticker="AAPL", side=OrderSide.BUY, quantity=100, timestamp=t0))
        fills = broker.get_fills(t0)
        assert len(fills) >= 1
        assert fills[0].ticker == "AAPL"

    def test_get_account_value(self):
        broker = make_sim_broker(initial_cash=1_000_000.0)
        nav = broker.get_account_value()
        assert nav == pytest.approx(1_000_000.0, rel=0.01)

    def test_cancel_order(self):
        broker = make_sim_broker()
        # Sim broker fills immediately, so cancel returns False for non-existent
        assert broker.cancel_order("nonexistent") is False

    def test_get_market_data(self):
        broker = make_sim_broker()
        md = broker.get_market_data(["AAPL", "MSFT"])
        assert "AAPL" in md.index
        assert "bid" in md.columns
        assert md.loc["AAPL", "mid"] == 150.0

    def test_slippage_applied(self):
        """Buy fill price should be above mid (slippage)."""
        broker = make_sim_broker()
        broker.submit_order(Order(ticker="AAPL", side=OrderSide.BUY, quantity=100,
                                  timestamp=pd.Timestamp("2024-01-01")))
        fill = broker.all_fills[0]
        assert fill.fill_price > 150.0  # mid is 150

    def test_sell_fill_below_mid(self):
        """Sell fill price should be below mid."""
        broker = make_sim_broker()
        broker.submit_order(Order(ticker="AAPL", side=OrderSide.BUY, quantity=200,
                                  timestamp=pd.Timestamp("2024-01-01")))
        broker.submit_order(Order(ticker="AAPL", side=OrderSide.SELL, quantity=100,
                                  timestamp=pd.Timestamp("2024-01-01")))
        sell_fill = broker.all_fills[-1]
        assert sell_fill.fill_price < 150.0

    def test_commission_charged(self):
        broker = make_sim_broker()
        broker.submit_order(Order(ticker="AAPL", side=OrderSide.BUY, quantity=100,
                                  timestamp=pd.Timestamp("2024-01-01")))
        fill = broker.all_fills[0]
        assert fill.commission > 0


# ── Bid-Ask Spread Model ────────────────────────────────────────────

class TestBidAskSpreadModel:
    def test_quoted_spread(self):
        model = BidAskSpreadModel()
        spread = model.estimate_quoted_spread(99.0, 101.0)
        assert spread == pytest.approx(200.0, rel=0.01)  # 2/100 * 10000

    def test_quoted_spread_zero_prices(self):
        model = BidAskSpreadModel()
        assert model.estimate_quoted_spread(0, 100) == 0.0

    def test_roll_estimator_negative_autocov(self):
        model = BidAskSpreadModel({"min_observations": 5})
        # Create returns with negative autocovariance (bid-ask bounce)
        rng = np.random.default_rng(42)
        n = 100
        bounce = np.array([1, -1] * (n // 2)) * 0.001
        noise = rng.normal(0, 0.0001, n)
        returns = pd.Series(bounce + noise)
        spread = model.estimate_effective_spread_roll(returns)
        assert spread > 0

    def test_from_quotes(self):
        model = BidAskSpreadModel()
        quotes = pd.DataFrame({
            "ticker": ["AAPL"] * 5,
            "bid": [149.0, 149.1, 149.2, 149.0, 149.1],
            "ask": [151.0, 151.1, 151.2, 151.0, 151.1],
        })
        result = model.estimate_from_quotes(quotes)
        assert "AAPL" in result
        assert result["AAPL"].quoted_spread_bps > 0


# ── Order Book Liquidity Model ──────────────────────────────────────

class TestOrderBookLiquidityModel:
    def test_depth_imbalance(self):
        model = OrderBookLiquidityModel()
        assert model.compute_depth_imbalance(1000, 500) == pytest.approx(1/3, rel=0.01)
        assert model.compute_depth_imbalance(500, 1000) == pytest.approx(-1/3, rel=0.01)
        assert model.compute_depth_imbalance(0, 0) == 0.0

    def test_cost_to_trade(self):
        model = OrderBookLiquidityModel()
        levels = [
            OrderBookLevel(price=101.0, size=500),
            OrderBookLevel(price=102.0, size=500),
        ]
        cost = model.estimate_cost_to_trade(300, levels, mid_price=100.0)
        assert cost > 0

    def test_snapshot(self):
        model = OrderBookLiquidityModel()
        bids = [OrderBookLevel(99.0, 1000), OrderBookLevel(98.0, 2000)]
        asks = [OrderBookLevel(101.0, 800), OrderBookLevel(102.0, 1500)]
        snap = model.compute_snapshot("AAPL", bids, asks, mid_price=100.0)
        assert snap.total_bid_depth == 3000
        assert snap.total_ask_depth == 2300
        assert snap.ticker == "AAPL"


# ── Adverse Selection Model ─────────────────────────────────────────

class TestAdverseSelectionModel:
    def test_vpin_balanced(self):
        model = AdverseSelectionModel()
        trades = pd.DataFrame({
            "price": [100.0] * 100,
            "volume": [100] * 100,
            "side": [1, -1] * 50,
        })
        vpin = model.estimate_vpin(trades)
        assert vpin == pytest.approx(0.0, abs=0.05)

    def test_vpin_imbalanced(self):
        model = AdverseSelectionModel()
        trades = pd.DataFrame({
            "price": list(range(100, 200)),
            "volume": [100] * 100,
            "side": [1] * 80 + [-1] * 20,
        })
        vpin = model.estimate_vpin(trades)
        assert vpin > 0.5

    def test_adverse_component(self):
        model = AdverseSelectionModel()
        adverse = model.estimate_adverse_component(10.0, 4.0)
        assert adverse == pytest.approx(6.0)

    def test_full_estimate(self):
        model = AdverseSelectionModel()
        result = model.estimate(
            ticker="AAPL",
            effective_spread_bps=10.0,
            realised_spread_bps=4.0,
        )
        assert result.ticker == "AAPL"
        assert result.adverse_component_bps == pytest.approx(6.0)


# ── Fill Probability Model ──────────────────────────────────────────

class TestFillProbabilityModel:
    def test_aggressive_order_high_fill_prob(self):
        model = FillProbabilityModel()
        result = model.estimate_fill_probability(
            ticker="AAPL", limit_price=151.0, mid_price=150.0,
            order_size=100, side="buy",
        )
        assert result.fill_prob > 0.9

    def test_passive_order_lower_fill_prob(self):
        model = FillProbabilityModel()
        result = model.estimate_fill_probability(
            ticker="AAPL", limit_price=145.0, mid_price=150.0,
            order_size=100, side="buy",
        )
        assert result.fill_prob < 0.5

    def test_optimal_limit_offset(self):
        model = FillProbabilityModel()
        # Patient → positive offset (behind mid)
        offset_patient = model.optimal_limit_offset(10.0, urgency=0.0)
        # Aggressive → negative offset (cross spread)
        offset_aggressive = model.optimal_limit_offset(10.0, urgency=1.0)
        assert offset_patient > offset_aggressive


# ── Queue Position Estimator ────────────────────────────────────────

class TestQueuePositionEstimator:
    def test_basic_estimate(self):
        model = QueuePositionEstimator()
        result = model.estimate("AAPL", order_size=100, depth_ahead=1000)
        assert result.estimated_wait_s > 0
        assert result.queue_depth == 1000
        assert result.fill_before_cancel_prob > 0

    def test_no_queue(self):
        model = QueuePositionEstimator()
        result = model.estimate("AAPL", order_size=100, depth_ahead=0)
        assert result.estimated_wait_s > 0  # still takes time to fill our order
        assert result.fill_before_cancel_prob > 0.9  # very likely to fill


# ── VWAP Execution ──────────────────────────────────────────────────

class TestVWAPExecution:
    def test_create_plan(self):
        vwap = VWAPExecution()
        plan = vwap.create_plan("AAPL", OrderSide.BUY, 10000)
        assert plan.total_shares == 10000
        assert len(plan.slices) == 13
        total_target = sum(s.target_shares for s in plan.slices)
        assert total_target > 0

    def test_get_next_child_order(self):
        vwap = VWAPExecution()
        plan = vwap.create_plan("AAPL", OrderSide.BUY, 10000)
        child = vwap.get_next_child_order(plan, current_price=150.0)
        assert child is not None
        assert child.ticker == "AAPL"
        assert child.side == OrderSide.BUY

    def test_record_fill(self):
        vwap = VWAPExecution()
        plan = vwap.create_plan("AAPL", OrderSide.BUY, 10000)
        first_target = plan.slices[0].target_shares
        vwap.record_fill(plan, first_target)
        assert plan.filled_shares == first_target
        assert plan.slices[0].is_complete

    def test_plan_completes(self):
        vwap = VWAPExecution()
        plan = vwap.create_plan("AAPL", OrderSide.BUY, 100)
        total = sum(s.target_shares for s in plan.slices)
        vwap.record_fill(plan, total)
        assert plan.is_complete


# ── TWAP Execution ──────────────────────────────────────────────────

class TestTWAPExecution:
    def test_create_plan(self):
        twap = TWAPExecution()
        plan = twap.create_plan("MSFT", OrderSide.SELL, 5000, num_slices=5)
        assert plan.total_shares == 5000
        assert plan.num_slices == 5
        total_target = sum(s.target_shares for s in plan.slices)
        assert total_target == 5000

    def test_equal_distribution(self):
        twap = TWAPExecution()
        plan = twap.create_plan("MSFT", OrderSide.SELL, 5000, num_slices=5)
        sizes = [s.target_shares for s in plan.slices]
        assert all(s == 1000 for s in sizes)

    def test_randomization_preserves_total(self):
        twap = TWAPExecution({"randomize_pct": 0.20})
        plan = twap.create_plan("MSFT", OrderSide.BUY, 5000, num_slices=5)
        plan = twap.add_randomization(plan, rng=np.random.default_rng(42))
        total = sum(s.target_shares for s in plan.slices)
        assert total == 5000

    def test_scheduled_times(self):
        twap = TWAPExecution()
        t0 = pd.Timestamp("2024-01-02 09:30:00")
        plan = twap.create_plan("MSFT", OrderSide.BUY, 1000, num_slices=4,
                                window_minutes=60, start_time=t0)
        assert plan.slices[0].scheduled_time == t0
        assert plan.slices[1].scheduled_time > t0


# ── Liquidity Seeking Execution ─────────────────────────────────────

class TestLiquiditySeekingExecution:
    def test_urgency_low_when_ahead(self):
        algo = LiquiditySeekingExecution()
        urgency = algo.compute_urgency(elapsed_pct=0.5, filled_pct=0.6)
        assert urgency == Urgency.LOW

    def test_urgency_critical_when_behind_and_late(self):
        algo = LiquiditySeekingExecution()
        urgency = algo.compute_urgency(elapsed_pct=0.9, filled_pct=0.3)
        assert urgency == Urgency.CRITICAL

    def test_passive_limit_for_low_urgency(self):
        algo = LiquiditySeekingExecution()
        state = LiquidityState(
            ticker="AAPL", mid_price=150.0, spread_bps=10.0,
            available_depth=10000, adv=50_000_000, urgency=Urgency.LOW,
        )
        otype = algo.decide_order_type(state)
        assert otype == OrderType.LIMIT

    def test_market_for_critical_urgency(self):
        algo = LiquiditySeekingExecution()
        state = LiquidityState(
            ticker="AAPL", mid_price=150.0, spread_bps=10.0,
            available_depth=10000, adv=50_000_000, urgency=Urgency.CRITICAL,
        )
        otype = algo.decide_order_type(state)
        assert otype == OrderType.MARKET

    def test_generate_child_order(self):
        algo = LiquiditySeekingExecution()
        state = LiquidityState(
            ticker="AAPL", mid_price=150.0, spread_bps=10.0,
            available_depth=10000, adv=50_000_000, urgency=Urgency.MEDIUM,
        )
        child = algo.generate_child_order("AAPL", OrderSide.BUY, 1000, state)
        assert child is not None
        assert child.quantity > 0


# ── Order Generator ─────────────────────────────────────────────────

class TestOrderGenerator:
    def test_generate_buy_orders(self):
        gen = OrderGenerator({"min_trade_value": 100})
        target = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        current = pd.Series(dtype=float)
        prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})
        orders = gen.generate_orders(target, current, prices, nav=100_000.0)
        assert len(orders) == 2
        assert all(o.side == OrderSide.BUY for o in orders)

    def test_generate_sell_orders(self):
        gen = OrderGenerator({"min_trade_value": 100})
        target = pd.Series({"AAPL": 0.0})
        current = pd.Series({"AAPL": 100.0})
        prices = pd.Series({"AAPL": 150.0})
        orders = gen.generate_orders(target, current, prices, nav=100_000.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    def test_no_orders_when_on_target(self):
        gen = OrderGenerator({"min_trade_value": 100})
        target = pd.Series({"AAPL": 0.5})
        prices = pd.Series({"AAPL": 150.0})
        nav = 100_000.0
        # current shares that match target
        current = pd.Series({"AAPL": nav * 0.5 / 150.0})
        orders = gen.generate_orders(target, current, prices, nav=nav)
        assert len(orders) == 0

    def test_liquidation_orders(self):
        gen = OrderGenerator()
        positions = pd.Series({"AAPL": 100, "MSFT": -50})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})
        orders = gen.generate_liquidation_orders(positions, prices)
        assert len(orders) == 2
        # AAPL is long, should sell
        aapl_order = [o for o in orders if o.ticker == "AAPL"][0]
        assert aapl_order.side == OrderSide.SELL
        # MSFT is short, should buy
        msft_order = [o for o in orders if o.ticker == "MSFT"][0]
        assert msft_order.side == OrderSide.BUY

    def test_zero_nav_returns_empty(self):
        gen = OrderGenerator()
        orders = gen.generate_orders(
            pd.Series({"AAPL": 0.5}),
            pd.Series(dtype=float),
            pd.Series({"AAPL": 150.0}),
            nav=0.0,
        )
        assert len(orders) == 0


# ── Order Router ────────────────────────────────────────────────────

class TestOrderRouter:
    def test_direct_market_order(self):
        broker = make_sim_broker()
        router = OrderRouter(broker)
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=100,
                      order_type=OrderType.MARKET, timestamp=pd.Timestamp("2024-01-01"))
        ack = router.route_single(order)
        assert ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL)

    def test_vwap_routing(self):
        broker = make_sim_broker()
        router = OrderRouter(broker)
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=100,
                      order_type=OrderType.VWAP, timestamp=pd.Timestamp("2024-01-01"))
        ack = router.route_single(order, adv=50_000_000)
        assert ack.order_id != ""

    def test_twap_routing(self):
        broker = make_sim_broker()
        router = OrderRouter(broker)
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=100,
                      order_type=OrderType.TWAP, timestamp=pd.Timestamp("2024-01-01"))
        ack = router.route_single(order, adv=50_000_000)
        assert ack.order_id != ""

    def test_large_order_auto_routes_to_algo(self):
        broker = make_sim_broker()
        router = OrderRouter(broker, {"large_order_threshold_adv": 0.01})
        # Order > 1% of ADV (500_000 > 1% of 50M)
        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=600_000,
                      order_type=OrderType.MARKET, timestamp=pd.Timestamp("2024-01-01"))
        ack = router.route_single(order, adv=50_000_000)
        # Should have been routed through algo (VWAP by default)
        completed = router.get_completed_orders()
        assert len(completed) > 0
        assert completed[0].algo == "vwap"

    def test_route_multiple_orders(self):
        broker = make_sim_broker()
        router = OrderRouter(broker)
        orders = [
            Order(ticker="AAPL", side=OrderSide.BUY, quantity=100,
                  timestamp=pd.Timestamp("2024-01-01")),
            Order(ticker="MSFT", side=OrderSide.BUY, quantity=50,
                  timestamp=pd.Timestamp("2024-01-01")),
        ]
        acks = router.route_orders(orders)
        assert len(acks) == 2

    def test_cancel_all(self):
        broker = make_sim_broker()
        router = OrderRouter(broker)
        # Sim broker fills immediately, so nothing to cancel
        cancelled = router.cancel_all()
        assert cancelled == 0
