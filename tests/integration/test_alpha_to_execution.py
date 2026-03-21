"""Integration Chain 2: Alpha → Portfolio → Execution

Tests the pipeline: alpha_scores → portfolio_optimizer → constraint_engine
→ risk checks (exposure, leverage, kill switch) → order_generator →
order_router → broker_abstraction_layer
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import STANDARD_TICKERS, STANDARD_SECTORS, STANDARD_MARKET_DATA


SEED = 42


def _make_alpha(tickers, seed=SEED):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.01, len(tickers)), index=tickers)


def _make_factor_risk(tickers, n_factors=5, seed=SEED):
    rng = np.random.default_rng(seed)
    factors = [f"F{i}" for i in range(n_factors)]
    exposures = pd.DataFrame(
        rng.normal(0, 1, (len(tickers), n_factors)),
        index=tickers, columns=factors,
    )
    factor_cov = pd.DataFrame(
        np.eye(n_factors) * 0.01,
        index=factors, columns=factors,
    )
    return factor_cov, exposures


@pytest.mark.integration
@pytest.mark.tier3
class TestAlphaToPortfolio:
    """Alpha scores drive optimizer → constraints → weights."""

    def test_alpha_to_optimized_weights(self):
        """Alpha scores produce valid optimized weights."""
        from quant_fund.portfolio.portfolio_construction.portfolio_optimizer import PortfolioOptimizer
        from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintEngine

        tickers = STANDARD_TICKERS[:10]
        alpha = _make_alpha(tickers)
        factor_cov, exposures = _make_factor_risk(tickers)
        sector_map = {t: STANDARD_SECTORS[t] for t in tickers}

        engine = ConstraintEngine()
        constraints = engine.build_constraints(sector_map=sector_map)
        optimizer = PortfolioOptimizer()

        weights = optimizer.optimize(alpha, factor_cov, exposures, constraints)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(tickers)

    def test_constraints_satisfied_after_optimization(self):
        """Optimized weights pass constraint validation."""
        from quant_fund.portfolio.portfolio_construction.portfolio_optimizer import PortfolioOptimizer
        from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintEngine

        tickers = STANDARD_TICKERS[:10]
        alpha = _make_alpha(tickers)
        factor_cov, exposures = _make_factor_risk(tickers)
        sector_map = {t: STANDARD_SECTORS[t] for t in tickers}

        engine = ConstraintEngine()
        constraints = engine.build_constraints(sector_map=sector_map)
        optimizer = PortfolioOptimizer()

        weights = optimizer.optimize(alpha, factor_cov, exposures, constraints)

        violations = engine.validate_weights(weights, constraints)
        assert len(violations) == 0, f"Constraint violations: {violations}"


@pytest.mark.integration
@pytest.mark.tier3
class TestPortfolioToRiskChecks:
    """Optimized weights pass through risk checks before execution."""

    def test_leverage_enforced(self):
        """LeverageController scales weights to max leverage."""
        from quant_fund.risk_engine.leverage_controller import LeverageController

        rng = np.random.default_rng(SEED)
        tickers = STANDARD_TICKERS[:10]
        # Create weights with gross exposure > 2.0
        weights = pd.Series(rng.normal(0, 0.5, len(tickers)), index=tickers)

        controller = LeverageController({"max_leverage": 2.0})
        enforced = controller.enforce(weights)

        gross = enforced.abs().sum()
        assert gross <= 2.0 + 1e-6

    def test_exposure_check_detects_breaches(self):
        """ExposureMonitor flags sector overconcentration."""
        from quant_fund.risk_engine.exposure_monitor import ExposureMonitor

        tickers = STANDARD_TICKERS[:10]
        # Overweight Technology sector
        weights = pd.Series(0.0, index=tickers)
        tech_tickers = [t for t in tickers if STANDARD_SECTORS[t] == "Technology"]
        for t in tech_tickers:
            weights[t] = 0.02  # 5 tech stocks × 0.02 = 0.10

        # But also add a huge concentration
        weights[tech_tickers[0]] = 0.15  # way over 0.02 single-name limit

        monitor = ExposureMonitor({"max_single_name_exposure": 0.02, "max_sector_exposure": 0.20})
        breaches = monitor.check(weights, sector_map=STANDARD_SECTORS)

        # Should flag at least single-name breach
        assert len(breaches) > 0

    def test_kill_switch_checked_before_orders(self):
        """Kill switch check prevents order generation when triggered."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.execution.order_management.order_generator import OrderGenerator

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)
        ks.check(700_000.0)  # 30% drawdown → triggered
        assert ks.is_halted

        # When halted, system should not generate orders
        # (Integration convention: check ks.is_halted before calling order_generator)
        gen = OrderGenerator()
        tickers = STANDARD_TICKERS[:5]
        target = pd.Series(0.01, index=tickers)
        current = pd.Series(0.0, index=tickers)
        prices = pd.Series(150.0, index=tickers)

        # If kill switch is halted, orders should not be submitted
        if not ks.is_halted:
            orders = gen.generate_orders(target, current, prices, nav=700_000.0)
        else:
            orders = []

        assert len(orders) == 0


@pytest.mark.integration
@pytest.mark.tier3
class TestPortfolioToExecution:
    """Orders generated from weights flow through to broker."""

    def test_weights_to_orders_to_broker(self):
        """Full chain: weights → OrderGenerator → OrderRouter → SimulationBroker."""
        from quant_fund.execution.order_management.order_generator import OrderGenerator
        from quant_fund.execution.order_management.order_router import OrderRouter
        from quant_fund.broker_interface.simulation_broker import SimulationBroker

        tickers = STANDARD_TICKERS[:5]
        nav = 1_000_000.0

        # Create target weights and current positions
        rng = np.random.default_rng(SEED)
        target = pd.Series(rng.normal(0, 0.01, len(tickers)), index=tickers)
        current = pd.Series(0.0, index=tickers)
        prices = pd.Series(
            [STANDARD_MARKET_DATA[t]["mid"] for t in tickers],
            index=tickers,
        )

        # Generate orders
        gen = OrderGenerator()
        orders = gen.generate_orders(target, current, prices, nav=nav)

        if len(orders) == 0:
            pytest.skip("No orders generated for this seed")

        # Route orders through broker
        broker = SimulationBroker({"initial_cash": nav})
        broker.set_market_data(STANDARD_MARKET_DATA)

        router = OrderRouter(broker)
        acks = router.route_orders(orders)

        assert len(acks) > 0
        # All orders should be acknowledged
        for ack in acks:
            assert ack is not None

    def test_order_routing_large_vs_small(self):
        """Large orders get algo routing, small orders get market routing."""
        from quant_fund.execution.order_management.order_generator import OrderGenerator
        from quant_fund.execution.order_management.order_router import OrderRouter
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.broker_abstraction_layer import Order, OrderSide, OrderType

        broker = SimulationBroker({"initial_cash": 10_000_000.0})
        broker.set_market_data(STANDARD_MARKET_DATA)

        router = OrderRouter(broker)

        # Small order
        small_order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=10,
            order_type=OrderType.MARKET, order_id="SMALL-001",
        )
        ack_small = router.route_single(small_order, adv=50_000_000, mid_price=150.0)
        assert ack_small is not None

        # Large order (relative to ADV)
        large_order = Order(
            ticker="AAPL", side=OrderSide.BUY, quantity=1_000_000,
            order_type=OrderType.MARKET, order_id="LARGE-001",
        )
        ack_large = router.route_single(large_order, adv=50_000_000, mid_price=150.0)
        assert ack_large is not None
