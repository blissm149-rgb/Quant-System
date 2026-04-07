"""Convergence loop — order convergence toward target weights.

Extracted from TradingEngine._convergence_tick() to reduce the
monolithic orchestrator to a thin dispatcher.

The convergence loop:
1. Reads current target weights and broker positions
2. Runs risk gates (cascade or inline)
3. Applies capacity constraints (market impact filter)
4. Generates and routes orders
5. Records fills and execution quality
"""

import logging
from typing import TYPE_CHECKING, Dict, Optional, Set

import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    BrokerInterface,
    OrderStatus,
)
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.infrastructure.event_bus import (
    Event,
    EventBus,
    EventPriority,
    EventType,
)
from quant_fund.risk_engine.risk_cascade_coordinator import RiskCascadeCoordinator

if TYPE_CHECKING:
    from quant_fund.infrastructure.system_state_machine import SystemStateMachine

logger = logging.getLogger(__name__)


def run_convergence_tick(
    *,
    target_weights: Optional[pd.Series],
    broker: Optional[BrokerInterface],
    order_generator: Optional[OrderGenerator],
    order_router: Optional[OrderRouter],
    event_bus: EventBus,
    state_machine: "SystemStateMachine",
    risk_cascade=None,
    leverage_controller=None,
    exposure_monitor=None,
    capacity_model=None,
    execution_quality_monitor=None,
    trade_recorder=None,
    strategy_allocator=None,
    strategy_nav_allocations: Optional[Dict[str, float]] = None,
    strategy_id: str = "default",
    sector_map: Optional[Dict[str, str]] = None,
    factor_exposures: Optional[pd.DataFrame] = None,
    volatility_estimates: Optional[pd.Series] = None,
    max_impact_bps: float = 50.0,
) -> None:
    """One iteration of the portfolio convergence loop.

    1. Get current target weights
    2. Get current positions from broker
    3. Enforce leverage constraints
    4. Check exposure limits (may block orders)
    5. Generate orders for delta
    6. Route orders
    """
    if target_weights is None:
        return
    if broker is None or order_generator is None or order_router is None:
        return

    nav = broker.get_account_value()
    if nav <= 0:
        return

    # Scale NAV by strategy allocation if allocator is active
    if (
        strategy_allocator is not None
        and strategy_nav_allocations
        and strategy_id in strategy_nav_allocations
    ):
        alloc_fraction = strategy_nav_allocations[strategy_id]
        if alloc_fraction <= 0:
            return
        nav = nav * alloc_fraction

    current_positions = broker.get_positions()
    prices = pd.Series(dtype=float)
    tickers = list(target_weights.index)
    if tickers:
        md = broker.get_market_data(tickers)
        if not md.empty and "mid" in md.columns:
            prices = md["mid"]

    # --- Risk gate ---
    working_weights = _apply_risk_gate(
        target_weights=target_weights,
        nav=nav,
        risk_cascade=risk_cascade,
        leverage_controller=leverage_controller,
        exposure_monitor=exposure_monitor,
        event_bus=event_bus,
        state_machine=state_machine,
        sector_map=sector_map,
        factor_exposures=factor_exposures,
    )
    if working_weights is None:
        return  # Orders blocked by risk gate

    # --- Capacity constraint: scale high-impact trades ---
    working_weights = _apply_capacity_constraints(
        working_weights=working_weights,
        nav=nav,
        prices=prices,
        tickers=tickers,
        broker=broker,
        current_positions=current_positions,
        capacity_model=capacity_model,
        volatility_estimates=volatility_estimates,
        max_impact_bps=max_impact_bps,
    )

    orders = order_generator.generate_orders(
        target_weights=working_weights,
        current_positions=current_positions,
        prices=prices,
        nav=nav,
        strategy_id=strategy_id,
    )

    # Record orders for compliance audit trail
    if trade_recorder is not None:
        for order in orders:
            trade_recorder.record_order(
                strategy_id=getattr(order, "strategy_id", strategy_id),
                ticker=order.ticker,
                side=order.side.value,
                quantity=order.quantity,
                order_type=order.order_type.value,
            )

    if orders:
        _route_and_record(
            orders=orders,
            order_router=order_router,
            broker=broker,
            event_bus=event_bus,
            execution_quality_monitor=execution_quality_monitor,
            trade_recorder=trade_recorder,
            prices=prices,
        )


def _apply_risk_gate(
    *,
    target_weights: pd.Series,
    nav: float,
    risk_cascade,
    leverage_controller,
    exposure_monitor,
    event_bus: EventBus,
    state_machine,
    sector_map,
    factor_exposures,
) -> Optional[pd.Series]:
    """Apply risk gates. Returns adjusted weights, or None if orders blocked."""
    # If no cascade provided, create a default one from available components
    cascade = risk_cascade
    if cascade is None:
        cascade = RiskCascadeCoordinator(
            leverage_controller=leverage_controller,
            exposure_monitor=exposure_monitor,
        )

    cascade_result = cascade.run_cascade(
        nav, target_weights, sector_map, factor_exposures,
    )
    if cascade_result.orders_blocked:
        action = "halt" if cascade_result.kill_switch_triggered else "exposure_breach"
        event_bus.publish(
            Event(
                event_type=EventType.RISK_CHECK,
                payload={
                    "action": action,
                    "kill_switch": cascade_result.kill_switch_triggered,
                    "breaches": [
                        {
                            "type": b.exposure_type,
                            "current": b.current_value,
                            "limit": b.limit,
                        }
                        for b in cascade_result.exposure_breaches
                    ],
                },
                source="convergence_loop",
                priority=EventPriority.HIGH,
            )
        )
        if cascade_result.kill_switch_triggered:
            from quant_fund.infrastructure.system_state_machine import SystemState
            state_machine.try_transition(
                SystemState.RISK_HALT,
                reason=f"Kill switch triggered at NAV={nav:.2f}",
            )
        return None
    return cascade_result.adjusted_weights


def _apply_capacity_constraints(
    *,
    working_weights: pd.Series,
    nav: float,
    prices: pd.Series,
    tickers: list,
    broker: BrokerInterface,
    current_positions: pd.Series,
    capacity_model,
    volatility_estimates: Optional[pd.Series],
    max_impact_bps: float,
) -> pd.Series:
    """Scale down high-impact trades using the capacity model."""
    if capacity_model is None or prices.empty:
        return working_weights

    current_weight = pd.Series(dtype=float)
    if nav > 0 and not current_positions.empty:
        current_value = current_positions * prices.reindex(
            current_positions.index
        ).fillna(0)
        current_weight = current_value / nav

    weight_delta = working_weights.subtract(current_weight, fill_value=0.0).abs()
    trade_notional = weight_delta * nav

    adv_series = pd.Series(dtype=float)
    if tickers:
        md = broker.get_market_data(tickers)
        if not md.empty and "adv" in md.columns:
            adv_series = md["adv"]

    vol = volatility_estimates
    if vol is None:
        vol = pd.Series(0.02, index=trade_notional.index)

    if not adv_series.empty:
        impact = capacity_model.estimate_impact_portfolio(
            trade_notional, adv_series, vol,
        )
        for ticker in impact.index:
            if impact[ticker] > max_impact_bps and ticker in working_weights.index:
                midpoint = (
                    working_weights[ticker] + current_weight.get(ticker, 0.0)
                ) / 2.0
                working_weights = working_weights.copy()
                working_weights[ticker] = midpoint
                logger.info(
                    "Capacity constraint: %s impact %.1f bps > %.1f, "
                    "scaling toward current weight",
                    ticker, impact[ticker], max_impact_bps,
                )

    return working_weights


def _route_and_record(
    *,
    orders,
    order_router: OrderRouter,
    broker: BrokerInterface,
    event_bus: EventBus,
    execution_quality_monitor,
    trade_recorder,
    prices: pd.Series,
) -> None:
    """Route orders and record fills + execution quality."""
    acks = order_router.route_orders(orders)
    fills = sum(
        1
        for a in acks
        if a.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL)
    )
    logger.info("Convergence: %d orders, %d fills", len(orders), fills)

    order_by_id = {o.order_id: o for o in orders}

    for ack in acks:
        if ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL):
            event_bus.publish(
                Event(
                    event_type=EventType.ORDER_FILL,
                    payload={
                        "order_id": ack.order_id,
                        "status": ack.status.value,
                    },
                    source="convergence_loop",
                    priority=EventPriority.HIGH,
                )
            )

            order = order_by_id.get(ack.order_id)
            if order is not None:
                decision_price = prices.get(order.ticker, 0.0)
                fill_price = decision_price
                recent_fills = getattr(broker, "_fills", [])
                for f in reversed(recent_fills):
                    if f.order_id == ack.order_id:
                        fill_price = f.fill_price
                        break

                if execution_quality_monitor is not None:
                    execution_quality_monitor.record_execution(
                        order_id=ack.order_id,
                        ticker=order.ticker,
                        side=order.side.value,
                        target_qty=order.quantity,
                        filled_qty=order.quantity,
                        decision_price=decision_price,
                        fill_price=fill_price,
                    )

                if trade_recorder is not None:
                    trade_recorder.record_fill(
                        order_id=ack.order_id,
                        ticker=order.ticker,
                        side=order.side.value,
                        quantity=order.quantity,
                        fill_price=fill_price,
                    )
