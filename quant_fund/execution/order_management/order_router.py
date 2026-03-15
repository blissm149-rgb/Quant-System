"""Order router — routes orders through execution algorithms to the broker.

Selects the appropriate execution algorithm (VWAP/TWAP/liquidity-seeking)
based on order characteristics, then submits child orders through the
broker abstraction layer. All orders pass through OrderSafetyValidator
before reaching the broker.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    BrokerInterface,
    Fill,
    Order,
    OrderAcknowledgement,
    OrderSide,
    OrderStatus,
    OrderType,
)
from quant_fund.execution.execution_algorithms.vwap_execution import (
    VWAPExecution,
    VWAPPlan,
)
from quant_fund.execution.execution_algorithms.twap_execution import (
    TWAPExecution,
    TWAPPlan,
)
from quant_fund.execution.order_management.order_safety_validator import (
    OrderSafetyValidator,
)

logger = logging.getLogger(__name__)


@dataclass
class RoutedOrder:
    """An order with routing and execution state."""

    order: Order
    acknowledgement: Optional[OrderAcknowledgement] = None
    algo: str = "direct"  # direct, vwap, twap, liquidity_seeking
    status: OrderStatus = OrderStatus.PENDING


class OrderRouter:
    """Routes orders to the broker via execution algorithms.

    Selection logic:
    - VWAP orders → VWAPExecution algorithm
    - TWAP orders → TWAPExecution algorithm
    - Market/Limit orders → direct submission to broker
    - Large orders (>1% ADV) → auto-routed to VWAP

    All submissions go through BrokerInterface — never direct to adapters.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        config: Optional[dict] = None,
    ):
        self._broker = broker
        cfg = config or {}
        self._large_order_threshold_adv = cfg.get("large_order_threshold_adv", 0.01)
        self._default_algo = cfg.get("default_algo", "vwap")
        self._safety_enabled = cfg.get("safety_checks_enabled", True)

        self._vwap = VWAPExecution(cfg.get("vwap", {}))
        self._twap = TWAPExecution(cfg.get("twap", {}))
        self._safety = OrderSafetyValidator(cfg.get("safety", {}))

        self._active_orders: Dict[str, RoutedOrder] = {}
        self._completed_orders: List[RoutedOrder] = []
        self._nav: float = cfg.get("nav", 0.0)
        self._cash: float = cfg.get("cash", float("inf"))

    def update_context(self, nav: float = 0.0, cash: float = float("inf")) -> None:
        """Update NAV and cash for safety checks."""
        self._nav = nav
        self._cash = cash

    def route_orders(
        self,
        orders: List[Order],
        adv: Optional[Dict[str, float]] = None,
        positions: Optional[Dict[str, float]] = None,
        prices: Optional[Dict[str, float]] = None,
    ) -> List[OrderAcknowledgement]:
        """Route a list of orders for execution.

        Parameters
        ----------
        orders : list of Order
            Orders to execute.
        adv : dict, optional
            Mapping ticker → average daily volume for algo selection.
        positions : dict, optional
            Mapping ticker → current position in shares.
        prices : dict, optional
            Mapping ticker → current mid price.

        Returns
        -------
        list of OrderAcknowledgement
        """
        adv = adv or {}
        positions = positions or {}
        prices = prices or {}
        acks = []

        for order in orders:
            # Safety check before routing
            if self._safety_enabled:
                result = self._safety.validate(
                    order,
                    nav=self._nav,
                    adv=adv.get(order.ticker, 0.0),
                    current_position=positions.get(order.ticker, 0.0),
                    mid_price=prices.get(order.ticker, 0.0),
                    cash=self._cash,
                )
                if not result.passed:
                    reasons = "; ".join(r.message for r in result.rejections)
                    ack = OrderAcknowledgement(
                        order_id="",
                        status=OrderStatus.REJECTED,
                        message=f"Safety rejected: {reasons}",
                        timestamp=order.timestamp,
                    )
                    acks.append(ack)
                    continue

            algo = self._select_algo(order, adv.get(order.ticker, 0.0))
            ack = self._execute_order(order, algo, adv.get(order.ticker, 1e6))
            acks.append(ack)

        return acks

    def route_single(
        self,
        order: Order,
        adv: float = 0.0,
        current_position: float = 0.0,
        mid_price: float = 0.0,
    ) -> OrderAcknowledgement:
        """Route a single order."""
        if self._safety_enabled:
            result = self._safety.validate(
                order,
                nav=self._nav,
                adv=adv,
                current_position=current_position,
                mid_price=mid_price,
                cash=self._cash,
            )
            if not result.passed:
                reasons = "; ".join(r.message for r in result.rejections)
                return OrderAcknowledgement(
                    order_id="",
                    status=OrderStatus.REJECTED,
                    message=f"Safety rejected: {reasons}",
                    timestamp=order.timestamp,
                )

        algo = self._select_algo(order, adv)
        return self._execute_order(order, algo, adv)

    def get_active_orders(self) -> List[RoutedOrder]:
        """Get all active (non-completed) orders."""
        return list(self._active_orders.values())

    def get_completed_orders(self) -> List[RoutedOrder]:
        """Get all completed orders."""
        return list(self._completed_orders)

    def cancel_all(self) -> int:
        """Cancel all active orders. Returns count of cancellations."""
        cancelled = 0
        for order_id, routed in list(self._active_orders.items()):
            if self._broker.cancel_order(order_id):
                routed.status = OrderStatus.CANCELLED
                self._completed_orders.append(routed)
                del self._active_orders[order_id]
                cancelled += 1
        return cancelled

    def _select_algo(self, order: Order, adv: float) -> str:
        """Select execution algorithm based on order characteristics."""
        # Explicit algo type in order
        if order.order_type == OrderType.VWAP:
            return "vwap"
        if order.order_type == OrderType.TWAP:
            return "twap"

        # Large orders get default algo
        if adv > 0 and order.quantity > adv * self._large_order_threshold_adv:
            return self._default_algo

        # Small orders go direct
        return "direct"

    def _execute_order(
        self, order: Order, algo: str, adv: float
    ) -> OrderAcknowledgement:
        """Execute an order using the selected algorithm."""
        if algo == "vwap":
            return self._execute_vwap(order, adv)
        elif algo == "twap":
            return self._execute_twap(order, adv)
        else:
            return self._execute_direct(order)

    def _execute_direct(self, order: Order) -> OrderAcknowledgement:
        """Submit order directly to broker."""
        # For direct market/limit, override type to market or limit
        if order.order_type in (OrderType.VWAP, OrderType.TWAP):
            order = Order(
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                order_type=OrderType.MARKET,
                limit_price=order.limit_price,
                strategy_id=order.strategy_id,
                timestamp=order.timestamp,
            )

        ack = self._broker.submit_order(order)
        routed = RoutedOrder(order=order, acknowledgement=ack, algo="direct")

        if ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL):
            routed.status = ack.status
            self._completed_orders.append(routed)
        elif ack.status == OrderStatus.SUBMITTED:
            routed.status = OrderStatus.SUBMITTED
            self._active_orders[ack.order_id] = routed
        else:
            routed.status = ack.status
            self._completed_orders.append(routed)

        return ack

    def _execute_vwap(self, order: Order, adv: float) -> OrderAcknowledgement:
        """Execute using VWAP algorithm (simplified: submit as market)."""
        plan = self._vwap.create_plan(
            ticker=order.ticker,
            side=order.side,
            total_shares=order.quantity,
            adv=adv,
        )

        # For simulation, execute the first slice immediately
        child = self._vwap.get_next_child_order(plan, current_price=0.0)
        if child is None:
            return OrderAcknowledgement(
                order_id="",
                status=OrderStatus.REJECTED,
                message="VWAP plan produced no child orders",
            )

        ack = self._broker.submit_order(child)
        routed = RoutedOrder(
            order=order, acknowledgement=ack, algo="vwap",
            status=ack.status,
        )

        if ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL):
            self._completed_orders.append(routed)
        else:
            self._active_orders[ack.order_id] = routed

        return ack

    def _execute_twap(self, order: Order, adv: float) -> OrderAcknowledgement:
        """Execute using TWAP algorithm (simplified: submit as market)."""
        plan = self._twap.create_plan(
            ticker=order.ticker,
            side=order.side,
            total_shares=order.quantity,
        )

        child = self._twap.get_next_child_order(plan)
        if child is None:
            return OrderAcknowledgement(
                order_id="",
                status=OrderStatus.REJECTED,
                message="TWAP plan produced no child orders",
            )

        ack = self._broker.submit_order(child)
        routed = RoutedOrder(
            order=order, acknowledgement=ack, algo="twap",
            status=ack.status,
        )

        if ack.status in (OrderStatus.FILLED, OrderStatus.PARTIAL_FILL):
            self._completed_orders.append(routed)
        else:
            self._active_orders[ack.order_id] = routed

        return ack
