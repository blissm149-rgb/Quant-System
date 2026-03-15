"""Order safety validator — pre-submission checks on every order.

All orders must pass through this validator before reaching the broker.
Checks: size limits, ADV limits, fat-finger, rate limiting, duplicate
detection, price sanity, quantity validation, and buying power.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderType,
)

logger = logging.getLogger(__name__)


@dataclass
class SafetyRejection:
    """A single safety check rejection."""

    check_name: str
    message: str
    order_ticker: str = ""
    order_qty: int = 0


@dataclass
class SafetyResult:
    """Result of order safety validation."""

    passed: bool
    rejections: List[SafetyRejection] = field(default_factory=list)


class OrderSafetyValidator:
    """Pre-submission safety checks for all orders.

    Wired into OrderRouter — every order passes through here
    before reaching the broker. Rejects dangerous orders with
    clear reasons.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        # Max single order as fraction of NAV
        self._max_order_pct_nav = cfg.get("max_order_pct_nav", 0.05)
        # Max single order as fraction of ADV
        self._max_order_pct_adv = cfg.get("max_order_pct_adv", 0.10)
        # Fat-finger: reject if delta > multiplier * current position
        self._fat_finger_multiplier = cfg.get("fat_finger_multiplier", 10.0)
        # Max orders per ticker per second
        self._max_orders_per_second = cfg.get("max_orders_per_second", 5)
        # Duplicate detection window in seconds
        self._dedup_window_s = cfg.get("dedup_window_s", 60)
        # Limit order price deviation from mid (bps)
        self._max_limit_deviation_bps = cfg.get("max_limit_deviation_bps", 500)
        # Absolute max shares per order
        self._max_shares_per_order = cfg.get("max_shares_per_order", 1_000_000)

        # Rate limiting state
        self._order_timestamps: Dict[str, List[float]] = defaultdict(list)
        # Dedup state: (ticker, side, qty) → last submission time
        self._recent_orders: Dict[Tuple[str, str, int], float] = {}

    def validate(
        self,
        order: Order,
        nav: float = 0.0,
        adv: float = 0.0,
        current_position: float = 0.0,
        mid_price: float = 0.0,
        cash: float = float("inf"),
    ) -> SafetyResult:
        """Validate an order against all safety checks.

        Parameters
        ----------
        order : Order
            The order to validate.
        nav : float
            Current NAV for size checks.
        adv : float
            Average daily volume for the ticker.
        current_position : float
            Current position in shares (positive=long, negative=short).
        mid_price : float
            Current mid price for price sanity and value checks.
        cash : float
            Available cash for buying power check.

        Returns
        -------
        SafetyResult
        """
        rejections: List[SafetyRejection] = []

        # 1. Basic quantity validation
        if order.quantity <= 0:
            rejections.append(SafetyRejection(
                check_name="quantity_positive",
                message=f"Quantity must be > 0, got {order.quantity}",
                order_ticker=order.ticker,
                order_qty=order.quantity,
            ))

        if not order.ticker or not order.ticker.strip():
            rejections.append(SafetyRejection(
                check_name="ticker_valid",
                message="Ticker is empty",
            ))

        # 2. Absolute max shares
        if order.quantity > self._max_shares_per_order:
            rejections.append(SafetyRejection(
                check_name="max_shares",
                message=(
                    f"Quantity {order.quantity:,} exceeds max "
                    f"{self._max_shares_per_order:,} shares"
                ),
                order_ticker=order.ticker,
                order_qty=order.quantity,
            ))

        # 3. Max order as % of NAV
        if nav > 0 and mid_price > 0:
            order_value = order.quantity * mid_price
            order_pct = order_value / nav
            if order_pct > self._max_order_pct_nav:
                rejections.append(SafetyRejection(
                    check_name="max_pct_nav",
                    message=(
                        f"Order value ${order_value:,.0f} is "
                        f"{order_pct:.1%} of NAV (limit {self._max_order_pct_nav:.1%})"
                    ),
                    order_ticker=order.ticker,
                    order_qty=order.quantity,
                ))

        # 4. Max order as % of ADV
        if adv > 0 and order.quantity > adv * self._max_order_pct_adv:
            pct_adv = order.quantity / adv
            rejections.append(SafetyRejection(
                check_name="max_pct_adv",
                message=(
                    f"Quantity {order.quantity:,} is {pct_adv:.1%} of "
                    f"ADV {adv:,.0f} (limit {self._max_order_pct_adv:.1%})"
                ),
                order_ticker=order.ticker,
                order_qty=order.quantity,
            ))

        # 5. Fat-finger check
        if (
            current_position != 0
            and abs(order.quantity) > abs(current_position) * self._fat_finger_multiplier
        ):
            rejections.append(SafetyRejection(
                check_name="fat_finger",
                message=(
                    f"Order qty {order.quantity:,} is "
                    f"{abs(order.quantity / current_position):.1f}x current "
                    f"position {current_position:,.0f} "
                    f"(limit {self._fat_finger_multiplier:.0f}x)"
                ),
                order_ticker=order.ticker,
                order_qty=order.quantity,
            ))

        # 6. Limit order price sanity
        if (
            order.order_type == OrderType.LIMIT
            and order.limit_price is not None
            and mid_price > 0
        ):
            deviation_bps = abs(order.limit_price - mid_price) / mid_price * 10000
            if deviation_bps > self._max_limit_deviation_bps:
                rejections.append(SafetyRejection(
                    check_name="limit_price_sanity",
                    message=(
                        f"Limit price {order.limit_price:.2f} is "
                        f"{deviation_bps:.0f} bps from mid {mid_price:.2f} "
                        f"(limit {self._max_limit_deviation_bps:.0f} bps)"
                    ),
                    order_ticker=order.ticker,
                    order_qty=order.quantity,
                ))

        # 7. Buying power check (BUY orders only)
        if order.side == OrderSide.BUY and mid_price > 0:
            cost = order.quantity * mid_price
            if cost > cash:
                rejections.append(SafetyRejection(
                    check_name="buying_power",
                    message=(
                        f"Order cost ${cost:,.0f} exceeds "
                        f"available cash ${cash:,.0f}"
                    ),
                    order_ticker=order.ticker,
                    order_qty=order.quantity,
                ))

        # 8. Rate limiting
        now = time.monotonic()
        ticker_times = self._order_timestamps[order.ticker]
        # Clean old entries
        ticker_times[:] = [t for t in ticker_times if now - t < 1.0]
        if len(ticker_times) >= self._max_orders_per_second:
            rejections.append(SafetyRejection(
                check_name="rate_limit",
                message=(
                    f"Rate limit: {len(ticker_times)} orders for "
                    f"{order.ticker} in last second "
                    f"(limit {self._max_orders_per_second})"
                ),
                order_ticker=order.ticker,
                order_qty=order.quantity,
            ))
        else:
            ticker_times.append(now)

        # 9. Duplicate detection
        dedup_key = (order.ticker, order.side.value, order.quantity)
        last_time = self._recent_orders.get(dedup_key)
        if last_time is not None and (now - last_time) < self._dedup_window_s:
            rejections.append(SafetyRejection(
                check_name="duplicate",
                message=(
                    f"Duplicate order: same (ticker={order.ticker}, "
                    f"side={order.side.value}, qty={order.quantity}) "
                    f"submitted {now - last_time:.0f}s ago "
                    f"(window {self._dedup_window_s}s)"
                ),
                order_ticker=order.ticker,
                order_qty=order.quantity,
            ))
        else:
            self._recent_orders[dedup_key] = now

        passed = len(rejections) == 0
        if not passed:
            for r in rejections:
                logger.warning(
                    "Order REJECTED [%s] %s: %s",
                    r.check_name, order.ticker, r.message,
                )

        return SafetyResult(passed=passed, rejections=rejections)

    def clear_state(self) -> None:
        """Clear rate limiting and dedup state (for testing)."""
        self._order_timestamps.clear()
        self._recent_orders.clear()
