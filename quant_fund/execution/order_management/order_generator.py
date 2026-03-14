"""Order generator — converts target portfolio weights to orders.

Takes the difference between target weights and current positions,
applies the capacity/impact model, and produces Order objects
ready for the order router.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderType,
)

logger = logging.getLogger(__name__)


class OrderGenerator:
    """Generates orders from target portfolio weights.

    Computes the trade list by diffing target weights against
    current positions, converting dollar amounts to shares,
    and filtering out sub-minimum trades.

    Flow: optimizer weights → OrderGenerator → Order list → OrderRouter
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._min_trade_value = cfg.get("min_trade_value", 1000.0)
        self._min_trade_shares = cfg.get("min_trade_shares", 1)
        self._default_order_type = OrderType(
            cfg.get("default_order_type", "market")
        )
        self._round_lot_size = cfg.get("round_lot_size", 1)

    def generate_orders(
        self,
        target_weights: pd.Series,
        current_positions: pd.Series,
        prices: pd.Series,
        nav: float,
        strategy_id: str = "",
        order_type: Optional[OrderType] = None,
    ) -> List[Order]:
        """Generate orders to move from current positions to target weights.

        Parameters
        ----------
        target_weights : pd.Series
            Target portfolio weights indexed by ticker. Positive = long.
        current_positions : pd.Series
            Current positions in shares indexed by ticker.
        prices : pd.Series
            Current prices indexed by ticker (mid prices).
        nav : float
            Current net asset value for weight → dollar conversion.
        strategy_id : str
            Strategy identifier for order tagging.
        order_type : OrderType, optional
            Order type override. Defaults to configured default.

        Returns
        -------
        list of Order
        """
        if nav <= 0:
            logger.warning("NAV is zero or negative, cannot generate orders")
            return []

        otype = order_type or self._default_order_type
        orders: List[Order] = []

        # All tickers we need to consider
        all_tickers = set(target_weights.index) | set(current_positions.index)

        for ticker in sorted(all_tickers):
            target_w = target_weights.get(ticker, 0.0)
            current_shares = current_positions.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)

            if price <= 0:
                if target_w != 0:
                    logger.warning("No price for %s, skipping", ticker)
                continue

            # Target shares from weight
            target_dollar = target_w * nav
            target_shares = target_dollar / price

            # Trade size
            delta_shares = target_shares - current_shares
            delta_shares = self._round_shares(delta_shares)

            if abs(delta_shares) < self._min_trade_shares:
                continue

            trade_value = abs(delta_shares * price)
            if trade_value < self._min_trade_value:
                continue

            side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
            orders.append(Order(
                ticker=ticker,
                side=side,
                quantity=abs(int(delta_shares)),
                order_type=otype,
                strategy_id=strategy_id,
                timestamp=pd.Timestamp.now(),
            ))

        logger.info("Generated %d orders for strategy %s", len(orders), strategy_id)
        return orders

    def generate_liquidation_orders(
        self,
        current_positions: pd.Series,
        prices: pd.Series,
        strategy_id: str = "",
    ) -> List[Order]:
        """Generate orders to fully liquidate all positions.

        Used by kill switch or strategy retirement.
        """
        orders: List[Order] = []
        for ticker, shares in current_positions.items():
            if shares == 0:
                continue
            price = prices.get(ticker, 0.0)
            if price <= 0:
                logger.warning("No price for %s during liquidation", ticker)
                continue

            side = OrderSide.SELL if shares > 0 else OrderSide.BUY
            orders.append(Order(
                ticker=str(ticker),
                side=side,
                quantity=abs(int(shares)),
                order_type=OrderType.MARKET,
                strategy_id=strategy_id,
                timestamp=pd.Timestamp.now(),
            ))
        return orders

    def _round_shares(self, shares: float) -> int:
        """Round shares to nearest lot size."""
        if self._round_lot_size <= 1:
            return int(round(shares))
        rounded = round(shares / self._round_lot_size) * self._round_lot_size
        return int(rounded)
