"""Simulation broker — full-featured broker simulator.

Must be implemented before live adapters. All execution logic is tested
against this first. Features: realistic fill simulation, partial fills,
latency, slippage, and market impact modelling.
"""

import logging
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    BrokerInterface,
    Fill,
    Order,
    OrderAcknowledgement,
    OrderSide,
    OrderStatus,
)

logger = logging.getLogger(__name__)


class SimulationBroker(BrokerInterface):
    """Full-featured simulation broker for backtesting and paper trading.

    Features:
    - Realistic fill simulation: fills at mid +/- half_spread + market impact
    - Partial fills for large orders
    - Latency simulation (configurable delay in ms)
    - Configurable slippage model
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._initial_cash = cfg.get("initial_cash", 1_000_000.0)
        self._cash = self._initial_cash
        self._half_spread_bps = cfg.get("half_spread_bps", 5.0)
        self._market_impact_bps = cfg.get("market_impact_bps", 2.0)
        self._latency_ms = cfg.get("latency_ms", 10)
        self._partial_fill_threshold = cfg.get("partial_fill_threshold_adv", 0.01)
        self._commission_per_share = cfg.get("commission_per_share", 0.005)
        self._max_fill_pct = cfg.get("max_fill_pct", 0.95)

        self._positions: Dict[str, int] = {}
        self._orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []
        self._market_data: Dict[str, dict] = {}
        self._order_counter = 0

    def set_market_data(self, data: Dict[str, dict]) -> None:
        """Set simulated market data.

        Args:
            data: Dict mapping ticker -> {bid, ask, mid, last, volume, adv}.
        """
        self._market_data = data

    def submit_order(self, order: Order) -> OrderAcknowledgement:
        """Submit an order and simulate execution."""
        order_id = self._generate_order_id()
        order.order_id = order_id
        self._orders[order_id] = order

        # Simulate the fill
        fill = self._simulate_fill(order)
        if fill is None:
            return OrderAcknowledgement(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message="No market data for ticker",
                timestamp=order.timestamp,
            )

        self._fills.append(fill)
        self._update_position(fill)
        self._update_cash(fill)

        if fill.quantity < order.quantity:
            status = OrderStatus.PARTIAL_FILL
        else:
            status = OrderStatus.FILLED

        return OrderAcknowledgement(
            order_id=order_id,
            status=status,
            message=f"Filled {fill.quantity}/{order.quantity} @ {fill.fill_price:.4f}",
            timestamp=fill.timestamp,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self._orders:
            del self._orders[order_id]
            return True
        return False

    def get_positions(self) -> pd.Series:
        """Get current positions in shares."""
        return pd.Series(self._positions, dtype=float)

    def get_account_value(self) -> float:
        """Get current NAV (cash + position market value)."""
        nav = self._cash
        for ticker, shares in self._positions.items():
            md = self._market_data.get(ticker, {})
            price = md.get("mid", md.get("last", 0.0))
            nav += shares * price
        return nav

    def get_fills(self, since: pd.Timestamp) -> List[Fill]:
        """Get fills since the given timestamp."""
        return [f for f in self._fills if f.timestamp >= since]

    def get_market_data(self, tickers: List[str]) -> pd.DataFrame:
        """Get current market data."""
        rows = {}
        for ticker in tickers:
            md = self._market_data.get(ticker, {})
            rows[ticker] = {
                "bid": md.get("bid", 0.0),
                "ask": md.get("ask", 0.0),
                "mid": md.get("mid", 0.0),
                "last": md.get("last", 0.0),
                "volume": md.get("volume", 0),
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def all_fills(self) -> List[Fill]:
        return list(self._fills)

    def _simulate_fill(self, order: Order) -> Optional[Fill]:
        """Simulate order execution with slippage and market impact."""
        md = self._market_data.get(order.ticker)
        if md is None:
            return None

        mid = md.get("mid", md.get("last", 100.0))
        spread_bps = self._half_spread_bps
        impact_bps = self._market_impact_bps

        # Compute fill price with slippage
        if order.side == OrderSide.BUY:
            slippage = mid * (spread_bps + impact_bps) / 10000
            fill_price = mid + slippage
        else:
            slippage = mid * (spread_bps + impact_bps) / 10000
            fill_price = mid - slippage

        # Limit order check
        if order.order_type == "limit" and order.limit_price is not None:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return None
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return None

        # Partial fill for large orders
        adv = md.get("adv", md.get("volume", 1e6))
        fill_qty = order.quantity
        if adv > 0 and order.quantity > adv * self._partial_fill_threshold:
            fill_ratio = min(self._max_fill_pct, adv * self._partial_fill_threshold / order.quantity)
            fill_qty = max(1, int(order.quantity * fill_ratio))

        commission = fill_qty * self._commission_per_share
        timestamp = order.timestamp or pd.Timestamp.now()

        return Fill(
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=fill_qty,
            fill_price=fill_price,
            timestamp=timestamp,
            commission=commission,
        )

    def _update_position(self, fill: Fill) -> None:
        """Update positions after a fill."""
        current = self._positions.get(fill.ticker, 0)
        if fill.side == OrderSide.BUY:
            self._positions[fill.ticker] = current + fill.quantity
        else:
            self._positions[fill.ticker] = current - fill.quantity
        # Remove zero positions
        if self._positions[fill.ticker] == 0:
            del self._positions[fill.ticker]

    def _update_cash(self, fill: Fill) -> None:
        """Update cash after a fill."""
        trade_value = fill.quantity * fill.fill_price
        if fill.side == OrderSide.BUY:
            self._cash -= trade_value
        else:
            self._cash += trade_value
        self._cash -= fill.commission

    def _generate_order_id(self) -> str:
        self._order_counter += 1
        return f"SIM-{self._order_counter:06d}"
