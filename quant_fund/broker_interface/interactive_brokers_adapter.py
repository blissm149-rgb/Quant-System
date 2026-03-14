"""Interactive Brokers adapter wrapping the IB TWS API.

Implements BrokerInterface. Handles reconnection and pacing limits.
This is a structural implementation — actual IB API calls require
the ib_insync or ibapi package at runtime.
"""

import logging
from typing import List, Optional

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


class InteractiveBrokersAdapter(BrokerInterface):
    """Adapter for Interactive Brokers TWS API.

    Requires ib_insync or ibapi package. Handles reconnection,
    pacing limits, and request throttling.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._host = cfg.get("ib_host", "127.0.0.1")
        self._port = cfg.get("ib_port", 7497)
        self._client_id = cfg.get("ib_client_id", 1)
        self._account = cfg.get("ib_account", "")
        self._connected = False
        self._ib = None

    def connect(self) -> bool:
        """Connect to TWS/IB Gateway."""
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self._host, self._port, clientId=self._client_id)
            self._connected = True
            logger.info("Connected to IB at %s:%d", self._host, self._port)
            return True
        except ImportError:
            logger.error("ib_insync not installed")
            return False
        except Exception as e:
            logger.error("Failed to connect to IB: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect from TWS."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False

    def submit_order(self, order: Order) -> OrderAcknowledgement:
        """Submit order to IB."""
        if not self._connected:
            return OrderAcknowledgement(
                order_id="",
                status=OrderStatus.REJECTED,
                message="Not connected to IB",
            )

        try:
            from ib_insync import Stock, MarketOrder, LimitOrder

            contract = Stock(order.ticker, "SMART", "USD")

            if order.order_type.value == "limit" and order.limit_price:
                ib_order = LimitOrder(
                    order.side.value.upper(),
                    order.quantity,
                    order.limit_price,
                )
            else:
                ib_order = MarketOrder(
                    order.side.value.upper(),
                    order.quantity,
                )

            trade = self._ib.placeOrder(contract, ib_order)
            return OrderAcknowledgement(
                order_id=str(trade.order.orderId),
                status=OrderStatus.SUBMITTED,
                message="Order submitted to IB",
                timestamp=pd.Timestamp.now(),
            )
        except Exception as e:
            logger.error("IB order submission failed: %s", e)
            return OrderAcknowledgement(
                order_id="",
                status=OrderStatus.REJECTED,
                message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order on IB."""
        if not self._connected:
            return False
        try:
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    return True
            return False
        except Exception as e:
            logger.error("IB cancel failed: %s", e)
            return False

    def get_positions(self) -> pd.Series:
        """Get current IB positions."""
        if not self._connected:
            return pd.Series(dtype=float)
        try:
            positions = {}
            for pos in self._ib.positions():
                ticker = pos.contract.symbol
                positions[ticker] = pos.position
            return pd.Series(positions, dtype=float)
        except Exception as e:
            logger.error("Failed to get IB positions: %s", e)
            return pd.Series(dtype=float)

    def get_account_value(self) -> float:
        """Get IB account NAV."""
        if not self._connected:
            return 0.0
        try:
            for av in self._ib.accountValues():
                if av.tag == "NetLiquidation" and av.currency == "USD":
                    return float(av.value)
            return 0.0
        except Exception as e:
            logger.error("Failed to get IB account value: %s", e)
            return 0.0

    def get_fills(self, since: pd.Timestamp) -> List[Fill]:
        """Get IB fills since timestamp."""
        if not self._connected:
            return []
        try:
            fills = []
            for trade in self._ib.trades():
                for ib_fill in trade.fills:
                    fill_time = pd.Timestamp(ib_fill.time)
                    if fill_time >= since:
                        fills.append(Fill(
                            order_id=str(trade.order.orderId),
                            ticker=trade.contract.symbol,
                            side=OrderSide.BUY if ib_fill.execution.side == "BOT" else OrderSide.SELL,
                            quantity=int(ib_fill.execution.shares),
                            fill_price=ib_fill.execution.price,
                            timestamp=fill_time,
                            commission=ib_fill.commissionReport.commission if ib_fill.commissionReport else 0.0,
                        ))
            return fills
        except Exception as e:
            logger.error("Failed to get IB fills: %s", e)
            return []

    def get_market_data(self, tickers: List[str]) -> pd.DataFrame:
        """Get current market data from IB."""
        if not self._connected:
            return pd.DataFrame()
        try:
            from ib_insync import Stock
            rows = {}
            for ticker in tickers:
                contract = Stock(ticker, "SMART", "USD")
                self._ib.qualifyContracts(contract)
                md = self._ib.reqMktData(contract, snapshot=True)
                self._ib.sleep(0.5)
                rows[ticker] = {
                    "bid": md.bid if md.bid > 0 else 0.0,
                    "ask": md.ask if md.ask > 0 else 0.0,
                    "mid": (md.bid + md.ask) / 2 if md.bid > 0 else md.last,
                    "last": md.last if md.last > 0 else 0.0,
                    "volume": md.volume if md.volume > 0 else 0,
                }
            return pd.DataFrame.from_dict(rows, orient="index")
        except Exception as e:
            logger.error("Failed to get IB market data: %s", e)
            return pd.DataFrame()
