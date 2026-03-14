"""Alpaca broker adapter wrapping the Alpaca REST/WebSocket API.

Implements BrokerInterface. Used for paper trading and smaller live accounts.
Requires the alpaca-trade-api package at runtime.
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


class AlpacaAdapter(BrokerInterface):
    """Adapter for the Alpaca REST/WebSocket API.

    Used for paper trading and smaller live accounts. Requires
    alpaca-trade-api package.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._api_key = cfg.get("alpaca_api_key", "")
        self._secret_key = cfg.get("alpaca_secret_key", "")
        self._base_url = cfg.get("alpaca_base_url", "https://paper-api.alpaca.markets")
        self._api = None

    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            import alpaca_trade_api as tradeapi
            self._api = tradeapi.REST(
                self._api_key,
                self._secret_key,
                self._base_url,
                api_version="v2",
            )
            account = self._api.get_account()
            logger.info("Connected to Alpaca. Account status: %s", account.status)
            return True
        except ImportError:
            logger.error("alpaca-trade-api not installed")
            return False
        except Exception as e:
            logger.error("Failed to connect to Alpaca: %s", e)
            return False

    def submit_order(self, order: Order) -> OrderAcknowledgement:
        """Submit order to Alpaca."""
        if self._api is None:
            return OrderAcknowledgement(
                order_id="", status=OrderStatus.REJECTED,
                message="Not connected",
            )
        try:
            alpaca_order = self._api.submit_order(
                symbol=order.ticker,
                qty=order.quantity,
                side=order.side.value,
                type="market" if order.order_type.value in ("market", "vwap", "twap") else "limit",
                time_in_force="day",
                limit_price=order.limit_price if order.order_type.value == "limit" else None,
            )
            return OrderAcknowledgement(
                order_id=alpaca_order.id,
                status=OrderStatus.SUBMITTED,
                message="Submitted to Alpaca",
                timestamp=pd.Timestamp.now(),
            )
        except Exception as e:
            logger.error("Alpaca order failed: %s", e)
            return OrderAcknowledgement(
                order_id="", status=OrderStatus.REJECTED, message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca."""
        if self._api is None:
            return False
        try:
            self._api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error("Alpaca cancel failed: %s", e)
            return False

    def get_positions(self) -> pd.Series:
        """Get current Alpaca positions."""
        if self._api is None:
            return pd.Series(dtype=float)
        try:
            positions = {}
            for pos in self._api.list_positions():
                positions[pos.symbol] = float(pos.qty)
            return pd.Series(positions, dtype=float)
        except Exception as e:
            logger.error("Failed to get Alpaca positions: %s", e)
            return pd.Series(dtype=float)

    def get_account_value(self) -> float:
        """Get Alpaca account equity."""
        if self._api is None:
            return 0.0
        try:
            account = self._api.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error("Failed to get Alpaca account value: %s", e)
            return 0.0

    def get_fills(self, since: pd.Timestamp) -> List[Fill]:
        """Get Alpaca fills since timestamp."""
        if self._api is None:
            return []
        try:
            fills = []
            orders = self._api.list_orders(
                status="filled",
                after=since.isoformat(),
            )
            for o in orders:
                fills.append(Fill(
                    order_id=o.id,
                    ticker=o.symbol,
                    side=OrderSide.BUY if o.side == "buy" else OrderSide.SELL,
                    quantity=int(float(o.filled_qty)),
                    fill_price=float(o.filled_avg_price),
                    timestamp=pd.Timestamp(o.filled_at),
                ))
            return fills
        except Exception as e:
            logger.error("Failed to get Alpaca fills: %s", e)
            return []

    def get_market_data(self, tickers: List[str]) -> pd.DataFrame:
        """Get current market data from Alpaca."""
        if self._api is None:
            return pd.DataFrame()
        try:
            rows = {}
            for ticker in tickers:
                quote = self._api.get_latest_quote(ticker)
                rows[ticker] = {
                    "bid": float(quote.bp) if hasattr(quote, "bp") else 0.0,
                    "ask": float(quote.ap) if hasattr(quote, "ap") else 0.0,
                    "mid": (float(quote.bp) + float(quote.ap)) / 2 if hasattr(quote, "bp") else 0.0,
                    "last": 0.0,
                    "volume": 0,
                }
            return pd.DataFrame.from_dict(rows, orient="index")
        except Exception as e:
            logger.error("Failed to get Alpaca market data: %s", e)
            return pd.DataFrame()
