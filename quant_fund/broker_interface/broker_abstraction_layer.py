"""Broker abstraction layer — interface all adapters implement.

All broker-specific code is hidden behind this interface. The rest of
the system only calls BrokerInterface. Never calls adapters directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional

import pandas as pd


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """An order to be submitted to a broker."""

    ticker: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    algo_params: dict = field(default_factory=dict)
    strategy_id: str = ""
    timestamp: Optional[pd.Timestamp] = None
    order_id: str = ""


@dataclass
class OrderAcknowledgement:
    """Acknowledgement returned after order submission."""

    order_id: str
    status: OrderStatus
    message: str = ""
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class Fill:
    """A fill (execution) record."""

    order_id: str
    ticker: str
    side: OrderSide
    quantity: int
    fill_price: float
    timestamp: pd.Timestamp
    commission: float = 0.0


class BrokerInterface(ABC):
    """Abstract interface that all broker adapters must implement.

    The rest of the system only interacts with this interface.
    """

    @abstractmethod
    def submit_order(self, order: Order) -> OrderAcknowledgement:
        """Submit an order to the broker."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successfully cancelled."""
        ...

    @abstractmethod
    def get_positions(self) -> pd.Series:
        """Get current positions indexed by ticker (in shares)."""
        ...

    @abstractmethod
    def get_account_value(self) -> float:
        """Get current total account value (NAV)."""
        ...

    @abstractmethod
    def get_fills(self, since: pd.Timestamp) -> List[Fill]:
        """Get all fills since the given timestamp."""
        ...

    @abstractmethod
    def get_market_data(self, tickers: List[str]) -> pd.DataFrame:
        """Get current market data for the given tickers.

        Returns DataFrame indexed by ticker with columns:
        bid, ask, mid, last, volume.
        """
        ...
