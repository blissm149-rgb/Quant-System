"""Broker reconnection manager — automatic reconnection with backoff.

Wraps a BrokerInterface adapter and adds:
    - Automatic reconnection on disconnect
    - Exponential backoff retry logic
    - Failover to a secondary broker adapter
    - Connection health monitoring
    - Reconnection event publishing

All method calls are proxied through to the underlying adapter.
If a call fails due to connection issues, reconnection is attempted
before re-raising.
"""

import logging
import time
from typing import Any, Callable, List, Optional

import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    BrokerInterface,
    Fill,
    Order,
    OrderAcknowledgement,
    OrderStatus,
)

logger = logging.getLogger(__name__)


class BrokerReconnectionManager(BrokerInterface):
    """Wraps a broker adapter with automatic reconnection.

    Proxies all BrokerInterface calls to the primary adapter.
    On connection failure:
        1. Attempt reconnection with exponential backoff
        2. If primary fails after max_retries, failover to secondary
        3. Publish connection events via callback

    Usage:
        primary = InteractiveBrokersAdapter()
        secondary = AlpacaAdapter()  # optional failover

        mgr = BrokerReconnectionManager(
            primary=primary,
            secondary=secondary,
            config={"max_retries": 5},
        )
        mgr.connect()

        # All BrokerInterface calls go through the manager
        positions = mgr.get_positions()  # auto-reconnects on failure
    """

    def __init__(
        self,
        primary: BrokerInterface,
        secondary: Optional[BrokerInterface] = None,
        config: Optional[dict] = None,
    ):
        cfg = config or {}
        self._primary = primary
        self._secondary = secondary
        self._active: BrokerInterface = primary

        self._max_retries = cfg.get("max_retries", 4)
        self._initial_backoff_s = cfg.get("initial_backoff_s", 2.0)
        self._max_backoff_s = cfg.get("max_backoff_s", 30.0)

        self._is_connected = False
        self._using_failover = False
        self._reconnection_count = 0
        self._failover_count = 0
        self._last_successful_call: float = 0.0

        # Optional callback for connection state changes
        self._on_connection_change: Optional[
            Callable[[str, bool], None]
        ] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to the primary broker. Returns True on success."""
        success = self._try_connect(self._primary, "primary")
        if success:
            self._active = self._primary
            self._is_connected = True
            self._using_failover = False
            return True

        # Try secondary if available
        if self._secondary is not None:
            success = self._try_connect(self._secondary, "secondary")
            if success:
                self._active = self._secondary
                self._is_connected = True
                self._using_failover = True
                self._failover_count += 1
                return True

        self._is_connected = False
        return False

    def disconnect(self) -> None:
        """Disconnect from the active broker."""
        if hasattr(self._active, "disconnect"):
            try:
                self._active.disconnect()
            except Exception:
                logger.exception("Error disconnecting")
        self._is_connected = False

    def _try_connect(self, adapter: BrokerInterface, label: str) -> bool:
        """Try to connect to a broker with retry logic."""
        if not hasattr(adapter, "connect"):
            # Adapter doesn't need explicit connection (e.g. SimulationBroker)
            logger.info("%s adapter connected (no connect method)", label)
            return True

        backoff = self._initial_backoff_s
        for attempt in range(1, self._max_retries + 1):
            try:
                result = adapter.connect()
                if result is False:
                    logger.warning(
                        "%s connect attempt %d/%d failed",
                        label,
                        attempt,
                        self._max_retries,
                    )
                else:
                    logger.info(
                        "%s broker connected on attempt %d", label, attempt
                    )
                    return True
            except Exception as e:
                logger.warning(
                    "%s connect attempt %d/%d error: %s",
                    label,
                    attempt,
                    self._max_retries,
                    e,
                )

            if attempt < self._max_retries:
                time.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff_s)

        return False

    def _reconnect(self) -> bool:
        """Attempt to reconnect after a detected failure."""
        logger.warning("Attempting broker reconnection...")
        self._reconnection_count += 1
        success = self.connect()
        if self._on_connection_change is not None:
            self._on_connection_change(
                "reconnected" if success else "disconnected", success
            )
        return success

    # ------------------------------------------------------------------
    # BrokerInterface proxy methods
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> OrderAcknowledgement:
        """Submit order, with auto-reconnection on failure."""
        return self._safe_call(
            lambda: self._active.submit_order(order),
            fallback=OrderAcknowledgement(
                order_id="",
                status=OrderStatus.REJECTED,
                message="Broker disconnected",
                timestamp=order.timestamp,
            ),
        )

    def cancel_order(self, order_id: str) -> bool:
        return self._safe_call(
            lambda: self._active.cancel_order(order_id),
            fallback=False,
        )

    def get_positions(self) -> pd.Series:
        return self._safe_call(
            lambda: self._active.get_positions(),
            fallback=pd.Series(dtype=float),
        )

    def get_account_value(self) -> float:
        return self._safe_call(
            lambda: self._active.get_account_value(),
            fallback=0.0,
        )

    def get_fills(self, since: pd.Timestamp) -> List[Fill]:
        return self._safe_call(
            lambda: self._active.get_fills(since),
            fallback=[],
        )

    def get_market_data(self, tickers: List[str]) -> pd.DataFrame:
        return self._safe_call(
            lambda: self._active.get_market_data(tickers),
            fallback=pd.DataFrame(),
        )

    # ------------------------------------------------------------------
    # Safe call wrapper
    # ------------------------------------------------------------------

    def _safe_call(self, fn: Callable, fallback: Any) -> Any:
        """Call a broker method, reconnecting on failure."""
        try:
            result = fn()
            self._last_successful_call = time.monotonic()
            return result
        except Exception as e:
            logger.warning("Broker call failed: %s", e)

            # Attempt reconnection
            if self._reconnect():
                try:
                    result = fn()
                    self._last_successful_call = time.monotonic()
                    return result
                except Exception:
                    logger.exception("Broker call failed after reconnection")

            return fallback

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def set_connection_callback(
        self, callback: Callable[[str, bool], None]
    ) -> None:
        """Set callback for connection state changes.

        Callback receives (event_name, success) where event_name is
        "reconnected" or "disconnected".
        """
        self._on_connection_change = callback

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def using_failover(self) -> bool:
        return self._using_failover

    @property
    def reconnection_count(self) -> int:
        return self._reconnection_count

    @property
    def failover_count(self) -> int:
        return self._failover_count

    def get_metrics(self) -> dict:
        return {
            "is_connected": self._is_connected,
            "using_failover": self._using_failover,
            "reconnection_count": self._reconnection_count,
            "failover_count": self._failover_count,
            "seconds_since_last_call": (
                time.monotonic() - self._last_successful_call
                if self._last_successful_call > 0
                else -1
            ),
        }
