"""Live data stream adapter for real-time market data feeds.

Wraps a real-time data feed (WebSocket or REST polling). Produces the same
DataFrame schema as ``historical_data_loader`` so the rest of the pipeline
is feed-agnostic. Emits an event on each new bar.

The adapter uses a pluggable feed interface: callers supply a concrete
``FeedProvider`` that knows how to fetch the latest bars from a vendor
(Polygon, Alpaca, etc.).  When no provider is supplied the adapter falls
back to a no-op stub so that it can be exercised in unit tests and with the
simulation broker.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Set

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_FIELDS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Pluggable feed interface
# ---------------------------------------------------------------------------

class FeedProvider(ABC):
    """Abstract interface that vendor-specific data feeds must implement."""

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the upstream data source."""

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the upstream connection."""

    @abstractmethod
    def fetch_latest_bars(self, tickers: List[str]) -> pd.DataFrame:
        """Return the most recent bar for each ticker.

        Returns:
            DataFrame with MultiIndex (date, ticker) and columns
            ``open, high, low, close, volume`` at minimum.
        """

    @abstractmethod
    def fetch_snapshot(
        self, tickers: List[str], fields: List[str]
    ) -> pd.DataFrame:
        """Return a point-in-time snapshot for the given tickers and fields.

        Returns:
            DataFrame with MultiIndex (date, ticker) and the requested
            field columns.
        """


class NoOpFeedProvider(FeedProvider):
    """Stub provider used when no real vendor is configured.

    Returns empty DataFrames so that the adapter can be instantiated in
    tests or dry-run mode without a live connection.
    """

    def connect(self) -> None:
        logger.info("NoOpFeedProvider connected (no-op)")

    def disconnect(self) -> None:
        logger.info("NoOpFeedProvider disconnected (no-op)")

    def fetch_latest_bars(self, tickers: List[str]) -> pd.DataFrame:
        index = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return pd.DataFrame(index=index, columns=DEFAULT_FIELDS, dtype=float)

    def fetch_snapshot(
        self, tickers: List[str], fields: List[str]
    ) -> pd.DataFrame:
        index = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return pd.DataFrame(index=index, columns=fields, dtype=float)


# ---------------------------------------------------------------------------
# Live data stream adapter
# ---------------------------------------------------------------------------

class LiveDataStreamAdapter:
    """Streams real-time market data and exposes it in the same MultiIndex
    ``(date, ticker)`` schema used by :class:`HistoricalDataLoader`.

    Features
    --------
    * Pluggable ``FeedProvider`` for vendor abstraction.
    * Background polling thread with configurable interval.
    * Thread-safe internal ring buffer of recent bars.
    * Callback registration for event-driven bar processing.
    * ``subscribe`` / ``unsubscribe`` to dynamically manage the ticker set.
    * Context-manager support (``with`` statement).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        feed_provider: Optional[FeedProvider] = None,
    ) -> None:
        self._config = config or {}

        live_cfg = self._config.get("live_feed", {})
        self._polling_interval: float = live_cfg.get("polling_interval_sec", 5.0)
        self._buffer_size: int = live_cfg.get("buffer_size", 500)
        self._data_source: str = self._config.get("data_sources", {}).get(
            "price_data", "polygon_io"
        )

        self._feed: FeedProvider = feed_provider or NoOpFeedProvider()

        # Subscriptions & callbacks
        self._subscribed_tickers: Set[str] = set()
        self._bar_callbacks: List[Callable[[pd.DataFrame], None]] = []

        # Ring buffer: stores DataFrames (one per poll cycle)
        self._buffer: Deque[pd.DataFrame] = deque(maxlen=self._buffer_size)

        # Threading primitives
        self._lock = threading.Lock()
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None

        logger.info(
            "LiveDataStreamAdapter initialised  source=%s  poll=%.1fs  buffer=%d",
            self._data_source,
            self._polling_interval,
            self._buffer_size,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config_file(
        cls,
        config_path: str,
        feed_provider: Optional[FeedProvider] = None,
    ) -> "LiveDataStreamAdapter":
        """Construct an adapter from a YAML configuration file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config, feed_provider=feed_provider)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Start the data feed and begin the background polling loop."""
        with self._lock:
            if self._running:
                logger.warning("connect() called but adapter is already running")
                return
            self._feed.connect()
            self._running = True
            self._poll_thread = threading.Thread(
                target=self._polling_loop,
                name="LiveDataStreamAdapter-poll",
                daemon=True,
            )
            self._poll_thread.start()
            logger.info("Live data stream connected and polling started")

    def disconnect(self) -> None:
        """Stop the polling loop and disconnect from the upstream feed."""
        with self._lock:
            if not self._running:
                logger.warning("disconnect() called but adapter is not running")
                return
            self._running = False

        # Wait for the polling thread to finish (outside lock to avoid
        # deadlock if the thread is blocked on the lock).
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=self._polling_interval * 3)
            self._poll_thread = None

        self._feed.disconnect()
        logger.info("Live data stream disconnected")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LiveDataStreamAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.disconnect()

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, tickers: List[str]) -> None:
        """Add tickers to the live subscription set."""
        with self._lock:
            before = len(self._subscribed_tickers)
            self._subscribed_tickers.update(tickers)
            added = len(self._subscribed_tickers) - before
        logger.info(
            "Subscribed to %d new ticker(s) — total subscriptions: %d",
            added,
            len(self._subscribed_tickers),
        )

    def unsubscribe(self, tickers: List[str]) -> None:
        """Remove tickers from the live subscription set."""
        with self._lock:
            before = len(self._subscribed_tickers)
            self._subscribed_tickers.difference_update(tickers)
            removed = before - len(self._subscribed_tickers)
        logger.info(
            "Unsubscribed from %d ticker(s) — total subscriptions: %d",
            removed,
            len(self._subscribed_tickers),
        )

    @property
    def subscribed_tickers(self) -> List[str]:
        """Return a sorted copy of the current subscription set."""
        with self._lock:
            return sorted(self._subscribed_tickers)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_bar(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Register a callback that is invoked on every new bar.

        The callback receives a DataFrame with the same MultiIndex
        ``(date, ticker)`` schema used throughout the data layer.
        """
        with self._lock:
            self._bar_callbacks.append(callback)
        logger.debug("Registered bar callback: %s", callback)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_latest_bar(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Return the most recent bar for each of the requested tickers.

        If *tickers* is ``None``, bars for all subscribed tickers are returned.

        Returns:
            DataFrame with MultiIndex ``(date, ticker)`` and columns
            ``open, high, low, close, volume``.
        """
        tickers = tickers or self.subscribed_tickers
        if not tickers:
            logger.warning("get_latest_bar called with no tickers")
            return self._empty_frame(DEFAULT_FIELDS)

        with self._lock:
            if not self._buffer:
                logger.warning("Buffer is empty — no bars available yet")
                return self._empty_frame(DEFAULT_FIELDS)

            # Walk the buffer from newest to oldest, collecting the most
            # recent row for each requested ticker.
            needed: Set[str] = set(tickers)
            frames: List[pd.DataFrame] = []
            for bars_df in reversed(self._buffer):
                if bars_df.empty:
                    continue
                bar_tickers = set(
                    bars_df.index.get_level_values("ticker")
                )
                overlap = needed & bar_tickers
                if overlap:
                    mask = bars_df.index.get_level_values("ticker").isin(overlap)
                    frames.append(bars_df.loc[mask])
                    needed -= overlap
                if not needed:
                    break

        if not frames:
            return self._empty_frame(DEFAULT_FIELDS)

        result = pd.concat(frames)
        # Keep only the requested fields that are present
        available = [f for f in DEFAULT_FIELDS if f in result.columns]
        return result[available] if available else result

    def get_snapshot(
        self,
        tickers: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return a point-in-time snapshot for the given tickers and fields.

        Delegates to the underlying :class:`FeedProvider` for a fresh fetch.

        Returns:
            DataFrame with MultiIndex ``(date, ticker)`` and the requested
            field columns.
        """
        tickers = tickers or self.subscribed_tickers
        fields = fields or DEFAULT_FIELDS
        if not tickers:
            logger.warning("get_snapshot called with no tickers")
            return self._empty_frame(fields)

        return self._feed.fetch_snapshot(tickers, fields)

    def get_buffered_bars(self, n: Optional[int] = None) -> pd.DataFrame:
        """Return the last *n* buffered bar DataFrames concatenated.

        If *n* is ``None`` the entire buffer is returned.

        Returns:
            DataFrame with MultiIndex ``(date, ticker)`` and OHLCV columns.
        """
        with self._lock:
            items = list(self._buffer)

        if not items:
            return self._empty_frame(DEFAULT_FIELDS)

        if n is not None:
            items = items[-n:]

        return pd.concat(items)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _polling_loop(self) -> None:
        """Background loop that polls the feed provider at fixed intervals."""
        logger.debug("Polling loop started (interval=%.1fs)", self._polling_interval)
        while True:
            with self._lock:
                if not self._running:
                    break
                tickers = sorted(self._subscribed_tickers)

            if not tickers:
                time.sleep(self._polling_interval)
                continue

            try:
                bars = self._feed.fetch_latest_bars(tickers)
            except Exception:
                logger.exception("Error fetching latest bars")
                time.sleep(self._polling_interval)
                continue

            if bars is not None and not bars.empty:
                with self._lock:
                    self._buffer.append(bars)
                    callbacks = list(self._bar_callbacks)

                # Fire callbacks outside the lock so slow consumers cannot
                # block the polling thread from updating the buffer.
                for cb in callbacks:
                    try:
                        cb(bars)
                    except Exception:
                        logger.exception("Error in bar callback %s", cb)

            time.sleep(self._polling_interval)

        logger.debug("Polling loop exited")

    @staticmethod
    def _empty_frame(columns: List[str]) -> pd.DataFrame:
        """Return an empty DataFrame with the canonical MultiIndex schema."""
        index = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return pd.DataFrame(index=index, columns=columns, dtype=float)
