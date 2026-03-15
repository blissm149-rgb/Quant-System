"""Event bus — in-process asynchronous message bus for the trading engine.

All system components communicate through this event bus rather than
direct method calls. Events are processed asynchronously and support
idempotency via unique event IDs.

Event types:
    market_data_event      — new bar or tick received
    signal_generated_event — alpha scores computed
    risk_check_event       — risk validation requested/completed
    order_submission_event — order submitted to broker
    order_fill_event       — fill received from broker
    reconciliation_event   — reconciliation cycle completed
    system_health_event    — health check result

The bus supports:
    - Typed publish/subscribe
    - Synchronous and async dispatch
    - Event persistence for replay
    - Idempotency via event_id deduplication
    - Priority ordering within event types
"""

import logging
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """All event types in the trading system."""

    MARKET_DATA = "market_data_event"
    SIGNAL_GENERATED = "signal_generated_event"
    RISK_CHECK = "risk_check_event"
    ORDER_SUBMISSION = "order_submission_event"
    ORDER_FILL = "order_fill_event"
    RECONCILIATION = "reconciliation_event"
    SYSTEM_HEALTH = "system_health_event"
    STATE_CHANGE = "state_change_event"
    SHUTDOWN = "shutdown_event"


class EventPriority(int, Enum):
    """Event processing priority (lower = higher priority)."""

    CRITICAL = 0  # kill switch, halt
    HIGH = 10  # risk checks, fills
    NORMAL = 20  # signals, orders
    LOW = 30  # monitoring, health
    BACKGROUND = 40  # logging, metrics


@dataclass
class Event:
    """A single event in the system.

    Each event has a unique ID for idempotency. Events carry
    a typed payload (dict) and metadata for routing.
    """

    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = ""
    priority: EventPriority = EventPriority.NORMAL
    idempotency_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
            "priority": self.priority.value,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", ""),
            source=data.get("source", ""),
            priority=EventPriority(data.get("priority", EventPriority.NORMAL)),
            idempotency_key=data.get("idempotency_key", ""),
        )


# Type alias for event handler callbacks
EventHandler = Callable[[Event], None]


@dataclass
class Subscription:
    """A registered event subscription."""

    handler: EventHandler
    event_types: Set[EventType]
    subscriber_id: str
    priority: EventPriority = EventPriority.NORMAL


class EventBus:
    """In-process event bus with publish/subscribe semantics.

    Thread-safe. Supports both synchronous dispatch (for testing
    and deterministic replay) and background dispatch via a
    worker thread.

    Usage:
        bus = EventBus()

        def on_market_data(event: Event):
            print(f"Got data: {event.payload}")

        bus.subscribe("my_strategy", on_market_data, {EventType.MARKET_DATA})
        bus.publish(Event(event_type=EventType.MARKET_DATA, payload={"ticker": "AAPL"}))
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        synchronous: bool = False,
    ) -> None:
        cfg = config or {}
        self._synchronous = synchronous
        self._max_queue_size = cfg.get("max_queue_size", 10_000)
        self._dedup_window_size = cfg.get("dedup_window_size", 5_000)

        self._subscriptions: Dict[str, Subscription] = {}
        self._handlers_by_type: Dict[EventType, List[Subscription]] = defaultdict(list)
        self._lock = threading.Lock()

        # Event queue for async dispatch
        self._queue: Deque[Event] = deque(maxlen=self._max_queue_size)

        # Idempotency: track recently processed event IDs
        self._processed_ids: Deque[str] = deque(maxlen=self._dedup_window_size)
        self._processed_set: Set[str] = set()

        # Event history for replay/debugging
        self._history: Deque[Event] = deque(
            maxlen=cfg.get("history_size", 1_000)
        )

        # Persistence callback (optional, for StateStore integration)
        self._persist_callback: Optional[Callable[[Event], None]] = None

        # Background worker
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._work_available = threading.Event()

        # Metrics
        self._publish_count = 0
        self._dispatch_count = 0
        self._dedup_count = 0
        self._error_count = 0

        logger.info(
            "EventBus initialized synchronous=%s max_queue=%d",
            synchronous,
            self._max_queue_size,
        )

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(
        self,
        subscriber_id: str,
        handler: EventHandler,
        event_types: Set[EventType],
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """Register a handler for one or more event types.

        Args:
            subscriber_id: Unique identifier for the subscriber.
            handler: Callable that receives an Event.
            event_types: Set of EventType values to subscribe to.
            priority: Handler execution priority.
        """
        with self._lock:
            sub = Subscription(
                handler=handler,
                event_types=event_types,
                subscriber_id=subscriber_id,
                priority=priority,
            )
            self._subscriptions[subscriber_id] = sub
            for et in event_types:
                self._handlers_by_type[et].append(sub)
                # Sort by priority so higher-priority handlers run first
                self._handlers_by_type[et].sort(key=lambda s: s.priority)

        logger.debug(
            "Subscribed %s to %s",
            subscriber_id,
            [et.value for et in event_types],
        )

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove a subscriber. Returns True if found."""
        with self._lock:
            sub = self._subscriptions.pop(subscriber_id, None)
            if sub is None:
                return False
            for et in sub.event_types:
                self._handlers_by_type[et] = [
                    s for s in self._handlers_by_type[et]
                    if s.subscriber_id != subscriber_id
                ]
        logger.debug("Unsubscribed %s", subscriber_id)
        return True

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> bool:
        """Publish an event to the bus.

        In synchronous mode, dispatches immediately.
        In async mode, enqueues for the worker thread.

        Returns False if the event was deduplicated (already processed).
        """
        # Idempotency check
        dedup_key = event.idempotency_key or event.event_id
        with self._lock:
            if dedup_key in self._processed_set:
                self._dedup_count += 1
                logger.debug("Deduplicated event %s", dedup_key)
                return False

            self._publish_count += 1
            self._history.append(event)

        # Persist if callback is set
        if self._persist_callback is not None:
            try:
                self._persist_callback(event)
            except Exception:
                logger.exception("Event persistence failed for %s", event.event_id)

        if self._synchronous:
            self._dispatch(event)
        else:
            with self._lock:
                self._queue.append(event)
            self._work_available.set()

        return True

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, event: Event) -> None:
        """Dispatch an event to all matching subscribers."""
        with self._lock:
            handlers = list(self._handlers_by_type.get(event.event_type, []))

        for sub in handlers:
            try:
                sub.handler(event)
                self._dispatch_count += 1
            except Exception:
                self._error_count += 1
                logger.exception(
                    "Handler %s failed for event %s",
                    sub.subscriber_id,
                    event.event_id,
                )

        # Mark as processed for idempotency
        dedup_key = event.idempotency_key or event.event_id
        with self._lock:
            self._processed_set.add(dedup_key)
            self._processed_ids.append(dedup_key)
            # Evict oldest from the set when deque drops them
            while len(self._processed_set) > len(self._processed_ids):
                # This shouldn't happen with proper tracking, but guard anyway
                break
            if len(self._processed_ids) == self._processed_ids.maxlen:
                # The deque auto-evicts; sync the set by rebuilding
                self._processed_set = set(self._processed_ids)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background dispatch worker thread."""
        if self._synchronous:
            logger.info("EventBus is synchronous, no worker thread needed")
            return
        with self._lock:
            if self._running:
                return
            self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="EventBus-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("EventBus worker thread started")

    def stop(self) -> None:
        """Stop the background dispatch worker."""
        with self._lock:
            self._running = False
        self._work_available.set()  # Wake the worker so it can exit
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        logger.info("EventBus worker thread stopped")

    def _worker_loop(self) -> None:
        """Background loop that processes queued events."""
        while True:
            self._work_available.wait(timeout=0.1)
            self._work_available.clear()

            with self._lock:
                if not self._running and not self._queue:
                    break

            while True:
                with self._lock:
                    if not self._queue:
                        break
                    event = self._queue.popleft()
                self._dispatch(event)

    # ------------------------------------------------------------------
    # Persistence integration
    # ------------------------------------------------------------------

    def set_persist_callback(
        self, callback: Callable[[Event], None]
    ) -> None:
        """Set a callback for persisting events (e.g. to StateStore)."""
        self._persist_callback = callback

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(self, events: List[Event]) -> int:
        """Replay a list of events through the bus.

        Respects idempotency — already-processed events are skipped.
        Returns count of events actually dispatched.
        """
        dispatched = 0
        for event in events:
            if self.publish(event):
                dispatched += 1
        return dispatched

    # ------------------------------------------------------------------
    # Query / metrics
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 100) -> List[Event]:
        """Return recent event history."""
        with self._lock:
            items = list(self._history)
        return items[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """Return bus metrics."""
        with self._lock:
            return {
                "publish_count": self._publish_count,
                "dispatch_count": self._dispatch_count,
                "dedup_count": self._dedup_count,
                "error_count": self._error_count,
                "queue_depth": len(self._queue),
                "subscriber_count": len(self._subscriptions),
                "history_size": len(self._history),
            }

    def drain(self) -> int:
        """Process all queued events synchronously. Returns count processed."""
        count = 0
        while True:
            with self._lock:
                if not self._queue:
                    break
                event = self._queue.popleft()
            self._dispatch(event)
            count += 1
        return count

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "EventBus":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
