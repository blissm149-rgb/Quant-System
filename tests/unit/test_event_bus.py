"""Unit tests for event bus.

TESTING_PLAN.md Section 3.12 — event_bus (CRITICAL).
"""

import pytest

from quant_fund.infrastructure.event_bus import (
    Event,
    EventBus,
    EventPriority,
    EventType,
)


@pytest.mark.unit
@pytest.mark.tier1
class TestEventBus:
    """EventBus — CRITICAL: pub/sub, idempotency, priority."""

    @pytest.fixture
    def bus(self):
        return EventBus(synchronous=True)

    def test_publish_and_subscribe(self, bus):
        """Subscriber receives published event."""
        received = []
        bus.subscribe("sub1", lambda e: received.append(e), {EventType.MARKET_DATA})
        event = Event(event_type=EventType.MARKET_DATA)
        bus.publish(event)
        assert len(received) == 1
        assert received[0].event_type == EventType.MARKET_DATA

    def test_unsubscribe(self, bus):
        """Unsubscribed handler no longer receives events."""
        received = []
        bus.subscribe("sub1", lambda e: received.append(e), {EventType.MARKET_DATA})
        bus.unsubscribe("sub1")
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        assert len(received) == 0

    def test_event_type_filtering(self, bus):
        """Subscriber only receives subscribed event types."""
        received = []
        bus.subscribe("sub1", lambda e: received.append(e), {EventType.ORDER_FILL})
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        assert len(received) == 0

    def test_idempotency(self, bus):
        """Duplicate idempotency_key events are deduplicated."""
        received = []
        bus.subscribe("sub1", lambda e: received.append(e), {EventType.MARKET_DATA})
        event = Event(event_type=EventType.MARKET_DATA, idempotency_key="dedup_1")
        bus.publish(event)
        bus.publish(event)
        assert len(received) == 1

    def test_priority_ordering(self, bus):
        """Higher priority events are dispatched before lower."""
        order = []
        bus.subscribe(
            "critical", lambda e: order.append("critical"),
            {EventType.RISK_CHECK}, priority=EventPriority.CRITICAL,
        )
        bus.subscribe(
            "low", lambda e: order.append("low"),
            {EventType.RISK_CHECK}, priority=EventPriority.LOW,
        )
        bus.publish(Event(event_type=EventType.RISK_CHECK))
        assert order[0] == "critical"

    def test_multiple_subscribers(self, bus):
        """Multiple subscribers all receive the event."""
        counts = {"a": 0, "b": 0}
        bus.subscribe("a", lambda e: counts.__setitem__("a", counts["a"] + 1), {EventType.MARKET_DATA})
        bus.subscribe("b", lambda e: counts.__setitem__("b", counts["b"] + 1), {EventType.MARKET_DATA})
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_get_history(self, bus):
        """get_history returns published events."""
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        bus.publish(Event(event_type=EventType.ORDER_FILL))
        history = bus.get_history(limit=10)
        assert len(history) >= 2

    def test_replay(self, bus):
        """replay re-dispatches events to subscribers."""
        received = []
        bus.subscribe("sub1", lambda e: received.append(e), {EventType.MARKET_DATA})
        events = [Event(event_type=EventType.MARKET_DATA) for _ in range(3)]
        count = bus.replay(events)
        assert count == 3
        assert len(received) == 3

    def test_get_metrics(self, bus):
        """get_metrics returns dict with event bus statistics."""
        bus.publish(Event(event_type=EventType.MARKET_DATA))
        metrics = bus.get_metrics()
        assert isinstance(metrics, dict)

    def test_event_types_enum(self):
        """All expected event types exist."""
        assert EventType.MARKET_DATA
        assert EventType.SIGNAL_GENERATED
        assert EventType.RISK_CHECK
        assert EventType.ORDER_SUBMISSION
        assert EventType.ORDER_FILL
        assert EventType.SHUTDOWN
