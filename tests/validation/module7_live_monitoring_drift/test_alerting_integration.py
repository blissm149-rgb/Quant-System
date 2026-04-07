"""Test alerting system integration and dispatch correctness.

Validates AlertingSystem handler dispatch, level escalation,
duplicate suppression, acknowledgement, and history filtering
under realistic multi-component alert streams.
"""

import pytest
import pandas as pd

from quant_fund.monitoring.alerting_system import (
    Alert,
    AlertingSystem,
    AlertLevel,
)


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestAlertDispatch:
    """Validate alert dispatch routing and level escalation."""

    def test_info_handler_receives_all_levels(self):
        """INFO-level handlers should receive INFO, WARNING, and CRITICAL alerts."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 0})
        received = []
        alerting.register_handler(AlertLevel.INFO, lambda a: received.append(a))

        alerting.send(AlertLevel.INFO, "test", "info msg")
        alerting.send(AlertLevel.WARNING, "test", "warn msg")
        alerting.send(AlertLevel.CRITICAL, "test", "crit msg")

        assert len(received) == 3

    def test_warning_handler_skips_info(self):
        """WARNING-level handlers should not receive INFO alerts."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 0})
        received = []
        alerting.register_handler(AlertLevel.WARNING, lambda a: received.append(a))

        alerting.send(AlertLevel.INFO, "test", "info msg")
        alerting.send(AlertLevel.WARNING, "test", "warn msg")
        alerting.send(AlertLevel.CRITICAL, "test", "crit msg")

        assert len(received) == 2
        levels = {a.level for a in received}
        assert AlertLevel.INFO not in levels

    def test_critical_handler_only_critical(self):
        """CRITICAL-level handlers should only receive CRITICAL alerts."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 0})
        received = []
        alerting.register_handler(AlertLevel.CRITICAL, lambda a: received.append(a))

        alerting.send(AlertLevel.INFO, "test", "info msg")
        alerting.send(AlertLevel.WARNING, "test", "warn msg")
        alerting.send(AlertLevel.CRITICAL, "test", "crit msg")

        assert len(received) == 1
        assert received[0].level == AlertLevel.CRITICAL

    def test_multiple_handlers_same_level(self):
        """Multiple handlers at the same level should all be called."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 0})
        received_a = []
        received_b = []
        alerting.register_handler(AlertLevel.INFO, lambda a: received_a.append(a))
        alerting.register_handler(AlertLevel.INFO, lambda a: received_b.append(a))

        alerting.send(AlertLevel.INFO, "test", "hello")

        assert len(received_a) == 1
        assert len(received_b) == 1


class TestAlertDeduplication:
    """Validate duplicate suppression logic."""

    def test_duplicate_suppressed_within_window(self):
        """Identical alerts within suppression window should be deduplicated."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 600})

        a1 = alerting.send(AlertLevel.WARNING, "risk", "Limit breach")
        a2 = alerting.send(AlertLevel.WARNING, "risk", "Limit breach")

        # Second send returns the first alert (duplicate suppressed)
        assert a1.alert_id == a2.alert_id

    def test_different_messages_not_suppressed(self):
        """Different messages from the same source should not be suppressed."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 600})

        a1 = alerting.send(AlertLevel.WARNING, "risk", "Breach type A")
        a2 = alerting.send(AlertLevel.WARNING, "risk", "Breach type B")

        assert a1.alert_id != a2.alert_id

    def test_different_sources_not_suppressed(self):
        """Same message from different sources should not be suppressed."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 600})

        a1 = alerting.send(AlertLevel.WARNING, "risk_engine", "Limit breach")
        a2 = alerting.send(AlertLevel.WARNING, "kill_switch", "Limit breach")

        assert a1.alert_id != a2.alert_id


class TestAlertHistoryAndAcknowledge:
    """Validate history filtering and acknowledgement flows."""

    def test_acknowledge_removes_from_active(self):
        """Acknowledged alerts should not appear in active alerts."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 0})
        alert = alerting.send(AlertLevel.CRITICAL, "test", "fix me")

        assert len(alerting.get_active_alerts()) == 1
        alerting.acknowledge(alert.alert_id)
        assert len(alerting.get_active_alerts()) == 0

    def test_acknowledge_nonexistent_returns_false(self):
        """Acknowledging a nonexistent alert should return False."""
        alerting = AlertingSystem()
        assert alerting.acknowledge("ALERT-999999") is False

    def test_history_level_filter(self):
        """History filtered by level should only return matching alerts."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 0})
        alerting.send(AlertLevel.INFO, "a", "info 1")
        alerting.send(AlertLevel.INFO, "a", "info 2")
        alerting.send(AlertLevel.WARNING, "b", "warn 1")
        alerting.send(AlertLevel.CRITICAL, "c", "crit 1")

        warnings = alerting.get_history(level=AlertLevel.WARNING)
        assert len(warnings) == 1
        assert all(a.level == AlertLevel.WARNING for a in warnings)

    def test_critical_count_tracks_unacknowledged(self):
        """critical_count should only count unacknowledged CRITICAL alerts."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 0})

        a1 = alerting.send(AlertLevel.CRITICAL, "r", "problem 1")
        alerting.send(AlertLevel.CRITICAL, "r", "problem 2")
        alerting.send(AlertLevel.WARNING, "r", "not critical")

        assert alerting.critical_count == 2
        alerting.acknowledge(a1.alert_id)
        assert alerting.critical_count == 1

    def test_handler_exception_does_not_block_dispatch(self):
        """A failing handler should not prevent other handlers from executing."""
        alerting = AlertingSystem(config={"suppress_duplicates_seconds": 0})
        good_received = []

        def bad_handler(alert):
            raise RuntimeError("boom")

        alerting.register_handler(AlertLevel.INFO, bad_handler)
        alerting.register_handler(AlertLevel.INFO, lambda a: good_received.append(a))

        alerting.send(AlertLevel.INFO, "test", "should still dispatch")
        assert len(good_received) == 1
