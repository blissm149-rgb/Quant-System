"""Test system health monitoring and data feed drift detection.

Validates SystemHealthMonitor data feed freshness checks,
overall status derivation, and degradation detection under
various staleness scenarios.
"""

import pytest
import pandas as pd

from quant_fund.monitoring.system_health_monitor import (
    HealthCheck,
    SystemHealthMonitor,
    SystemHealthSnapshot,
)


pytestmark = [pytest.mark.validation]


class TestDataFeedFreshness:
    """Validate data feed health detection based on timestamp lag."""

    def test_fresh_feed_marked_healthy(self):
        """Feed with recent timestamp should be marked healthy."""
        monitor = SystemHealthMonitor(config={"max_data_lag_s": 300})
        monitor.record_data_timestamp("prices", pd.Timestamp.now())

        snap = monitor.run_health_check()
        assert snap.overall_status == "healthy"
        assert snap.data_feed_lag_s < 5  # should be near-zero

    def test_stale_feed_marked_degraded(self):
        """Feed stale beyond max_data_lag_s but under 3x should be degraded."""
        monitor = SystemHealthMonitor(config={"max_data_lag_s": 60})
        stale = pd.Timestamp.now() - pd.Timedelta(seconds=120)
        monitor.record_data_timestamp("prices", stale)

        snap = monitor.run_health_check()
        assert snap.overall_status == "degraded"

    def test_very_stale_feed_marked_down(self):
        """Feed stale beyond 3x max_data_lag_s should be marked down."""
        monitor = SystemHealthMonitor(config={"max_data_lag_s": 60})
        very_stale = pd.Timestamp.now() - pd.Timedelta(seconds=300)
        monitor.record_data_timestamp("prices", very_stale)

        snap = monitor.run_health_check()
        assert snap.overall_status == "down"

    def test_mixed_feeds_worst_status_wins(self):
        """Overall status should reflect the worst individual feed."""
        monitor = SystemHealthMonitor(config={"max_data_lag_s": 60})
        monitor.record_data_timestamp("prices", pd.Timestamp.now())  # healthy
        very_stale = pd.Timestamp.now() - pd.Timedelta(seconds=300)
        monitor.record_data_timestamp("news", very_stale)  # down

        snap = monitor.run_health_check()
        assert snap.overall_status == "down"

    def test_no_feeds_returns_unknown(self):
        """No registered feeds should produce 'unknown' status."""
        monitor = SystemHealthMonitor()
        snap = monitor.run_health_check()
        assert snap.overall_status == "unknown"


class TestCustomHealthChecks:
    """Validate custom component health checks integration."""

    def test_custom_check_affects_overall_status(self):
        """A degraded custom check should degrade overall status."""
        monitor = SystemHealthMonitor()
        custom = [HealthCheck(
            component="broker_connection",
            status="degraded",
            latency_ms=5000,
            message="High latency",
        )]

        snap = monitor.run_health_check(custom_checks=custom)
        assert snap.overall_status == "degraded"

    def test_all_healthy_custom_checks(self):
        """All healthy custom checks with healthy feeds should be healthy."""
        monitor = SystemHealthMonitor()
        monitor.record_data_timestamp("prices", pd.Timestamp.now())
        custom = [
            HealthCheck(component="broker", status="healthy"),
            HealthCheck(component="database", status="healthy"),
        ]

        snap = monitor.run_health_check(custom_checks=custom)
        assert snap.overall_status == "healthy"

    def test_down_custom_check_overrides_healthy_feeds(self):
        """A down custom check should override healthy feed status."""
        monitor = SystemHealthMonitor()
        monitor.record_data_timestamp("prices", pd.Timestamp.now())
        custom = [HealthCheck(component="broker", status="down")]

        snap = monitor.run_health_check(custom_checks=custom)
        assert snap.overall_status == "down"


class TestSystemHealthHistory:
    """Validate health check history and uptime tracking."""

    def test_health_history_accumulates(self):
        """Each run_health_check should append to history."""
        monitor = SystemHealthMonitor()
        monitor.record_data_timestamp("prices", pd.Timestamp.now())

        for _ in range(5):
            monitor.run_health_check()

        assert monitor.latest is not None
        assert isinstance(monitor.latest, SystemHealthSnapshot)

    def test_uptime_positive(self):
        """Uptime should be a positive number after construction."""
        monitor = SystemHealthMonitor()
        assert monitor.uptime_seconds > 0

    def test_is_healthy_reflects_latest_check(self):
        """is_healthy() should reflect the most recent health check."""
        monitor = SystemHealthMonitor(config={"max_data_lag_s": 60})
        monitor.record_data_timestamp("prices", pd.Timestamp.now())
        monitor.run_health_check()
        assert monitor.is_healthy() is True

        # Make feed stale
        very_stale = pd.Timestamp.now() - pd.Timedelta(seconds=300)
        monitor.record_data_timestamp("prices", very_stale)
        monitor.run_health_check()
        assert monitor.is_healthy() is False

    def test_max_data_feed_lag_reported(self):
        """data_feed_lag_s should report the maximum lag across all feeds."""
        monitor = SystemHealthMonitor()
        monitor.record_data_timestamp(
            "prices", pd.Timestamp.now() - pd.Timedelta(seconds=10)
        )
        monitor.record_data_timestamp(
            "news", pd.Timestamp.now() - pd.Timedelta(seconds=100)
        )

        snap = monitor.run_health_check()
        assert snap.data_feed_lag_s >= 99  # at least ~100s for news feed
