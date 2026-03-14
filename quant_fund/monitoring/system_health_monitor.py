"""System health monitor.

Monitors data feed health, process uptime, memory usage,
and latency metrics. Raises alerts on anomalies.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Result of a single health check."""

    component: str
    status: str  # "healthy", "degraded", "down"
    latency_ms: float = 0.0
    message: str = ""
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class SystemHealthSnapshot:
    """Overall system health at a point in time."""

    timestamp: pd.Timestamp
    overall_status: str
    checks: List[HealthCheck] = field(default_factory=list)
    data_feed_lag_s: float = 0.0
    uptime_s: float = 0.0


class SystemHealthMonitor:
    """Monitors overall system health.

    Tracks:
    - Data feed freshness and lag
    - Component health (broker connection, data feeds)
    - Process uptime
    - Configurable health checks
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._max_data_lag_s = cfg.get("max_data_lag_s", 300)
        self._start_time = time.monotonic()
        self._last_data_timestamps: Dict[str, pd.Timestamp] = {}
        self._health_history: List[SystemHealthSnapshot] = []

    def record_data_timestamp(
        self, feed_name: str, timestamp: pd.Timestamp
    ) -> None:
        """Record the latest timestamp from a data feed."""
        self._last_data_timestamps[feed_name] = timestamp

    def check_data_feed_health(self) -> List[HealthCheck]:
        """Check if data feeds are fresh."""
        checks = []
        now = pd.Timestamp.now()
        for feed, last_ts in self._last_data_timestamps.items():
            lag = (now - last_ts).total_seconds()
            status = "healthy" if lag < self._max_data_lag_s else "degraded"
            if lag > self._max_data_lag_s * 3:
                status = "down"
            checks.append(HealthCheck(
                component=f"data_feed:{feed}",
                status=status,
                latency_ms=lag * 1000,
                message=f"Last update {lag:.0f}s ago",
                timestamp=now,
            ))
        return checks

    def run_health_check(
        self,
        custom_checks: Optional[List[HealthCheck]] = None,
    ) -> SystemHealthSnapshot:
        """Run a full system health check.

        Parameters
        ----------
        custom_checks : list of HealthCheck, optional
            Additional component checks to include.

        Returns
        -------
        SystemHealthSnapshot
        """
        now = pd.Timestamp.now()
        all_checks = self.check_data_feed_health()
        if custom_checks:
            all_checks.extend(custom_checks)

        # Determine overall status
        statuses = [c.status for c in all_checks]
        if "down" in statuses:
            overall = "down"
        elif "degraded" in statuses:
            overall = "degraded"
        elif statuses:
            overall = "healthy"
        else:
            overall = "unknown"

        # Data feed lag
        max_lag = 0.0
        if self._last_data_timestamps:
            lags = [(now - ts).total_seconds()
                    for ts in self._last_data_timestamps.values()]
            max_lag = max(lags)

        uptime = time.monotonic() - self._start_time

        snapshot = SystemHealthSnapshot(
            timestamp=now,
            overall_status=overall,
            checks=all_checks,
            data_feed_lag_s=max_lag,
            uptime_s=uptime,
        )
        self._health_history.append(snapshot)
        return snapshot

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def latest(self) -> Optional[SystemHealthSnapshot]:
        return self._health_history[-1] if self._health_history else None

    def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        if not self._health_history:
            return True  # No checks run yet
        return self._health_history[-1].overall_status == "healthy"
