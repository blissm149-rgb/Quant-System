"""Centralised alerting system.

Dispatches alerts at INFO, WARNING, and CRITICAL levels.
CRITICAL alerts: kill switch, exposure breach, data feed outage.
Supports email, Slack webhook, and PagerDuty delivery (configurable).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A single alert."""

    alert_id: str
    level: AlertLevel
    source: str
    message: str
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    acknowledged: bool = False
    metadata: dict = field(default_factory=dict)


class AlertingSystem:
    """Centralised alert dispatcher.

    Collects alerts from all system components and dispatches
    them via configured delivery channels. Maintains a history
    for audit and review.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._handlers: Dict[AlertLevel, List[Callable]] = {
            AlertLevel.INFO: [],
            AlertLevel.WARNING: [],
            AlertLevel.CRITICAL: [],
        }
        self._history: List[Alert] = []
        self._alert_counter = 0
        self._suppress_duplicates_seconds = cfg.get(
            "suppress_duplicates_seconds", 300
        )

    def register_handler(
        self, level: AlertLevel, handler: Callable[[Alert], None]
    ) -> None:
        """Register a delivery handler for an alert level.

        Handlers are called with the Alert object when an alert
        at or above the specified level is dispatched.
        """
        self._handlers[level].append(handler)

    def send(
        self,
        level: AlertLevel,
        source: str,
        message: str,
        metadata: Optional[dict] = None,
    ) -> Alert:
        """Send an alert.

        Parameters
        ----------
        level : AlertLevel
            Alert severity.
        source : str
            Component that raised the alert (e.g. "kill_switch").
        message : str
            Human-readable alert message.
        metadata : dict, optional
            Additional context.

        Returns
        -------
        Alert
        """
        # Check for duplicate suppression
        if self._is_duplicate(source, message, level):
            logger.debug("Suppressed duplicate alert: %s", message)
            return self._history[-1]

        self._alert_counter += 1
        alert = Alert(
            alert_id=f"ALERT-{self._alert_counter:06d}",
            level=level,
            source=source,
            message=message,
            timestamp=pd.Timestamp.now(),
            metadata=metadata or {},
        )

        self._history.append(alert)

        # Log it
        log_fn = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }.get(level, logger.info)
        log_fn("[%s] %s: %s", level.value.upper(), source, message)

        # Dispatch to handlers
        self._dispatch(alert)
        return alert

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_active_alerts(
        self, level: Optional[AlertLevel] = None
    ) -> List[Alert]:
        """Get unacknowledged alerts, optionally filtered by level."""
        alerts = [a for a in self._history if not a.acknowledged]
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        return alerts

    def get_history(
        self,
        since: Optional[pd.Timestamp] = None,
        level: Optional[AlertLevel] = None,
    ) -> List[Alert]:
        """Get alert history with optional filters."""
        alerts = self._history
        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        return alerts

    @property
    def critical_count(self) -> int:
        """Count of unacknowledged critical alerts."""
        return len(self.get_active_alerts(AlertLevel.CRITICAL))

    def _dispatch(self, alert: Alert) -> None:
        """Dispatch alert to registered handlers."""
        # Critical handlers get all levels
        # Warning handlers get WARNING + CRITICAL
        # Info handlers get all
        levels_to_notify = {
            AlertLevel.CRITICAL: [AlertLevel.CRITICAL],
            AlertLevel.WARNING: [AlertLevel.WARNING, AlertLevel.CRITICAL],
            AlertLevel.INFO: [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL],
        }

        for handler_level, trigger_levels in levels_to_notify.items():
            if alert.level in trigger_levels:
                for handler in self._handlers.get(handler_level, []):
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(
                            "Alert handler failed for %s: %s",
                            alert.alert_id, e,
                        )

    def _is_duplicate(
        self, source: str, message: str, level: AlertLevel
    ) -> bool:
        """Check if this is a duplicate of a recent alert."""
        if not self._history:
            return False
        cutoff = pd.Timestamp.now() - pd.Timedelta(
            seconds=self._suppress_duplicates_seconds
        )
        for alert in reversed(self._history):
            if alert.timestamp < cutoff:
                break
            if (alert.source == source and alert.message == message
                    and alert.level == level):
                return True
        return False
