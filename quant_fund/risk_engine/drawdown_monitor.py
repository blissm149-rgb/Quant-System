"""Drawdown monitor tracking peak-to-trough NAV drawdown.

Emits alerts at configured thresholds: warning (10%), alert (15%),
critical (20% — triggers kill switch).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class DrawdownAlertLevel(str, Enum):
    WARNING = "WARNING"
    ALERT = "ALERT"
    CRITICAL = "CRITICAL"


@dataclass
class DrawdownAlert:
    """Alert emitted when drawdown exceeds a threshold."""

    level: DrawdownAlertLevel
    drawdown_pct: float
    peak_nav: float
    current_nav: float
    message: str


class DrawdownMonitor:
    """Tracks peak-to-trough NAV drawdown in real time.

    Emits alerts at:
    - 10%: WARNING
    - 15%: ALERT
    - 20%: CRITICAL (kill switch trigger level)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._warning_threshold = cfg.get("drawdown_warning", 0.10)
        self._alert_threshold = cfg.get("drawdown_alert", 0.15)
        self._critical_threshold = cfg.get("drawdown_limit", 0.20)
        self._peak_nav: float = cfg.get("initial_nav", 1_000_000.0)
        self._current_drawdown: float = 0.0

    def update(self, current_nav: float) -> List[DrawdownAlert]:
        """Update with current NAV and return any alerts.

        Args:
            current_nav: Current portfolio NAV.

        Returns:
            List of DrawdownAlert objects (may be empty).
        """
        if current_nav > self._peak_nav:
            self._peak_nav = current_nav

        if self._peak_nav <= 0:
            return []

        self._current_drawdown = (self._peak_nav - current_nav) / self._peak_nav
        alerts = []

        if self._current_drawdown >= self._critical_threshold:
            alerts.append(
                DrawdownAlert(
                    level=DrawdownAlertLevel.CRITICAL,
                    drawdown_pct=self._current_drawdown,
                    peak_nav=self._peak_nav,
                    current_nav=current_nav,
                    message=f"CRITICAL: Drawdown {self._current_drawdown:.2%} >= {self._critical_threshold:.2%}",
                )
            )
        elif self._current_drawdown >= self._alert_threshold:
            alerts.append(
                DrawdownAlert(
                    level=DrawdownAlertLevel.ALERT,
                    drawdown_pct=self._current_drawdown,
                    peak_nav=self._peak_nav,
                    current_nav=current_nav,
                    message=f"ALERT: Drawdown {self._current_drawdown:.2%} >= {self._alert_threshold:.2%}",
                )
            )
        elif self._current_drawdown >= self._warning_threshold:
            alerts.append(
                DrawdownAlert(
                    level=DrawdownAlertLevel.WARNING,
                    drawdown_pct=self._current_drawdown,
                    peak_nav=self._peak_nav,
                    current_nav=current_nav,
                    message=f"WARNING: Drawdown {self._current_drawdown:.2%} >= {self._warning_threshold:.2%}",
                )
            )

        return alerts

    @property
    def current_drawdown(self) -> float:
        return self._current_drawdown

    @property
    def peak_nav(self) -> float:
        return self._peak_nav
