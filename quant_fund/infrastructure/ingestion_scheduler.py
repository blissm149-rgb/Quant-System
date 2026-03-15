"""Lightweight data ingestion scheduler.

Parses schedule definitions from ``data_config.yaml`` and provides
a run loop that invokes registered ingestion callables at their
configured times. Uses only stdlib — no APScheduler dependency.

Schedule format examples (from data_config.yaml):
- ``"09:30-16:00/5min"`` — every 5 minutes between 09:30 and 16:00
- ``"16:30"`` — once daily at 16:30
- ``"06:00"`` — once daily at 06:00

This is a cooperative scheduler: callers invoke ``tick()`` from their
own event loop or timer, and the scheduler decides which jobs are due.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ScheduleEntry:
    """A parsed schedule definition."""

    name: str
    schedule_str: str
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    interval_minutes: Optional[int] = None
    callback: Optional[Callable] = None
    last_run: Optional[datetime] = None
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "schedule_str": self.schedule_str,
            "start_time": str(self.start_time) if self.start_time else None,
            "end_time": str(self.end_time) if self.end_time else None,
            "interval_minutes": self.interval_minutes,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "enabled": self.enabled,
        }


# Regex patterns for schedule parsing
_TIME_ONLY = re.compile(r"^(\d{1,2}):(\d{2})$")
_RANGE_INTERVAL = re.compile(
    r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})/(\d+)(min|h)$"
)


class IngestionScheduler:
    """Cooperative scheduler for data ingestion tasks.

    The scheduler does not run its own background thread. Instead,
    the caller should invoke ``tick(now)`` periodically (e.g., every
    minute) and the scheduler will fire any callbacks that are due.

    Usage:
        scheduler = IngestionScheduler.from_config_file("config/data_config.yaml")
        scheduler.register_callback("price_update", my_price_fetcher)
        scheduler.register_callback("daily_close", my_close_fetcher)

        # In your main loop:
        while running:
            results = scheduler.tick()
            time.sleep(60)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or {}
        self._schedules: Dict[str, ScheduleEntry] = {}
        self._run_history: List[Dict] = []

        # Parse schedules from config
        sched_cfg = self._config.get("schedules", {})
        for name, sched_str in sched_cfg.items():
            entry = self._parse_schedule(name, sched_str)
            if entry:
                self._schedules[name] = entry

        logger.info(
            "IngestionScheduler initialized with %d schedules: %s",
            len(self._schedules),
            list(self._schedules.keys()),
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> "IngestionScheduler":
        """Construct from a YAML config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_callback(
        self, schedule_name: str, callback: Callable
    ) -> bool:
        """Attach a callable to a named schedule.

        Args:
            schedule_name: Must match a schedule name from config.
            callback: Zero-argument callable to invoke when due.

        Returns:
            True if the schedule exists and was bound, False otherwise.
        """
        if schedule_name not in self._schedules:
            logger.warning(
                "No schedule named '%s' — available: %s",
                schedule_name,
                list(self._schedules.keys()),
            )
            return False

        self._schedules[schedule_name].callback = callback
        logger.info("Registered callback for schedule '%s'", schedule_name)
        return True

    def add_schedule(
        self,
        name: str,
        schedule_str: str,
        callback: Optional[Callable] = None,
    ) -> bool:
        """Add a new schedule entry programmatically.

        Args:
            name: Unique schedule name.
            schedule_str: Schedule definition string.
            callback: Optional callable to invoke when due.

        Returns:
            True if the schedule was parsed and added successfully.
        """
        entry = self._parse_schedule(name, schedule_str)
        if entry is None:
            return False

        entry.callback = callback
        self._schedules[name] = entry
        return True

    # ------------------------------------------------------------------
    # Tick / execution
    # ------------------------------------------------------------------

    def tick(self, now: Optional[datetime] = None) -> List[Dict]:
        """Check all schedules and fire any that are due.

        Args:
            now: Current time (default: ``datetime.now(timezone.utc)``).
                 Accepting this as a parameter makes testing deterministic.

        Returns:
            List of dicts describing which schedules fired and their results.
        """
        now = now or datetime.now(timezone.utc)
        current_time = now.time()
        results: List[Dict] = []

        for name, entry in self._schedules.items():
            if not entry.enabled:
                continue

            if not self._is_due(entry, now, current_time):
                continue

            result = self._execute_entry(entry, now)
            results.append(result)

        return results

    def get_schedules(self) -> List[Dict]:
        """Return all schedule entries as dicts."""
        return [e.to_dict() for e in self._schedules.values()]

    def get_run_history(self, limit: int = 50) -> List[Dict]:
        """Return recent execution history."""
        return self._run_history[-limit:]

    def enable_schedule(self, name: str) -> None:
        """Enable a schedule."""
        if name in self._schedules:
            self._schedules[name].enabled = True

    def disable_schedule(self, name: str) -> None:
        """Disable a schedule without removing it."""
        if name in self._schedules:
            self._schedules[name].enabled = False

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_schedule(name: str, schedule_str: str) -> Optional[ScheduleEntry]:
        """Parse a schedule string into a ScheduleEntry.

        Supported formats:
        - ``"HH:MM"`` — run once daily at that time
        - ``"HH:MM-HH:MM/Nmin"`` — run every N minutes in that window
        - ``"HH:MM-HH:MM/Nh"`` — run every N hours in that window
        """
        schedule_str = schedule_str.strip()

        # Try time-only pattern: "16:30"
        m = _TIME_ONLY.match(schedule_str)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            return ScheduleEntry(
                name=name,
                schedule_str=schedule_str,
                start_time=time(h, mi),
                end_time=time(h, mi),
                interval_minutes=None,  # once daily
            )

        # Try range+interval: "09:30-16:00/5min"
        m = _RANGE_INTERVAL.match(schedule_str)
        if m:
            sh, sm = int(m.group(1)), int(m.group(2))
            eh, em = int(m.group(3)), int(m.group(4))
            interval = int(m.group(5))
            unit = m.group(6)
            if unit == "h":
                interval *= 60

            return ScheduleEntry(
                name=name,
                schedule_str=schedule_str,
                start_time=time(sh, sm),
                end_time=time(eh, em),
                interval_minutes=interval,
            )

        logger.warning(
            "Could not parse schedule '%s' for '%s'", schedule_str, name
        )
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_due(
        self, entry: ScheduleEntry, now: datetime, current_time: time
    ) -> bool:
        """Determine if a schedule entry should fire at *now*."""
        if entry.start_time is None:
            return False

        if entry.interval_minutes is None:
            # Once-daily: fire if current minute matches and hasn't run today
            if current_time.hour != entry.start_time.hour:
                return False
            if current_time.minute != entry.start_time.minute:
                return False
            if entry.last_run and entry.last_run.date() == now.date():
                return False
            return True

        # Interval-based: check if within window and enough time elapsed
        if current_time < entry.start_time or current_time > entry.end_time:
            return False

        if entry.last_run is None:
            return True

        elapsed = (now - entry.last_run).total_seconds() / 60
        return elapsed >= entry.interval_minutes

    def _execute_entry(self, entry: ScheduleEntry, now: datetime) -> Dict:
        """Execute a schedule entry's callback."""
        result: Dict[str, Any] = {
            "schedule": entry.name,
            "fired_at": now.isoformat(),
            "status": "no_callback",
        }

        if entry.callback is not None:
            try:
                entry.callback()
                result["status"] = "success"
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)
                logger.exception(
                    "Schedule '%s' callback failed: %s", entry.name, exc
                )

        entry.last_run = now
        self._run_history.append(result)

        logger.info(
            "Schedule '%s' fired at %s — status=%s",
            entry.name,
            now.isoformat(),
            result["status"],
        )
        return result
