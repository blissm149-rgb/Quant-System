"""
Market hours enforcer for US equity trading.

Enforces NYSE trading hours, handles extended hours sessions,
EOD cutoffs, weekends, and NYSE holidays (2024-2026).
"""

from datetime import date, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# NYSE Holiday Calendar (2024-2026)
# ---------------------------------------------------------------------------

# Full-day closures keyed by year
_NYSE_HOLIDAYS: Dict[int, List[date]] = {
    2024: [
        date(2024, 1, 1),    # New Year's Day
        date(2024, 1, 15),   # MLK Day
        date(2024, 2, 19),   # Presidents Day
        date(2024, 3, 29),   # Good Friday
        date(2024, 5, 27),   # Memorial Day
        date(2024, 6, 19),   # Juneteenth
        date(2024, 7, 4),    # Independence Day
        date(2024, 9, 2),    # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
    ],
    2025: [
        date(2025, 1, 1),    # New Year's Day
        date(2025, 1, 20),   # MLK Day
        date(2025, 2, 17),   # Presidents Day
        date(2025, 4, 18),   # Good Friday
        date(2025, 5, 26),   # Memorial Day
        date(2025, 6, 19),   # Juneteenth
        date(2025, 7, 4),    # Independence Day
        date(2025, 9, 1),    # Labor Day
        date(2025, 11, 27),  # Thanksgiving
        date(2025, 12, 25),  # Christmas
    ],
    2026: [
        date(2026, 1, 1),    # New Year's Day
        date(2026, 1, 19),   # MLK Day
        date(2026, 2, 16),   # Presidents Day
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Independence Day (observed - July 4 is Saturday)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
    ],
}

# Early close days (1:00 PM ET)
_NYSE_EARLY_CLOSE: Dict[int, List[date]] = {
    2024: [
        date(2024, 7, 3),    # Day before Independence Day
        date(2024, 11, 29),  # Black Friday
        date(2024, 12, 24),  # Christmas Eve
    ],
    2025: [
        date(2025, 7, 3),    # Day before Independence Day
        date(2025, 11, 28),  # Black Friday
        date(2025, 12, 24),  # Christmas Eve
    ],
    2026: [
        date(2026, 11, 27),  # Black Friday
        date(2026, 12, 24),  # Christmas Eve
    ],
}

# Flatten into sets for fast lookup
_ALL_HOLIDAYS = {d for dates in _NYSE_HOLIDAYS.values() for d in dates}
_ALL_EARLY_CLOSE = {d for dates in _NYSE_EARLY_CLOSE.values() for d in dates}


class MarketHoursEnforcer:
    """Enforces US equity market (NYSE) trading hours."""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.market_open_time = self._parse_time(
            config.get("market_open_time", "09:30")
        )
        self.market_close_time = self._parse_time(
            config.get("market_close_time", "16:00")
        )
        self.eod_cutoff_minutes: int = config.get("eod_cutoff_minutes", 5)
        self.allow_extended_hours: bool = config.get("allow_extended_hours", False)
        self.extended_open = self._parse_time(
            config.get("extended_open", "04:00")
        )
        self.extended_close = self._parse_time(
            config.get("extended_close", "20:00")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_market_open(self, timestamp: pd.Timestamp) -> bool:
        """Return True if *timestamp* falls within regular market hours."""
        ts_et = self._to_eastern(timestamp)
        current_date = ts_et.date()

        # Weekend check
        if current_date.weekday() >= 5:
            return False

        # Holiday check
        if current_date in _ALL_HOLIDAYS:
            return False

        current_time = ts_et.time()
        close = self._effective_close(current_date)
        return self.market_open_time <= current_time < close

    def can_submit_order(
        self, timestamp: pd.Timestamp
    ) -> Tuple[bool, str]:
        """Check whether a new order may be submitted at *timestamp*.

        Returns
        -------
        (allowed, reason) : Tuple[bool, str]
            *allowed* is True when submission is permitted.
            *reason* contains a human-readable explanation when rejected,
            or ``"OK"`` when allowed.
        """
        ts_et = self._to_eastern(timestamp)
        current_date = ts_et.date()
        current_time = ts_et.time()

        # --- Weekday check ---
        if current_date.weekday() >= 5:
            return False, "Market is closed on weekends"

        # --- Holiday check ---
        if current_date in _ALL_HOLIDAYS:
            return False, "Market is closed for a NYSE holiday"

        # Determine applicable session window
        if self.allow_extended_hours:
            session_open = self.extended_open
            session_close = self.extended_close
        else:
            session_open = self.market_open_time
            session_close = self._effective_close(current_date)

        # --- Outside session window ---
        if current_time < session_open or current_time >= session_close:
            if self.allow_extended_hours:
                return False, "Outside extended trading hours"
            return False, "Outside regular market hours"

        # --- EOD cutoff (applies to regular close, not extended) ---
        effective_close = self._effective_close(current_date)
        cutoff = self._subtract_minutes(effective_close, self.eod_cutoff_minutes)
        if current_time >= cutoff and current_time < effective_close:
            # During extended hours the cutoff still applies relative to
            # the regular close – new orders are blocked near close.
            return (
                False,
                f"Within EOD cutoff ({self.eod_cutoff_minutes} minutes before market close)",
            )

        return True, "OK"

    def next_market_open(self, from_timestamp: pd.Timestamp) -> pd.Timestamp:
        """Return the next regular-session market open at or after *from_timestamp*."""
        ts_et = self._to_eastern(from_timestamp)
        candidate = ts_et.normalize()  # midnight of that day

        # If we haven't passed market open today, today might be the answer
        if ts_et.time() < self.market_open_time:
            candidate_date = ts_et.date()
        else:
            # Already past open today, start looking from tomorrow
            candidate_date = ts_et.date() + timedelta(days=1)

        # Walk forward until we find a valid trading day
        for _ in range(10):  # at most ~10 days ahead (holiday + weekend)
            if candidate_date.weekday() < 5 and candidate_date not in _ALL_HOLIDAYS:
                open_dt = pd.Timestamp(
                    year=candidate_date.year,
                    month=candidate_date.month,
                    day=candidate_date.day,
                    hour=self.market_open_time.hour,
                    minute=self.market_open_time.minute,
                    tz=ET,
                )
                return open_dt
            candidate_date += timedelta(days=1)

        # Fallback – should never be reached in practice
        raise RuntimeError("Could not determine next market open within 10 days")

    def is_holiday(self, date_input: pd.Timestamp) -> bool:
        """Return True if *date_input* is a NYSE full-day holiday (2024-2026)."""
        d = self._to_eastern(date_input).date()
        return d in _ALL_HOLIDAYS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_time(value: str) -> time:
        """Parse an ``"HH:MM"`` string into a :class:`datetime.time`."""
        parts = value.split(":")
        return time(int(parts[0]), int(parts[1]))

    @staticmethod
    def _to_eastern(ts: pd.Timestamp) -> pd.Timestamp:
        """Convert a pandas Timestamp to US/Eastern."""
        if ts.tzinfo is None:
            return ts.tz_localize(ET)
        return ts.tz_convert(ET)

    def _effective_close(self, d: date) -> time:
        """Return the close time for *d*, accounting for early close days."""
        if d in _ALL_EARLY_CLOSE:
            return time(13, 0)
        return self.market_close_time

    @staticmethod
    def _subtract_minutes(t: time, minutes: int) -> time:
        """Subtract *minutes* from a :class:`datetime.time`."""
        total = t.hour * 60 + t.minute - minutes
        return time(total // 60, total % 60)
