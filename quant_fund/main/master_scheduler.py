"""Master scheduler -- 24/7 orchestrator for all operational phases.

Maintains a NYSE trading calendar and automatically transitions the
system between operational phases:

    04:00 ET -> PRE_MARKET   (data refresh, model warm-up, broker check)
    09:25 ET -> DATA_READY   (readiness checks)
    09:30 ET -> TRADING_ENABLED (live trading)
    16:00 ET -> POST_MARKET  (settlement, EOD reporting)
    17:30 ET -> OVERNIGHT    (training, ingestion, maintenance)
    04:00 ET -> PRE_MARKET   (next day cycle)

    Saturday/Sunday -> WEEKEND_MAINTENANCE
    NYSE holidays   -> OVERNIGHT (skip trading, run maintenance)
"""

import logging
import time as _time
from dataclasses import dataclass
from datetime import date, time, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from quant_fund.execution.order_management.market_hours_enforcer import (
    MarketHoursEnforcer,
)
from quant_fund.infrastructure.system_state_machine import (
    SystemState,
    SystemStateMachine,
)

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


@dataclass
class PhaseSchedule:
    """Defines when each operational phase begins (Eastern Time)."""

    pre_market: time = time(4, 0)
    data_ready: time = time(9, 25)
    trading_open: time = time(9, 30)
    post_market: time = time(16, 0)
    overnight: time = time(17, 30)


class MasterScheduler:
    """24/7 orchestrator that transitions between operational phases.

    Usage:
        scheduler = MasterScheduler(state_machine=sm)
        scheduler.register_phase_handler(SystemState.PRE_MARKET, pre_market_fn)
        scheduler.register_phase_handler(SystemState.POST_MARKET, post_market_fn)
        scheduler.register_phase_handler(SystemState.OVERNIGHT, overnight_fn)
        scheduler.register_phase_handler(SystemState.WEEKEND_MAINTENANCE, weekend_fn)

        # In the main loop:
        scheduler.tick()  # checks time and transitions as needed
    """

    def __init__(
        self,
        state_machine: SystemStateMachine,
        market_hours: Optional[MarketHoursEnforcer] = None,
        schedule: Optional[PhaseSchedule] = None,
        config: Optional[dict] = None,
    ):
        cfg = config or {}
        self._sm = state_machine
        self._market_hours = market_hours or MarketHoursEnforcer()
        self._schedule = schedule or PhaseSchedule()
        self._phase_handlers: Dict[SystemState, Callable] = {}
        self._last_phase_run: Dict[SystemState, date] = {}
        self._tick_interval_s: float = cfg.get("scheduler_tick_interval_s", 10.0)

    def register_phase_handler(
        self, phase: SystemState, handler: Callable[[], None]
    ) -> None:
        """Register a callable to run when entering a phase.

        The handler is called once per calendar day when the phase
        becomes active. It should be idempotent.
        """
        self._phase_handlers[phase] = handler

    def tick(self) -> Optional[SystemState]:
        """Check the current time and transition if needed.

        Returns the new state if a transition occurred, else None.
        """
        now_et = pd.Timestamp.now(tz=ET)
        today = now_et.date()
        current_time = now_et.time()
        target = self._determine_target_phase(today, current_time)

        if target is None or target == self._sm.state:
            return None

        transition = self._sm.try_transition(
            target,
            reason=f"MasterScheduler phase change at {now_et.strftime('%H:%M ET')}",
        )

        if transition is not None:
            logger.info(
                "MasterScheduler: %s -> %s",
                transition.from_state.value,
                target.value,
            )
            self._run_phase_handler(target, today)
            return target

        return None

    def _determine_target_phase(
        self, today: date, current_time: time
    ) -> Optional[SystemState]:
        """Determine which phase the system should be in right now."""
        # Weekend -> WEEKEND_MAINTENANCE
        if today.weekday() >= 5:
            return SystemState.WEEKEND_MAINTENANCE

        # Holiday -> OVERNIGHT (maintenance without trading)
        if self._market_hours.is_holiday(pd.Timestamp(
            year=today.year, month=today.month, day=today.day, tz=ET,
        )):
            return SystemState.OVERNIGHT

        # Weekday phase schedule
        s = self._schedule
        if current_time < s.pre_market:
            return SystemState.OVERNIGHT
        elif current_time < s.data_ready:
            return SystemState.PRE_MARKET
        elif current_time < s.trading_open:
            return SystemState.DATA_READY
        elif current_time < s.post_market:
            return SystemState.TRADING_ENABLED
        elif current_time < s.overnight:
            return SystemState.POST_MARKET
        else:
            return SystemState.OVERNIGHT

    def _run_phase_handler(self, phase: SystemState, today: date) -> None:
        """Run the registered handler for a phase, at most once per day."""
        if phase not in self._phase_handlers:
            return

        last_run = self._last_phase_run.get(phase)
        if last_run == today:
            logger.debug("Phase handler for %s already ran today", phase.value)
            return

        handler = self._phase_handlers[phase]
        try:
            logger.info("Running phase handler for %s", phase.value)
            handler()
            self._last_phase_run[phase] = today
        except Exception:
            logger.exception("Phase handler for %s failed", phase.value)

    def get_next_transition(self) -> Optional[str]:
        """Return a human-readable description of the next phase transition."""
        now_et = pd.Timestamp.now(tz=ET)
        today = now_et.date()
        current_time = now_et.time()

        if today.weekday() >= 5:
            # Weekend: next transition is Monday pre-market
            days_until_monday = 7 - today.weekday()
            monday = today + timedelta(days=days_until_monday)
            return f"PRE_MARKET at {monday} {self._schedule.pre_market}"

        s = self._schedule
        transitions = [
            (s.pre_market, "PRE_MARKET"),
            (s.data_ready, "DATA_READY"),
            (s.trading_open, "TRADING_ENABLED"),
            (s.post_market, "POST_MARKET"),
            (s.overnight, "OVERNIGHT"),
        ]

        for t, name in transitions:
            if current_time < t:
                return f"{name} at {t.strftime('%H:%M')} ET"

        # Past overnight start: next is tomorrow pre-market
        tomorrow = today + timedelta(days=1)
        return f"PRE_MARKET at {tomorrow} {self._schedule.pre_market}"

    @property
    def schedule(self) -> PhaseSchedule:
        return self._schedule
