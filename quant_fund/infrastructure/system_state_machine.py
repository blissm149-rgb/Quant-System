"""System state machine — explicit lifecycle control for the trading engine.

The trading system operates under strict state control. Trading may
only occur in the TRADING_ENABLED state. All state transitions are
validated and logged.

Valid states:
    INITIALIZING    — system is starting up, loading state
    DATA_READY      — market data feed is connected and validated
    TRADING_ENABLED — actively trading
    RISK_HALT       — risk limit breached, trading suspended
    SHUTDOWN        — graceful shutdown in progress

Valid transitions:
    INITIALIZING    → DATA_READY       (data feed connected)
    INITIALIZING    → SHUTDOWN         (startup failure)
    DATA_READY      → TRADING_ENABLED  (readiness checks passed)
    DATA_READY      → SHUTDOWN         (operator request)
    TRADING_ENABLED → RISK_HALT        (kill switch / exposure breach)
    TRADING_ENABLED → DATA_READY       (data feed lost)
    TRADING_ENABLED → SHUTDOWN         (end of day / operator request)
    RISK_HALT       → TRADING_ENABLED  (manual reset after review)
    RISK_HALT       → SHUTDOWN         (operator request)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SystemState(str, Enum):
    """System lifecycle states."""

    INITIALIZING = "INITIALIZING"
    DATA_READY = "DATA_READY"
    TRADING_ENABLED = "TRADING_ENABLED"
    RISK_HALT = "RISK_HALT"
    SHUTDOWN = "SHUTDOWN"


# Legal state transitions: from_state → set of allowed to_states
VALID_TRANSITIONS: Dict[SystemState, Set[SystemState]] = {
    SystemState.INITIALIZING: {
        SystemState.DATA_READY,
        SystemState.SHUTDOWN,
    },
    SystemState.DATA_READY: {
        SystemState.TRADING_ENABLED,
        SystemState.SHUTDOWN,
        SystemState.INITIALIZING,  # re-init after data loss
    },
    SystemState.TRADING_ENABLED: {
        SystemState.RISK_HALT,
        SystemState.DATA_READY,
        SystemState.SHUTDOWN,
    },
    SystemState.RISK_HALT: {
        SystemState.TRADING_ENABLED,
        SystemState.SHUTDOWN,
    },
    SystemState.SHUTDOWN: set(),  # terminal state
}


@dataclass
class StateTransition:
    """Record of a state transition."""

    from_state: SystemState
    to_state: SystemState
    reason: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# Type alias for transition callbacks
TransitionCallback = Callable[[StateTransition], None]


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, from_state: SystemState, to_state: SystemState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid transition: {from_state.value} → {to_state.value}"
        )


@dataclass
class ReadinessCheck:
    """Result of a single readiness check."""

    name: str
    passed: bool
    message: str = ""


class SystemStateMachine:
    """Controls system lifecycle and enforces state-dependent behavior.

    Usage:
        sm = SystemStateMachine()
        sm.on_transition(my_callback)

        # Startup
        sm.transition_to(SystemState.DATA_READY, reason="Data feed connected")
        sm.transition_to(SystemState.TRADING_ENABLED, reason="All checks passed")

        # Check before trading
        if sm.can_trade:
            execute_strategy()

        # Risk halt
        sm.transition_to(SystemState.RISK_HALT, reason="Kill switch triggered")
    """

    def __init__(self, initial_state: SystemState = SystemState.INITIALIZING):
        self._state = initial_state
        self._history: List[StateTransition] = []
        self._callbacks: List[TransitionCallback] = []
        self._readiness_checks: Dict[str, Callable[[], ReadinessCheck]] = {}

        logger.info("SystemStateMachine initialized in state %s", self._state.value)

    # ------------------------------------------------------------------
    # State query
    # ------------------------------------------------------------------

    @property
    def state(self) -> SystemState:
        """Current system state."""
        return self._state

    @property
    def can_trade(self) -> bool:
        """True only when in TRADING_ENABLED state."""
        return self._state == SystemState.TRADING_ENABLED

    @property
    def is_halted(self) -> bool:
        """True when in RISK_HALT state."""
        return self._state == SystemState.RISK_HALT

    @property
    def is_shutdown(self) -> bool:
        """True when in SHUTDOWN state."""
        return self._state == SystemState.SHUTDOWN

    @property
    def is_initializing(self) -> bool:
        return self._state == SystemState.INITIALIZING

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition_to(
        self,
        new_state: SystemState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateTransition:
        """Transition to a new state.

        Args:
            new_state: Target state.
            reason: Human-readable reason for the transition.
            metadata: Optional additional context.

        Returns:
            StateTransition record.

        Raises:
            InvalidTransitionError: If the transition is not allowed.
        """
        if new_state not in VALID_TRANSITIONS.get(self._state, set()):
            raise InvalidTransitionError(self._state, new_state)

        transition = StateTransition(
            from_state=self._state,
            to_state=new_state,
            reason=reason,
            metadata=metadata or {},
        )

        old_state = self._state
        self._state = new_state
        self._history.append(transition)

        logger.info(
            "State transition: %s → %s (reason: %s)",
            old_state.value,
            new_state.value,
            reason,
        )

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(transition)
            except Exception:
                logger.exception("Transition callback failed")

        return transition

    def try_transition(
        self,
        new_state: SystemState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StateTransition]:
        """Attempt a transition without raising on failure.

        Returns the transition if successful, None if invalid.
        """
        if new_state not in VALID_TRANSITIONS.get(self._state, set()):
            logger.debug(
                "Transition %s → %s not valid, ignoring",
                self._state.value,
                new_state.value,
            )
            return None
        return self.transition_to(new_state, reason, metadata)

    # ------------------------------------------------------------------
    # Readiness checks
    # ------------------------------------------------------------------

    def register_readiness_check(
        self, name: str, check_fn: Callable[[], ReadinessCheck]
    ) -> None:
        """Register a readiness check for the INITIALIZING → DATA_READY transition.

        All registered checks must pass before the system can enter
        TRADING_ENABLED state.
        """
        self._readiness_checks[name] = check_fn

    def run_readiness_checks(self) -> Tuple[bool, List[ReadinessCheck]]:
        """Run all registered readiness checks.

        Returns:
            Tuple of (all_passed, list_of_check_results).
        """
        results = []
        for name, check_fn in self._readiness_checks.items():
            try:
                result = check_fn()
            except Exception as e:
                result = ReadinessCheck(
                    name=name, passed=False, message=f"Check raised: {e}"
                )
            results.append(result)

        all_passed = all(r.passed for r in results)
        return all_passed, results

    def transition_if_ready(
        self, target: SystemState, reason: str = "Readiness checks passed"
    ) -> Tuple[bool, List[ReadinessCheck]]:
        """Run readiness checks and transition if all pass.

        Returns:
            Tuple of (transitioned, check_results).
        """
        all_passed, results = self.run_readiness_checks()
        if all_passed:
            self.transition_to(target, reason=reason)
        else:
            failed = [r for r in results if not r.passed]
            logger.warning(
                "Readiness checks failed: %s",
                [(r.name, r.message) for r in failed],
            )
        return all_passed, results

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_transition(self, callback: TransitionCallback) -> None:
        """Register a callback invoked on every state transition."""
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self) -> List[StateTransition]:
        """Return full transition history."""
        return list(self._history)

    @property
    def last_transition(self) -> Optional[StateTransition]:
        return self._history[-1] if self._history else None

    def time_in_current_state(self) -> float:
        """Seconds spent in the current state."""
        if not self._history:
            return 0.0
        last_ts = datetime.fromisoformat(self._history[-1].timestamp)
        now = datetime.now(timezone.utc)
        return (now - last_ts).total_seconds()

    # ------------------------------------------------------------------
    # Serialization for StateStore
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state machine for persistence."""
        return {
            "current_state": self._state.value,
            "history": [t.to_dict() for t in self._history[-50:]],
        }

    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        """Restore state from a persisted dict.

        Sets the state directly without transition validation,
        since we're restoring from a known-good snapshot.
        """
        state_str = data.get("current_state", SystemState.INITIALIZING.value)
        self._state = SystemState(state_str)
        logger.info("Restored state machine to %s", self._state.value)
