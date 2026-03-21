"""Unit tests for system state machine.

TESTING_PLAN.md Section 3.12 — system_state_machine (CRITICAL).
"""

import pytest

from quant_fund.infrastructure.system_state_machine import (
    InvalidTransitionError,
    SystemState,
    SystemStateMachine,
)


@pytest.mark.unit
@pytest.mark.tier1
class TestSystemStateMachine:
    """SystemStateMachine — CRITICAL: valid transitions only."""

    @pytest.fixture
    def sm(self):
        return SystemStateMachine()

    def test_initial_state_is_initializing(self, sm):
        """Default initial state is INITIALIZING."""
        assert sm.state == SystemState.INITIALIZING

    def test_valid_transition_init_to_data_ready(self, sm):
        """INITIALIZING → DATA_READY is valid."""
        transition = sm.transition_to(SystemState.DATA_READY, reason="data loaded")
        assert sm.state == SystemState.DATA_READY
        assert transition.from_state == SystemState.INITIALIZING
        assert transition.to_state == SystemState.DATA_READY

    def test_valid_transition_data_ready_to_trading(self, sm):
        """DATA_READY → TRADING_ENABLED is valid."""
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        assert sm.state == SystemState.TRADING_ENABLED

    def test_invalid_transition_raises(self, sm):
        """INITIALIZING → TRADING_ENABLED is invalid."""
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(SystemState.TRADING_ENABLED)

    def test_invalid_transition_init_to_risk_halt(self, sm):
        """INITIALIZING → RISK_HALT is invalid."""
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(SystemState.RISK_HALT)

    def test_try_transition_returns_none_on_invalid(self, sm):
        """try_transition returns None instead of raising."""
        result = sm.try_transition(SystemState.TRADING_ENABLED)
        assert result is None
        assert sm.state == SystemState.INITIALIZING  # unchanged

    def test_trading_to_risk_halt(self, sm):
        """TRADING_ENABLED → RISK_HALT is valid."""
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        sm.transition_to(SystemState.RISK_HALT)
        assert sm.state == SystemState.RISK_HALT

    def test_risk_halt_to_trading(self, sm):
        """RISK_HALT → TRADING_ENABLED is valid (recovery)."""
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        sm.transition_to(SystemState.RISK_HALT)
        sm.transition_to(SystemState.TRADING_ENABLED)
        assert sm.state == SystemState.TRADING_ENABLED

    def test_shutdown_is_terminal(self, sm):
        """SHUTDOWN has no outbound transitions."""
        sm.transition_to(SystemState.SHUTDOWN)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(SystemState.INITIALIZING)

    def test_can_trade_property(self, sm):
        """can_trade is True only in TRADING_ENABLED."""
        assert sm.can_trade is False
        sm.transition_to(SystemState.DATA_READY)
        assert sm.can_trade is False
        sm.transition_to(SystemState.TRADING_ENABLED)
        assert sm.can_trade is True

    def test_is_halted_property(self, sm):
        """is_halted is True only in RISK_HALT."""
        assert sm.is_halted is False
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        sm.transition_to(SystemState.RISK_HALT)
        assert sm.is_halted is True

    def test_transition_history(self, sm):
        """Transitions are recorded in history."""
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        history = sm.get_history()
        assert len(history) == 2

    def test_transition_callback(self, sm):
        """Registered callback fires on transition."""
        calls = []
        sm.on_transition(lambda t: calls.append(t))
        sm.transition_to(SystemState.DATA_READY)
        assert len(calls) == 1

    def test_to_dict_and_restore(self, sm):
        """State machine can be serialized and restored."""
        sm.transition_to(SystemState.DATA_READY)
        data = sm.to_dict()
        sm2 = SystemStateMachine()
        sm2.restore_from_dict(data)
        assert sm2.state == SystemState.DATA_READY
