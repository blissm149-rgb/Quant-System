"""Unit tests for VWAP execution algorithm.

TESTING_PLAN.md Section 3.10 — vwap_execution.
"""

import pytest

from quant_fund.broker_interface.broker_abstraction_layer import OrderSide
from quant_fund.execution.execution_algorithms.vwap_execution import (
    VWAPExecution,
    VWAPPlan,
)


@pytest.mark.unit
@pytest.mark.tier2
class TestVWAPExecution:
    """VWAPExecution — time-sliced volume-weighted execution."""

    @pytest.fixture
    def vwap(self):
        return VWAPExecution()

    def test_create_plan_returns_plan(self, vwap):
        """create_plan returns a VWAPPlan with slices."""
        plan = vwap.create_plan("AAPL", OrderSide.BUY, total_shares=1000, adv=1e6)
        assert isinstance(plan, VWAPPlan)
        assert plan.total_shares == 1000
        assert len(plan.slices) > 0

    def test_slices_sum_to_total(self, vwap):
        """Slice target shares sum to total shares."""
        plan = vwap.create_plan("AAPL", OrderSide.BUY, total_shares=1000, adv=1e6)
        total_sliced = sum(s.target_shares for s in plan.slices)
        assert total_sliced == 1000

    def test_get_next_child_order(self, vwap):
        """get_next_child_order returns an order for unfilled slices."""
        plan = vwap.create_plan("AAPL", OrderSide.BUY, total_shares=1000, adv=1e6)
        child = vwap.get_next_child_order(plan, current_price=150.0)
        assert child is not None
        assert child.ticker == "AAPL"
        assert child.side == OrderSide.BUY

    def test_record_fill_updates_progress(self, vwap):
        """record_fill advances the plan's filled_shares."""
        plan = vwap.create_plan("AAPL", OrderSide.BUY, total_shares=1000, adv=1e6)
        vwap.record_fill(plan, filled_shares=100)
        assert plan.filled_shares == 100

    def test_plan_completion(self, vwap):
        """Plan is complete when all shares are filled."""
        plan = vwap.create_plan("AAPL", OrderSide.BUY, total_shares=100, adv=1e6)
        vwap.record_fill(plan, filled_shares=100)
        assert plan.is_complete is True

    def test_remaining_shares(self, vwap):
        """remaining_shares decreases as fills arrive."""
        plan = vwap.create_plan("AAPL", OrderSide.BUY, total_shares=1000, adv=1e6)
        vwap.record_fill(plan, filled_shares=300)
        assert plan.remaining_shares == 700
