"""Test VWAP execution plan quality.

Validates that VWAP execution plans distribute shares correctly,
respect participation constraints, and track fills properly.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.execution.execution_algorithms.vwap_execution import (
    VWAPExecution,
    VWAPPlan,
    DEFAULT_VOLUME_PROFILE,
)
from quant_fund.broker_interface.broker_abstraction_layer import OrderSide


pytestmark = [pytest.mark.validation]


class TestVWAPExecutionQuality:
    """Verify VWAP execution plans are well-formed and respect constraints."""

    @pytest.fixture(autouse=True)
    def setup_vwap(self):
        self.vwap = VWAPExecution()

    def test_plan_slices_sum_to_effective_shares(self):
        """Total target shares across all slices should equal the effective
        share count (capped by participation rate * ADV)."""
        adv = 1_000_000
        total_shares = 100_000
        plan = self.vwap.create_plan("AAPL", OrderSide.BUY, total_shares, adv=adv)

        effective = min(total_shares, int(adv * plan.participation_rate))
        slice_total = sum(s.target_shares for s in plan.slices)

        assert slice_total == effective, (
            f"Sum of slice targets ({slice_total}) should equal "
            f"effective shares ({effective})"
        )

    def test_participation_rate_caps_order(self):
        """When total_shares > participation_rate * ADV, the plan should
        cap effective shares to avoid excessive market participation."""
        adv = 100_000  # 100k shares ADV
        total_shares = 50_000  # 50% of ADV — should be capped to 5%
        plan = self.vwap.create_plan("MSFT", OrderSide.BUY, total_shares, adv=adv)

        max_allowed = int(adv * 0.05)  # 5% participation = 5,000
        slice_total = sum(s.target_shares for s in plan.slices)

        assert slice_total <= max_allowed, (
            f"Slice total ({slice_total}) should be capped at "
            f"participation rate * ADV ({max_allowed})"
        )

    def test_fill_tracking_updates_plan(self):
        """Recording fills should correctly update plan and slice state."""
        plan = self.vwap.create_plan("GOOG", OrderSide.SELL, 10_000, adv=1_000_000)

        # Fill half the first slice
        first_slice_target = plan.slices[0].target_shares
        if first_slice_target > 0:
            half = max(1, first_slice_target // 2)
            self.vwap.record_fill(plan, half)

            assert plan.filled_shares == half
            assert plan.slices[0].filled_shares == half
            assert not plan.is_complete

    def test_no_slice_exceeds_max_slice_pct(self):
        """No individual slice should exceed max_slice_pct of total shares."""
        total = 50_000
        plan = self.vwap.create_plan("AMZN", OrderSide.BUY, total, adv=10_000_000)

        max_slice = int(total * 0.20)  # default max_slice_pct = 0.20
        for s in plan.slices:
            assert s.target_shares <= max_slice + 1, (
                f"Slice {s.slice_index} has {s.target_shares} shares, "
                f"exceeds max_slice_pct cap of {max_slice}"
            )
