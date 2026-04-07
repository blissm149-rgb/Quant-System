"""Test execution quality drift detection.

Validates that ExecutionQualityMonitor correctly identifies
deteriorating execution quality via implementation shortfall,
fill rates, adverse selection, and slippage drift.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.monitoring.execution_quality_monitor import (
    ExecutionQualityMonitor,
    ExecutionRecord,
    ExecutionSummary,
)


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestImplementationShortfall:
    """Validate implementation shortfall calculation correctness."""

    def test_buy_positive_shortfall_when_fill_above_decision(self):
        """Buy order filled above decision price should have positive shortfall."""
        eqm = ExecutionQualityMonitor()
        rec = eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=1000, filled_qty=1000,
            decision_price=150.0, fill_price=151.0,
        )
        expected_bps = (151.0 - 150.0) / 150.0 * 10000
        assert rec.implementation_shortfall_bps == pytest.approx(expected_bps, abs=0.1)

    def test_sell_positive_shortfall_when_fill_below_decision(self):
        """Sell order filled below decision price should have positive shortfall."""
        eqm = ExecutionQualityMonitor()
        rec = eqm.record_execution(
            order_id="O1", ticker="AAPL", side="sell",
            target_qty=1000, filled_qty=1000,
            decision_price=150.0, fill_price=149.0,
        )
        expected_bps = (150.0 - 149.0) / 150.0 * 10000
        assert rec.implementation_shortfall_bps == pytest.approx(expected_bps, abs=0.1)

    def test_perfect_fill_zero_shortfall(self):
        """Fill at decision price should have zero shortfall."""
        eqm = ExecutionQualityMonitor()
        rec = eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=100, filled_qty=100,
            decision_price=150.0, fill_price=150.0,
        )
        assert rec.implementation_shortfall_bps == pytest.approx(0.0, abs=0.01)


class TestFillRateTracking:
    """Validate fill rate computation and partial fill detection."""

    def test_full_fill_rate_one(self):
        """Full fill should have fill_rate = 1.0."""
        eqm = ExecutionQualityMonitor()
        rec = eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=1000, filled_qty=1000,
            decision_price=150.0, fill_price=150.0,
        )
        assert rec.fill_rate == 1.0

    def test_partial_fill_rate(self):
        """Partial fill should have fill_rate < 1.0."""
        eqm = ExecutionQualityMonitor()
        rec = eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=1000, filled_qty=750,
            decision_price=150.0, fill_price=150.0,
        )
        assert rec.fill_rate == pytest.approx(0.75, abs=0.001)

    def test_declining_fill_rates_detected_in_summary(self):
        """Summary should reflect aggregate degradation in fill rates."""
        eqm = ExecutionQualityMonitor()
        rng = np.random.default_rng(42)

        # Good fills first
        for i in range(20):
            eqm.record_execution(
                order_id=f"G{i}", ticker="AAPL", side="buy",
                target_qty=1000, filled_qty=int(1000 * rng.uniform(0.95, 1.0)),
                decision_price=150.0, fill_price=150.0 + rng.normal(0, 0.05),
            )

        good_summary = eqm.get_summary()

        # Bad fills
        for i in range(20):
            eqm.record_execution(
                order_id=f"B{i}", ticker="AAPL", side="buy",
                target_qty=1000, filled_qty=int(1000 * rng.uniform(0.5, 0.7)),
                decision_price=150.0, fill_price=150.0 + rng.uniform(0.5, 1.5),
            )

        combined_summary = eqm.get_summary()
        assert combined_summary.avg_fill_rate < good_summary.avg_fill_rate


class TestAdverseSelectionDetection:
    """Validate adverse selection flagging under different conditions."""

    def test_large_shortfall_flagged(self):
        """Orders with shortfall above threshold should be flagged."""
        eqm = ExecutionQualityMonitor(config={"adverse_threshold_bps": 20})
        eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=1000, filled_qty=1000,
            decision_price=100.0, fill_price=100.50,  # 50 bps
        )
        flagged = eqm.detect_adverse_selection()
        assert len(flagged) == 1

    def test_small_shortfall_not_flagged(self):
        """Orders with shortfall below threshold should not be flagged."""
        eqm = ExecutionQualityMonitor(config={"adverse_threshold_bps": 20})
        eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=1000, filled_qty=1000,
            decision_price=100.0, fill_price=100.10,  # 10 bps
        )
        flagged = eqm.detect_adverse_selection()
        assert len(flagged) == 0

    def test_adverse_count_in_summary(self):
        """Summary should report total adverse selection flags."""
        eqm = ExecutionQualityMonitor(config={"adverse_threshold_bps": 15})

        # 3 large shortfall orders
        for i in range(3):
            eqm.record_execution(
                order_id=f"BAD{i}", ticker="AAPL", side="buy",
                target_qty=1000, filled_qty=1000,
                decision_price=100.0, fill_price=100.50,  # 50 bps
            )
        # 2 normal orders
        for i in range(2):
            eqm.record_execution(
                order_id=f"OK{i}", ticker="AAPL", side="buy",
                target_qty=1000, filled_qty=1000,
                decision_price=100.0, fill_price=100.05,  # 5 bps
            )

        summary = eqm.get_summary()
        assert summary.total_adverse_selection_flags == 3


class TestSlippageVsVWAP:
    """Validate VWAP slippage computation."""

    def test_buy_above_vwap_positive_slippage(self):
        """Buying above VWAP should have positive slippage."""
        eqm = ExecutionQualityMonitor()
        rec = eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=100, filled_qty=100,
            decision_price=150.0, fill_price=151.0,
            vwap_benchmark=150.5,
        )
        assert rec.slippage_vs_vwap_bps > 0

    def test_sell_below_vwap_positive_slippage(self):
        """Selling below VWAP should have positive slippage."""
        eqm = ExecutionQualityMonitor()
        rec = eqm.record_execution(
            order_id="O1", ticker="AAPL", side="sell",
            target_qty=100, filled_qty=100,
            decision_price=150.0, fill_price=149.0,
            vwap_benchmark=149.5,
        )
        assert rec.slippage_vs_vwap_bps > 0

    def test_records_dataframe_has_expected_columns(self):
        """Records DataFrame should contain all key columns."""
        eqm = ExecutionQualityMonitor()
        eqm.record_execution(
            order_id="O1", ticker="AAPL", side="buy",
            target_qty=100, filled_qty=100,
            decision_price=150.0, fill_price=150.10,
        )
        df = eqm.get_records_dataframe()
        assert "implementation_shortfall_bps" in df.columns
        assert "slippage_vs_vwap_bps" in df.columns
        assert "fill_rate" in df.columns
        assert len(df) == 1
