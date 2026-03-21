"""Test position and NAV reconciliation drift detection.

Validates ReconciliationEngine detects position mismatches,
NAV discrepancies, and order tracking inconsistencies between
internal state and broker state.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.reconciliation_engine import (
    ReconciliationEngine,
    ReconciliationResult,
)
from tests.conftest import STANDARD_MARKET_DATA


pytestmark = [pytest.mark.validation]


def _make_broker(initial_cash=1_000_000.0):
    """Create a SimulationBroker with standard market data."""
    broker = SimulationBroker({
        "initial_cash": initial_cash,
        "enforce_cash_floor": True,
    })
    broker.set_market_data(STANDARD_MARKET_DATA)
    return broker


class TestPositionReconciliation:
    """Validate position discrepancy detection."""

    def test_matched_positions_no_discrepancy(self):
        """Identical internal and broker positions should match."""
        broker = _make_broker()
        engine = ReconciliationEngine(broker)

        # Both internal and broker have same positions (empty)
        internal = broker.get_positions()
        result = engine.reconcile(internal)
        assert result.positions_matched is True
        assert len(result.position_discrepancies) == 0

    def test_position_mismatch_detected(self):
        """Different internal and broker positions should produce discrepancy."""
        broker = _make_broker()
        engine = ReconciliationEngine(broker, config={"position_tolerance_shares": 0})

        # Internal thinks we have 100 AAPL, broker has 0
        internal = pd.Series({"AAPL": 100.0})
        result = engine.reconcile(internal)

        assert result.positions_matched is False
        assert len(result.position_discrepancies) >= 1
        aapl_disc = [d for d in result.position_discrepancies if d.ticker == "AAPL"]
        assert len(aapl_disc) == 1
        assert aapl_disc[0].internal_qty == 100.0
        assert aapl_disc[0].broker_qty == 0.0

    def test_auto_correct_resolves_discrepancy(self):
        """Auto-correction should mark discrepancies as resolved."""
        broker = _make_broker()
        engine = ReconciliationEngine(broker, config={"auto_correct": True})

        internal = pd.Series({"AAPL": 50.0})
        result = engine.reconcile(internal)

        if result.position_discrepancies:
            assert result.auto_corrected is True
            assert all(d.resolved for d in result.position_discrepancies)

    def test_tolerance_skips_small_differences(self):
        """Differences within tolerance should not be flagged."""
        broker = _make_broker()
        engine = ReconciliationEngine(
            broker, config={"position_tolerance_shares": 10}
        )

        internal = pd.Series({"AAPL": 5.0})
        result = engine.reconcile(internal)

        # 5 shares is within 10-share tolerance
        assert result.positions_matched is True


class TestNAVReconciliation:
    """Validate NAV/cash discrepancy detection."""

    def test_nav_within_tolerance_matches(self):
        """NAV within tolerance should be matched."""
        broker = _make_broker(initial_cash=1_000_000.0)
        engine = ReconciliationEngine(
            broker, config={"nav_tolerance_pct": 0.01}
        )

        # Broker has 1M, internal says ~1M (within 1%)
        broker_nav = broker.get_account_value()
        internal_nav = broker_nav * 0.999  # 0.1% off
        result = engine.reconcile(pd.Series(dtype=float), internal_nav=internal_nav)

        assert result.cash_matched is True

    def test_nav_outside_tolerance_flags(self):
        """NAV outside tolerance should be flagged."""
        broker = _make_broker(initial_cash=1_000_000.0)
        engine = ReconciliationEngine(
            broker, config={"nav_tolerance_pct": 0.01}
        )

        broker_nav = broker.get_account_value()
        internal_nav = broker_nav * 0.95  # 5% off
        result = engine.reconcile(pd.Series(dtype=float), internal_nav=internal_nav)

        assert result.cash_matched is False
        assert result.cash_discrepancy is not None
        assert result.cash_discrepancy.delta_pct > 0.01


class TestReconciliationMetrics:
    """Validate reconciliation history and metrics tracking."""

    def test_history_accumulates(self):
        """Each reconcile call should append to history."""
        broker = _make_broker()
        engine = ReconciliationEngine(broker)

        for _ in range(5):
            engine.reconcile(pd.Series(dtype=float))

        history = engine.get_history()
        assert len(history) == 5

    def test_metrics_track_mismatch_rate(self):
        """Metrics should track mismatch rate across reconciliation cycles."""
        broker = _make_broker()
        engine = ReconciliationEngine(
            broker, config={"position_tolerance_shares": 0}
        )

        # 3 matched reconciliations
        for _ in range(3):
            engine.reconcile(broker.get_positions())

        # 2 mismatched reconciliations
        for _ in range(2):
            engine.reconcile(pd.Series({"AAPL": 999.0}))

        metrics = engine.get_metrics()
        assert metrics["total_reconciliations"] == 5
        assert metrics["total_mismatches"] == 2
        assert metrics["mismatch_rate"] == pytest.approx(0.4, abs=0.01)

    def test_correction_count_increments(self):
        """Auto-corrections should increment correction_count."""
        broker = _make_broker()
        engine = ReconciliationEngine(
            broker,
            config={"auto_correct": True, "position_tolerance_shares": 0},
        )

        engine.reconcile(pd.Series({"AAPL": 100.0}))
        engine.reconcile(pd.Series({"MSFT": 200.0}))

        assert engine.correction_count >= 2

    def test_latest_returns_most_recent(self):
        """latest property should return the most recent result."""
        broker = _make_broker()
        engine = ReconciliationEngine(broker)

        engine.reconcile(pd.Series(dtype=float))
        engine.reconcile(pd.Series(dtype=float))

        assert engine.latest is not None
        assert isinstance(engine.latest, ReconciliationResult)

    def test_all_matched_property(self):
        """all_matched should be True only when all checks pass."""
        broker = _make_broker()
        engine = ReconciliationEngine(broker)

        result = engine.reconcile(broker.get_positions())
        assert result.all_matched is True

    def test_result_to_dict_serializable(self):
        """ReconciliationResult.to_dict should produce a JSON-serializable dict."""
        broker = _make_broker()
        engine = ReconciliationEngine(
            broker, config={"position_tolerance_shares": 0}
        )

        result = engine.reconcile(pd.Series({"AAPL": 50.0}))
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "all_matched" in d
        assert "position_discrepancies" in d
        assert isinstance(d["position_discrepancies"], list)
