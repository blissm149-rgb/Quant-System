"""Unit tests for reconciliation engine.

TESTING_PLAN.md Section 3.10 — reconciliation_engine.
"""

import pandas as pd
import pytest

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.reconciliation_engine import (
    ReconciliationEngine,
    ReconciliationResult,
)
from tests.conftest import STANDARD_MARKET_DATA


@pytest.mark.unit
@pytest.mark.tier2
class TestReconciliationEngine:
    """ReconciliationEngine — position and NAV reconciliation."""

    @pytest.fixture
    def broker(self):
        b = SimulationBroker(config={"initial_cash": 1_000_000})
        b.set_market_data(STANDARD_MARKET_DATA)
        return b

    @pytest.fixture
    def engine(self, broker):
        return ReconciliationEngine(broker)

    def test_reconcile_matching_positions(self, engine, broker):
        """Matching positions produce no discrepancies."""
        positions = broker.get_positions()
        result = engine.reconcile(positions, internal_nav=broker.get_account_value())
        assert isinstance(result, ReconciliationResult)
        assert result.all_matched is True

    def test_reconcile_detects_position_mismatch(self, engine, broker):
        """Mismatched positions are detected."""
        fake_internal = pd.Series({"AAPL": 999})
        result = engine.reconcile(fake_internal, internal_nav=broker.get_account_value())
        assert result.positions_matched is False
        assert len(result.position_discrepancies) > 0

    def test_reconcile_returns_result_with_nav(self, engine, broker):
        """Reconciliation with NAV returns a valid result."""
        positions = broker.get_positions()
        nav = broker.get_account_value()
        result = engine.reconcile(positions, internal_nav=nav)
        assert isinstance(result, ReconciliationResult)

    def test_get_corrected_positions(self, engine, broker):
        """get_corrected_positions adjusts to broker truth."""
        fake_internal = pd.Series({"AAPL": 999})
        result = engine.reconcile(fake_internal, internal_nav=broker.get_account_value())
        corrected = engine.get_corrected_positions(fake_internal, result)
        assert isinstance(corrected, pd.Series)

    def test_history_tracking(self, engine, broker):
        """Reconciliation results are stored in history."""
        positions = broker.get_positions()
        engine.reconcile(positions, internal_nav=broker.get_account_value())
        engine.reconcile(positions, internal_nav=broker.get_account_value())
        history = engine.get_history()
        assert len(history) >= 2

    def test_metrics(self, engine, broker):
        """get_metrics returns dict with reconciliation stats."""
        positions = broker.get_positions()
        engine.reconcile(positions, internal_nav=broker.get_account_value())
        metrics = engine.get_metrics()
        assert isinstance(metrics, dict)
