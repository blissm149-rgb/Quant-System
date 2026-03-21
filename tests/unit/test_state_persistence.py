"""Unit tests for state persistence manager.

TESTING_PLAN.md Section 3.12 — state_persistence_manager.
"""

import pandas as pd
import pytest

from quant_fund.infrastructure.state_persistence_manager import (
    StatePersistenceManager,
)
from quant_fund.infrastructure.state_store import StateStore


@pytest.mark.unit
@pytest.mark.tier2
class TestStatePersistenceManager:
    """StatePersistenceManager — namespaced state save/restore."""

    @pytest.fixture
    def store(self):
        return StateStore()

    @pytest.fixture
    def manager(self, store):
        return StatePersistenceManager(store)

    def test_save_and_restore_positions(self, manager):
        """Positions roundtrip through save/restore."""
        positions = pd.Series({"AAPL": 100.0, "MSFT": -50.0})
        manager.save_positions(positions)
        restored = manager.restore_positions()
        assert restored is not None
        pd.testing.assert_series_equal(restored, positions)

    def test_save_and_restore_signal_cache(self, manager):
        """Signal cache roundtrip."""
        signals = pd.Series({"AAPL": 0.85, "MSFT": -0.42})
        manager.save_signal_cache(signals)
        restored = manager.restore_signal_cache()
        assert restored is not None
        pd.testing.assert_series_equal(restored, signals)

    def test_save_and_restore_open_orders(self, manager):
        """Open order IDs roundtrip."""
        order_ids = ["order_1", "order_2", "order_3"]
        manager.save_open_orders(order_ids)
        restored = manager.restore_open_orders()
        assert restored == order_ids

    def test_restore_positions_when_empty(self, manager):
        """Restore returns None when no positions saved."""
        result = manager.restore_positions()
        assert result is None

    def test_snapshot_all(self, manager):
        """snapshot_all saves multiple namespaces."""
        positions = pd.Series({"AAPL": 100.0})
        signals = pd.Series({"AAPL": 0.5})
        manager.snapshot_all(
            positions=positions,
            alpha_scores=signals,
            open_order_ids=["o1"],
        )
        assert manager.restore_positions() is not None
        assert manager.restore_signal_cache() is not None
        assert manager.restore_open_orders() == ["o1"]

    def test_snapshot_count(self, manager):
        """snapshot_count tracks number of snapshots."""
        manager.snapshot_all(positions=pd.Series({"AAPL": 1.0}))
        assert manager.snapshot_count >= 1
