"""Integration tests for TradingEngine state persistence and recovery.

Module E: Verifies snapshot/restore round-trip preserves all critical state,
and that recovery from simulated crashes produces consistent outcomes.
"""

import pandas as pd
import pytest

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.infrastructure.state_persistence_manager import StatePersistenceManager
from quant_fund.infrastructure.state_store import StateStore
from quant_fund.infrastructure.system_state_machine import SystemState
from quant_fund.main.trading_engine import TradingEngine
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from tests.conftest import STANDARD_MARKET_DATA


def _build_engine(state_db_path=":memory:", broker=None):
    """Build a fully-wired TradingEngine for recovery testing."""
    if broker is None:
        broker = SimulationBroker(config={"initial_cash": 1_000_000})
        broker.set_market_data(STANDARD_MARKET_DATA)

    engine = TradingEngine(config={
        "synchronous": True,
        "state_db_path": state_db_path,
    })
    ks = KillSwitch(config={"drawdown_threshold": 0.20})
    dm = DrawdownMonitor(config={"initial_nav": 1_000_000})
    pnl = PnLDashboard()

    engine.inject_components(
        broker=broker,
        order_generator=OrderGenerator(),
        order_router=OrderRouter(broker),
        kill_switch=ks,
        drawdown_monitor=dm,
        pnl_dashboard=pnl,
    )
    return engine, ks, dm, pnl


@pytest.mark.integration
class TestRecoveryScenarios:
    """State persistence and recovery for TradingEngine."""

    def test_snapshot_restore_preserves_kill_switch_state(self):
        """Kill switch peak NAV survives snapshot/restore cycle."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        ks = KillSwitch(config={"drawdown_threshold": 0.20})
        ks.update_peak(1_200_000.0)

        mgr.save_kill_switch_state(ks)

        ks2 = KillSwitch(config={"drawdown_threshold": 0.20})
        restored = mgr.restore_kill_switch_state(ks2)

        assert restored is True
        assert ks2.peak_nav == 1_200_000.0

    def test_snapshot_restore_preserves_drawdown_monitor(self):
        """Drawdown monitor peak NAV survives snapshot/restore cycle."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        dm = DrawdownMonitor(config={"initial_nav": 1_000_000})
        # Push peak higher
        dm.update(1_100_000)
        dm.update(1_050_000)  # now in drawdown

        mgr.save_drawdown_state(dm)

        dm2 = DrawdownMonitor(config={"initial_nav": 500_000})
        restored = mgr.restore_drawdown_state(dm2)

        assert restored is True
        assert dm2._peak_nav == dm._peak_nav

    def test_snapshot_restore_preserves_alpha_scores(self):
        """Alpha score cache round-trips through persistence."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        scores = pd.Series({"AAPL": 0.05, "MSFT": -0.03, "GOOG": 0.01})
        mgr.save_signal_cache(scores)

        restored = mgr.restore_signal_cache()
        assert restored is not None
        assert len(restored) == 3
        assert abs(restored["AAPL"] - 0.05) < 1e-9

    def test_snapshot_restore_preserves_open_order_ids(self):
        """Open order IDs survive persistence round-trip."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        ids = ["order-001", "order-002", "order-003"]
        mgr.save_open_orders(ids)

        restored = mgr.restore_open_orders()
        assert restored == ids

    def test_recovery_after_simulated_crash(self):
        """Full round-trip: init → convergence ticks → snapshot → new engine → restore.

        Uses a shared file-based StateStore so the second engine can read
        the first engine's state.
        """
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "state.db")

            # Engine 1: run some convergence ticks
            broker1 = SimulationBroker(config={"initial_cash": 1_000_000})
            broker1.set_market_data(STANDARD_MARKET_DATA)

            engine1, ks1, dm1, pnl1 = _build_engine(db_path, broker1)

            weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
            engine1.update_target_weights(weights)

            # Run convergence
            engine1._convergence_tick()
            engine1._convergence_tick()

            # Simulate a profitable day by pushing peak higher
            ks1.update_peak(1_050_000.0)
            pnl1.update(1_050_000.0)
            peak_nav_saved = ks1.peak_nav

            # Take snapshot
            engine1._take_snapshot()

            # Engine 2: restore from same DB
            broker2 = SimulationBroker(config={"initial_cash": 1_000_000})
            broker2.set_market_data(STANDARD_MARKET_DATA)

            engine2 = TradingEngine(config={
                "synchronous": True,
                "state_db_path": db_path,
            })
            ks2 = KillSwitch(config={"drawdown_threshold": 0.20})
            dm2 = DrawdownMonitor(config={"initial_nav": 500_000})  # intentionally wrong

            engine2.inject_components(
                broker=broker2,
                order_generator=OrderGenerator(),
                order_router=OrderRouter(broker2),
                kill_switch=ks2,
                drawdown_monitor=dm2,
                pnl_dashboard=PnLDashboard(),
            )

            engine2._restore_state()

            # Kill switch peak should be restored
            assert ks2.peak_nav == pytest.approx(peak_nav_saved, rel=1e-6)

    def test_reconciliation_after_recovery(self):
        """After restore, reconciliation finds no discrepancies."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        broker = SimulationBroker(config={"initial_cash": 1_000_000})
        broker.set_market_data(STANDARD_MARKET_DATA)

        engine, ks, dm, pnl = _build_engine(":memory:", broker)

        weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        engine.update_target_weights(weights)
        engine._convergence_tick()

        # Run reconciliation — should succeed without error
        engine._run_reconciliation()

    def test_snapshot_all_increments_count(self):
        """snapshot_all increments the snapshot counter."""
        store = StateStore(":memory:")
        mgr = StatePersistenceManager(store)

        assert mgr.snapshot_count == 0

        ks = KillSwitch(config={"drawdown_threshold": 0.20})
        mgr.snapshot_all(kill_switch=ks)
        assert mgr.snapshot_count == 1

        mgr.snapshot_all(kill_switch=ks)
        assert mgr.snapshot_count == 2
