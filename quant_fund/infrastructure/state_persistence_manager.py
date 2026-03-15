"""State persistence manager — wires runtime components to StateStore.

Provides save/restore for all critical runtime state that was previously
in-memory only:
    - Kill switch peak NAV and halt status
    - Drawdown monitor peak NAV
    - PnL dashboard history
    - System state machine state
    - Open orders tracking
    - Signal cache (latest alpha scores)
    - Portfolio positions snapshot

Periodic snapshots are written by the trading engine's main loop.
On restart, the manager restores all component state from the most
recent snapshot.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quant_fund.infrastructure.state_store import StateStore

logger = logging.getLogger(__name__)

# Namespace constants for StateStore
NS_KILL_SWITCH = "kill_switch"
NS_DRAWDOWN = "drawdown_monitor"
NS_PNL = "pnl_dashboard"
NS_STATE_MACHINE = "state_machine"
NS_OPEN_ORDERS = "open_orders"
NS_SIGNALS = "signal_cache"
NS_POSITIONS = "positions"
NS_SYSTEM = "system"


class StatePersistenceManager:
    """Coordinates state save/restore across all system components.

    Usage:
        store = StateStore("state.db")
        mgr = StatePersistenceManager(store)

        # Save state from components
        mgr.save_kill_switch_state(kill_switch)
        mgr.save_drawdown_state(drawdown_monitor)
        mgr.save_pnl_state(pnl_dashboard)
        mgr.save_positions(broker.get_positions())
        mgr.save_system_state(state_machine)
        mgr.save_signal_cache(alpha_scores)

        # Convenience: save everything at once
        mgr.snapshot_all(kill_switch=ks, drawdown_monitor=dm, ...)

        # Restore on restart
        mgr.restore_kill_switch_state(kill_switch)
        mgr.restore_drawdown_state(drawdown_monitor)
    """

    def __init__(self, store: StateStore):
        self._store = store
        self._snapshot_count = 0

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def save_kill_switch_state(self, kill_switch) -> None:
        """Save kill switch state: peak_nav, is_halted."""
        self._store.save(NS_KILL_SWITCH, "peak_nav", kill_switch.peak_nav)
        self._store.save(NS_KILL_SWITCH, "is_halted", kill_switch.is_halted)
        self._store.save(
            NS_KILL_SWITCH, "max_drawdown", kill_switch.max_drawdown
        )
        logger.debug(
            "Saved kill switch state: peak=%.2f halted=%s",
            kill_switch.peak_nav,
            kill_switch.is_halted,
        )

    def restore_kill_switch_state(self, kill_switch) -> bool:
        """Restore kill switch state. Returns True if state was found."""
        peak = self._store.load(NS_KILL_SWITCH, "peak_nav")
        halted = self._store.load(NS_KILL_SWITCH, "is_halted")

        if peak is None:
            logger.info("No kill switch state to restore")
            return False

        kill_switch.peak_nav = peak
        kill_switch._is_halted = halted or False
        logger.info(
            "Restored kill switch: peak=%.2f halted=%s", peak, halted
        )
        return True

    # ------------------------------------------------------------------
    # Drawdown monitor
    # ------------------------------------------------------------------

    def save_drawdown_state(self, drawdown_monitor) -> None:
        """Save drawdown monitor peak NAV."""
        self._store.save(NS_DRAWDOWN, "peak_nav", drawdown_monitor.peak_nav)
        self._store.save(
            NS_DRAWDOWN,
            "current_drawdown",
            drawdown_monitor.current_drawdown,
        )

    def restore_drawdown_state(self, drawdown_monitor) -> bool:
        """Restore drawdown monitor state."""
        peak = self._store.load(NS_DRAWDOWN, "peak_nav")
        if peak is None:
            return False
        drawdown_monitor._peak_nav = peak
        logger.info("Restored drawdown monitor: peak=%.2f", peak)
        return True

    # ------------------------------------------------------------------
    # PnL dashboard
    # ------------------------------------------------------------------

    def save_pnl_state(self, pnl_dashboard) -> None:
        """Save PnL dashboard summary state."""
        self._store.save(NS_PNL, "peak_nav", pnl_dashboard._peak_nav)
        self._store.save(NS_PNL, "prev_nav", pnl_dashboard._prev_nav)
        self._store.save(NS_PNL, "initial_nav", pnl_dashboard._initial_nav)
        self._store.save(
            NS_PNL,
            "cumulative_realised",
            pnl_dashboard._cumulative_realised,
        )

        # Save last N snapshots as serializable dicts
        history_dicts = []
        for snap in pnl_dashboard._history[-100:]:
            history_dicts.append({
                "timestamp": str(snap.timestamp),
                "nav": snap.nav,
                "daily_return": snap.daily_return,
                "daily_pnl": snap.daily_pnl,
                "cumulative_return": snap.cumulative_return,
                "drawdown": snap.drawdown,
                "peak_nav": snap.peak_nav,
                "realised_pnl": snap.realised_pnl,
                "unrealised_pnl": snap.unrealised_pnl,
            })
        self._store.save(NS_PNL, "history", history_dicts)

    def restore_pnl_state(self, pnl_dashboard) -> bool:
        """Restore PnL dashboard state."""
        peak = self._store.load(NS_PNL, "peak_nav")
        if peak is None:
            return False

        pnl_dashboard._peak_nav = peak
        pnl_dashboard._prev_nav = self._store.load(
            NS_PNL, "prev_nav", pnl_dashboard._initial_nav
        )
        pnl_dashboard._cumulative_realised = self._store.load(
            NS_PNL, "cumulative_realised", 0.0
        )
        logger.info("Restored PnL dashboard: peak=%.2f", peak)
        return True

    # ------------------------------------------------------------------
    # System state machine
    # ------------------------------------------------------------------

    def save_system_state(self, state_machine) -> None:
        """Save system state machine state."""
        self._store.save(NS_STATE_MACHINE, "state", state_machine.to_dict())

    def restore_system_state(self, state_machine) -> bool:
        """Restore system state machine."""
        data = self._store.load(NS_STATE_MACHINE, "state")
        if data is None:
            return False
        state_machine.restore_from_dict(data)
        logger.info("Restored state machine")
        return True

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def save_positions(self, positions: pd.Series) -> None:
        """Save current positions snapshot."""
        pos_dict = {
            str(k): float(v) for k, v in positions.items() if v != 0
        }
        self._store.save(NS_POSITIONS, "current", pos_dict)
        self._store.save(
            NS_POSITIONS,
            "snapshot_time",
            pd.Timestamp.now().isoformat(),
        )

    def restore_positions(self) -> Optional[pd.Series]:
        """Restore positions snapshot. Returns None if not found."""
        pos_dict = self._store.load(NS_POSITIONS, "current")
        if pos_dict is None:
            return None
        return pd.Series(pos_dict, dtype=float)

    # ------------------------------------------------------------------
    # Signal cache
    # ------------------------------------------------------------------

    def save_signal_cache(self, alpha_scores: pd.Series) -> None:
        """Save latest alpha scores."""
        scores_dict = {
            str(k): float(v)
            for k, v in alpha_scores.items()
            if pd.notna(v)
        }
        self._store.save(NS_SIGNALS, "alpha_scores", scores_dict)
        self._store.save(
            NS_SIGNALS,
            "timestamp",
            pd.Timestamp.now().isoformat(),
        )

    def restore_signal_cache(self) -> Optional[pd.Series]:
        """Restore cached alpha scores."""
        data = self._store.load(NS_SIGNALS, "alpha_scores")
        if data is None:
            return None
        return pd.Series(data, dtype=float)

    # ------------------------------------------------------------------
    # Open orders
    # ------------------------------------------------------------------

    def save_open_orders(self, order_ids: List[str]) -> None:
        """Save list of open order IDs for duplicate prevention on restart."""
        self._store.save(NS_OPEN_ORDERS, "ids", order_ids)
        self._store.save(
            NS_OPEN_ORDERS,
            "timestamp",
            pd.Timestamp.now().isoformat(),
        )

    def restore_open_orders(self) -> List[str]:
        """Restore list of open order IDs."""
        ids = self._store.load(NS_OPEN_ORDERS, "ids", [])
        return ids

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def snapshot_all(
        self,
        kill_switch=None,
        drawdown_monitor=None,
        pnl_dashboard=None,
        state_machine=None,
        positions: Optional[pd.Series] = None,
        alpha_scores: Optional[pd.Series] = None,
        open_order_ids: Optional[List[str]] = None,
    ) -> None:
        """Save state snapshots for all provided components."""
        if kill_switch is not None:
            self.save_kill_switch_state(kill_switch)
        if drawdown_monitor is not None:
            self.save_drawdown_state(drawdown_monitor)
        if pnl_dashboard is not None:
            self.save_pnl_state(pnl_dashboard)
        if state_machine is not None:
            self.save_system_state(state_machine)
        if positions is not None:
            self.save_positions(positions)
        if alpha_scores is not None:
            self.save_signal_cache(alpha_scores)
        if open_order_ids is not None:
            self.save_open_orders(open_order_ids)

        self._snapshot_count += 1
        self._store.save(
            NS_SYSTEM,
            "last_snapshot",
            {
                "count": self._snapshot_count,
                "timestamp": pd.Timestamp.now().isoformat(),
            },
        )
        logger.debug("State snapshot #%d completed", self._snapshot_count)

    def restore_all(
        self,
        kill_switch=None,
        drawdown_monitor=None,
        pnl_dashboard=None,
        state_machine=None,
    ) -> Dict[str, bool]:
        """Restore state for all provided components.

        Returns dict mapping component name → whether restore succeeded.
        """
        results = {}
        if kill_switch is not None:
            results["kill_switch"] = self.restore_kill_switch_state(kill_switch)
        if drawdown_monitor is not None:
            results["drawdown_monitor"] = self.restore_drawdown_state(
                drawdown_monitor
            )
        if pnl_dashboard is not None:
            results["pnl_dashboard"] = self.restore_pnl_state(pnl_dashboard)
        if state_machine is not None:
            results["state_machine"] = self.restore_system_state(state_machine)

        logger.info("Restore results: %s", results)
        return results

    @property
    def snapshot_count(self) -> int:
        return self._snapshot_count
