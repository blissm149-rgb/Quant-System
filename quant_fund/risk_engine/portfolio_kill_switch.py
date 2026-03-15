"""Portfolio kill switch — hard stop on drawdown breach.

Provides two kill switch classes:
- KillSwitch: portfolio-level hard stop on drawdown breach.
- StrategyKillSwitch: per-strategy kill switch with auto-liquidation support.

Simplest module in the system. Must remain simple. No dependencies on
any other module except position data, drawdown calculation, and
broker order types.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderType,
)

logger = logging.getLogger(__name__)


class KillSwitch:
    """Hard stop triggered when drawdown exceeds configured limit.

    Called before every order batch. If triggered, order_generator produces
    no new orders. Existing positions are not automatically liquidated
    (manual decision required).
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.max_drawdown: float = cfg.get("drawdown_limit", 0.20)
        self.peak_nav: float = cfg.get("initial_nav", 1_000_000.0)
        self._is_halted: bool = False

    def check(self, current_nav: float) -> bool:
        """Return True if drawdown limit breached.

        Called before every order batch.
        """
        if self.peak_nav <= 0:
            return False
        drawdown = (self.peak_nav - current_nav) / self.peak_nav
        if drawdown >= self.max_drawdown:
            self._is_halted = True
            logger.critical(
                "Kill switch triggered: drawdown %.2f%% >= %.2f%%",
                drawdown * 100,
                self.max_drawdown * 100,
            )
            return True
        return False

    def update_peak(self, current_nav: float) -> None:
        """Call after every NAV update when not in halt state."""
        if not self._is_halted:
            self.peak_nav = max(self.peak_nav, current_nav)

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    def reset(self, new_peak_nav: float) -> None:
        """Manual resume only. Reset halt state and peak NAV."""
        self._is_halted = False
        self.peak_nav = new_peak_nav
        logger.info("Kill switch reset. New peak NAV: %.2f", new_peak_nav)


class StrategyKillSwitch:
    """Per-strategy kill switch with auto-liquidation support.

    Monitors drawdown per strategy_id independently. When triggered:
    1. Halts the strategy
    2. Generates liquidation orders if order_generator is provided
    3. Records trigger timestamp and drawdown level
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.max_drawdown: float = cfg.get("drawdown_limit", 0.20)
        self._strategies: Dict[str, dict] = {}

    def register_strategy(self, strategy_id: str, initial_nav: float) -> None:
        """Register a strategy for monitoring."""
        self._strategies[strategy_id] = {
            "peak_nav": initial_nav,
            "is_halted": False,
            "trigger_time": None,
            "trigger_drawdown": None,
        }
        logger.info(
            "Strategy '%s' registered with initial NAV: %.2f",
            strategy_id,
            initial_nav,
        )

    def check(self, strategy_id: str, current_nav: float) -> bool:
        """Check drawdown for a specific strategy. Updates peak if not halted.

        Returns True if the drawdown limit has been breached.
        """
        state = self._strategies[strategy_id]
        if state["is_halted"]:
            return True
        peak = state["peak_nav"]
        if peak <= 0:
            return False
        # Update peak before checking
        if current_nav > peak:
            state["peak_nav"] = current_nav
            peak = current_nav
        drawdown = (peak - current_nav) / peak
        if drawdown >= self.max_drawdown:
            state["is_halted"] = True
            state["trigger_time"] = datetime.now(timezone.utc)
            state["trigger_drawdown"] = drawdown
            logger.critical(
                "Strategy '%s' kill switch triggered: drawdown %.2f%% >= %.2f%%",
                strategy_id,
                drawdown * 100,
                self.max_drawdown * 100,
            )
            return True
        return False

    def manual_halt(self, strategy_id: str, reason: str = "") -> None:
        """Manually halt a strategy from outside the trading loop."""
        state = self._strategies[strategy_id]
        state["is_halted"] = True
        state["trigger_time"] = datetime.now(timezone.utc)
        state["trigger_drawdown"] = None
        logger.warning(
            "Strategy '%s' manually halted. Reason: %s",
            strategy_id,
            reason or "not specified",
        )

    def generate_liquidation_orders(
        self,
        strategy_id: str,
        positions: pd.Series,
        prices: pd.Series,
    ) -> List[Order]:
        """Generate market SELL orders to flatten all positions.

        Args:
            strategy_id: The strategy to liquidate.
            positions: Series mapping ticker -> signed quantity (positive=long).
            prices: Series mapping ticker -> current price (unused for market
                orders but recorded for audit).

        Returns:
            List of Order objects that will flatten each position.
        """
        orders: List[Order] = []
        for ticker, qty in positions.items():
            if qty == 0:
                continue
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            orders.append(
                Order(
                    ticker=str(ticker),
                    side=side,
                    quantity=abs(int(qty)),
                    order_type=OrderType.MARKET,
                    strategy_id=strategy_id,
                    timestamp=pd.Timestamp.now(tz="UTC"),
                )
            )
        logger.info(
            "Generated %d liquidation orders for strategy '%s'",
            len(orders),
            strategy_id,
        )
        return orders

    def is_halted(self, strategy_id: str) -> bool:
        """Check if a specific strategy is halted."""
        return self._strategies[strategy_id]["is_halted"]

    def reset(self, strategy_id: str, new_peak_nav: float) -> None:
        """Manual resume only. Reset halt state and peak NAV."""
        state = self._strategies[strategy_id]
        state["is_halted"] = False
        state["peak_nav"] = new_peak_nav
        state["trigger_time"] = None
        state["trigger_drawdown"] = None
        logger.info(
            "Strategy '%s' kill switch reset. New peak NAV: %.2f",
            strategy_id,
            new_peak_nav,
        )

    def get_status(self, strategy_id: str) -> dict:
        """Returns status dict with peak_nav, is_halted, trigger_time,
        trigger_drawdown, and current_drawdown."""
        state = self._strategies[strategy_id]
        return {
            "peak_nav": state["peak_nav"],
            "is_halted": state["is_halted"],
            "trigger_time": state["trigger_time"],
            "trigger_drawdown": state["trigger_drawdown"],
            "current_drawdown": state.get("trigger_drawdown"),
        }

    @property
    def halted_strategies(self) -> List[str]:
        """Return list of halted strategy IDs."""
        return [
            sid
            for sid, state in self._strategies.items()
            if state["is_halted"]
        ]
