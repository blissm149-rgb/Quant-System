"""Portfolio kill switch — hard stop on drawdown breach.

Simplest module in the system. Must remain simple. No dependencies on
any other module except position data and drawdown calculation.
"""

import logging
from typing import Optional

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
