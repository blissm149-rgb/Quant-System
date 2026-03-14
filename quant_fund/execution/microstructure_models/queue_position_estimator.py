"""Queue position estimator.

Estimates the queue position of a limit order and expected
waiting time based on order book depth and arrival rates.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QueuePositionEstimate:
    """Estimated queue position and waiting time."""

    ticker: str
    estimated_position: int  # position in queue (1 = front)
    queue_depth: float  # total shares ahead
    estimated_wait_s: float  # estimated seconds to reach front
    fill_before_cancel_prob: float  # P(fill before cancel)


class QueuePositionEstimator:
    """Estimates queue position for limit orders.

    Uses order book depth at the limit price level and
    historical fill rate to estimate waiting time and
    probability of execution.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._default_fill_rate = cfg.get("default_fill_rate_per_s", 10.0)
        self._cancel_rate = cfg.get("cancel_rate_per_s", 0.01)

    def estimate(
        self,
        ticker: str,
        order_size: float,
        depth_ahead: float,
        fill_rate_per_s: Optional[float] = None,
    ) -> QueuePositionEstimate:
        """Estimate queue position and expected waiting time.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        order_size : float
            Size of our order in shares.
        depth_ahead : float
            Total shares queued ahead of our order.
        fill_rate_per_s : float, optional
            Estimated shares filled per second at this price level.
            Defaults to configured default.

        Returns
        -------
        QueuePositionEstimate
        """
        rate = fill_rate_per_s if fill_rate_per_s is not None else self._default_fill_rate

        if rate <= 0:
            return QueuePositionEstimate(
                ticker=ticker,
                estimated_position=int(depth_ahead) + 1,
                queue_depth=depth_ahead,
                estimated_wait_s=float("inf"),
                fill_before_cancel_prob=0.0,
            )

        # Time to clear queue ahead of us
        time_to_front = depth_ahead / rate
        # Time to fill our order once at front
        time_to_fill = order_size / rate
        total_wait = time_to_front + time_to_fill

        # P(fill before cancel) using exponential cancel model
        # Survival probability: exp(-cancel_rate * wait_time)
        fill_prob = float(np.exp(-self._cancel_rate * total_wait))
        fill_prob = np.clip(fill_prob, 0.0, 1.0)

        # Approximate position (1 = front of queue)
        position = max(1, int(depth_ahead / max(order_size, 1)) + 1)

        return QueuePositionEstimate(
            ticker=ticker,
            estimated_position=position,
            queue_depth=depth_ahead,
            estimated_wait_s=total_wait,
            fill_before_cancel_prob=float(fill_prob),
        )

    def estimate_from_book(
        self,
        ticker: str,
        order_size: float,
        limit_price: float,
        book_levels: list,
        fill_rate_per_s: Optional[float] = None,
    ) -> QueuePositionEstimate:
        """Estimate queue position from order book levels.

        Parameters
        ----------
        book_levels : list
            List of (price, size) tuples sorted best to worst.
        """
        depth_ahead = 0.0
        for price, size in book_levels:
            if price == limit_price:
                # Assume we're at the back of this level
                depth_ahead += size
                break
            depth_ahead += size

        return self.estimate(
            ticker=ticker,
            order_size=order_size,
            depth_ahead=depth_ahead,
            fill_rate_per_s=fill_rate_per_s,
        )
