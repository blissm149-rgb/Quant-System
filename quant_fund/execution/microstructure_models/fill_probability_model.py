"""Fill probability model.

Estimates the probability that a limit order will be filled
given its distance from the current mid, order book conditions,
and historical fill rates.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FillProbability:
    """Fill probability estimate for a limit order."""

    ticker: str
    fill_prob: float  # probability of full fill [0, 1]
    partial_fill_prob: float  # probability of at least partial fill
    expected_fill_pct: float  # expected fraction filled [0, 1]
    expected_time_to_fill_s: float  # expected seconds to fill
    distance_bps: float  # distance from mid in bps


class FillProbabilityModel:
    """Estimates fill probability for limit orders.

    Uses a logistic model calibrated on distance from mid,
    spread width, and order size relative to available depth.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        # Logistic model parameters: P(fill) = 1 / (1 + exp(a * distance + b))
        self._logistic_a = cfg.get("logistic_a", 0.5)
        self._logistic_b = cfg.get("logistic_b", -1.0)
        self._size_penalty_factor = cfg.get("size_penalty_factor", 0.3)
        self._base_fill_time_s = cfg.get("base_fill_time_s", 60.0)

    def estimate_fill_probability(
        self,
        ticker: str,
        limit_price: float,
        mid_price: float,
        order_size: float,
        side: str,
        spread_bps: float = 10.0,
        available_depth: float = 0.0,
    ) -> FillProbability:
        """Estimate fill probability for a limit order.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        limit_price : float
            Limit order price.
        mid_price : float
            Current mid price.
        order_size : float
            Order size in shares.
        side : str
            'buy' or 'sell'.
        spread_bps : float
            Current quoted spread in bps.
        available_depth : float
            Available depth at or better than limit price.

        Returns
        -------
        FillProbability
        """
        if mid_price <= 0:
            return FillProbability(
                ticker=ticker,
                fill_prob=0.0,
                partial_fill_prob=0.0,
                expected_fill_pct=0.0,
                expected_time_to_fill_s=float("inf"),
                distance_bps=0.0,
            )

        # Distance from mid in bps (positive = passive/behind mid)
        if side == "buy":
            distance_bps = (mid_price - limit_price) / mid_price * 10000.0
        else:
            distance_bps = (limit_price - mid_price) / mid_price * 10000.0

        # Aggressive orders (crossing the spread) get high fill prob
        if distance_bps < 0:
            # Marketable limit order
            fill_prob = min(0.99, 0.95 + abs(distance_bps) / 1000.0)
        else:
            # Passive limit order: logistic decay with distance
            z = self._logistic_a * distance_bps + self._logistic_b
            fill_prob = 1.0 / (1.0 + np.exp(z))

        # Size penalty: larger orders relative to depth are harder to fill
        if available_depth > 0 and order_size > 0:
            size_ratio = order_size / available_depth
            size_penalty = np.exp(-self._size_penalty_factor * size_ratio)
            fill_prob *= size_penalty

        fill_prob = float(np.clip(fill_prob, 0.0, 0.99))
        partial_fill_prob = min(0.99, fill_prob + (1.0 - fill_prob) * 0.3)
        expected_fill_pct = fill_prob * 0.9 + partial_fill_prob * 0.1

        # Expected time to fill: inversely proportional to fill probability
        if fill_prob > 0.01:
            expected_time = self._base_fill_time_s / fill_prob
        else:
            expected_time = float("inf")

        return FillProbability(
            ticker=ticker,
            fill_prob=fill_prob,
            partial_fill_prob=partial_fill_prob,
            expected_fill_pct=expected_fill_pct,
            expected_time_to_fill_s=expected_time,
            distance_bps=distance_bps,
        )

    def optimal_limit_offset(
        self,
        spread_bps: float,
        urgency: float = 0.5,
    ) -> float:
        """Compute optimal limit price offset from mid in bps.

        Parameters
        ----------
        spread_bps : float
            Current quoted spread in bps.
        urgency : float
            Urgency parameter in [0, 1]. 0 = patient, 1 = aggressive.

        Returns
        -------
        float
            Recommended offset from mid in bps (positive = behind mid).
        """
        urgency = float(np.clip(urgency, 0.0, 1.0))
        # At urgency=1, cross the spread (negative offset)
        # At urgency=0, place well behind mid
        half_spread = spread_bps / 2.0
        offset = half_spread * (1.0 - 2.0 * urgency)
        return offset
