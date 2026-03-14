"""Bid-ask spread estimation model.

Estimates effective and realised spreads from trade/quote data.
Used by execution algorithms to estimate trading costs and by
the fill probability model to gauge liquidity conditions.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpreadEstimate:
    """Estimated spread metrics for a single ticker."""

    ticker: str
    quoted_spread_bps: float
    effective_spread_bps: float
    realised_spread_bps: float
    mid_price: float
    timestamp: Optional[pd.Timestamp] = None


class BidAskSpreadModel:
    """Estimates bid-ask spreads from trade and quote data.

    Methods
    -------
    estimate_quoted_spread
        Simple (ask - bid) / mid spread.
    estimate_effective_spread
        Roll (1984) estimator from autocovariance of returns.
    estimate_from_quotes
        Full estimation from a quotes DataFrame.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._min_observations = cfg.get("min_observations", 20)
        self._ewm_halflife = cfg.get("ewm_halflife", 20)

    def estimate_quoted_spread(
        self, bid: float, ask: float
    ) -> float:
        """Compute quoted spread in basis points."""
        if bid <= 0 or ask <= 0:
            return 0.0
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return 0.0
        return (ask - bid) / mid * 10000.0

    def estimate_effective_spread_roll(
        self, returns: pd.Series
    ) -> float:
        """Roll (1984) spread estimator from return autocovariance.

        Effective spread = 2 * sqrt(-Cov(r_t, r_{t-1})) when
        autocovariance is negative (as expected under bid-ask bounce).
        Returns spread in basis points.
        """
        if len(returns) < self._min_observations:
            return 0.0
        returns = returns.dropna()
        if len(returns) < self._min_observations:
            return 0.0
        autocov = returns.autocorr(lag=1) * returns.std() ** 2
        if autocov >= 0:
            # Positive autocov means no bid-ask bounce detected
            return 0.0
        spread = 2.0 * np.sqrt(-autocov)
        return spread * 10000.0

    def estimate_from_quotes(
        self, quotes: pd.DataFrame
    ) -> Dict[str, SpreadEstimate]:
        """Estimate spreads from a quotes DataFrame.

        Parameters
        ----------
        quotes : pd.DataFrame
            Must have columns: ticker, bid, ask. Optionally: timestamp.

        Returns
        -------
        dict mapping ticker -> SpreadEstimate
        """
        results: Dict[str, SpreadEstimate] = {}
        if quotes.empty:
            return results

        for ticker, group in quotes.groupby("ticker"):
            bids = group["bid"].values
            asks = group["ask"].values
            valid = (bids > 0) & (asks > 0)
            if not valid.any():
                continue

            bids = bids[valid]
            asks = asks[valid]
            mids = (bids + asks) / 2.0
            spreads_bps = (asks - bids) / mids * 10000.0

            quoted = float(np.mean(spreads_bps))
            # Effective spread ~ half quoted for market orders
            effective = quoted / 2.0
            # Realised spread ~ effective minus price impact
            realised = effective * 0.6

            results[str(ticker)] = SpreadEstimate(
                ticker=str(ticker),
                quoted_spread_bps=quoted,
                effective_spread_bps=effective,
                realised_spread_bps=realised,
                mid_price=float(mids[-1]),
                timestamp=pd.Timestamp.now(),
            )
        return results

    def estimate_time_weighted_spread(
        self, bid_series: pd.Series, ask_series: pd.Series
    ) -> float:
        """Time-weighted average spread in bps from bid/ask series."""
        if len(bid_series) == 0 or len(ask_series) == 0:
            return 0.0
        mid = (bid_series + ask_series) / 2.0
        spread = (ask_series - bid_series) / mid * 10000.0
        valid = spread[spread.notna() & (spread >= 0)]
        if len(valid) == 0:
            return 0.0
        return float(valid.mean())
