"""Adverse selection model.

Estimates the probability that a counterparty has superior
information, and the expected adverse price movement post-trade.
Used to adjust limit order pricing and urgency.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AdverseSelectionEstimate:
    """Adverse selection metrics for a ticker."""

    ticker: str
    pin_estimate: float  # Probability of Informed Trading (0-1)
    adverse_component_bps: float  # adverse selection component of spread
    permanent_impact_bps: float  # expected permanent price impact
    toxicity_score: float  # 0-1, higher = more toxic flow
    timestamp: Optional[pd.Timestamp] = None


class AdverseSelectionModel:
    """Estimates adverse selection risk from trade and quote data.

    Implements simplified versions of:
    - VPIN (Volume-Synchronized Probability of Informed Trading)
    - Adverse selection component of effective spread
    - Trade flow toxicity scoring
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._vpin_bucket_size = cfg.get("vpin_bucket_size", 50)
        self._vpin_lookback = cfg.get("vpin_lookback", 50)
        self._toxicity_threshold = cfg.get("toxicity_threshold", 0.7)

    def estimate_vpin(self, trades: pd.DataFrame) -> float:
        """Estimate VPIN from trade data.

        Parameters
        ----------
        trades : pd.DataFrame
            Must have columns: price, volume, side (1=buy, -1=sell).
            If side is missing, uses tick rule.

        Returns
        -------
        float
            VPIN estimate between 0 and 1.
        """
        if len(trades) < self._vpin_bucket_size:
            return 0.0

        if "side" not in trades.columns:
            trades = trades.copy()
            trades["side"] = np.sign(trades["price"].diff().fillna(0))
            trades.loc[trades["side"] == 0, "side"] = 1

        buy_vol = trades.loc[trades["side"] > 0, "volume"]
        sell_vol = trades.loc[trades["side"] < 0, "volume"]

        total_buy = buy_vol.sum()
        total_sell = sell_vol.sum()
        total_vol = total_buy + total_sell

        if total_vol <= 0:
            return 0.0

        vpin = abs(total_buy - total_sell) / total_vol
        return float(np.clip(vpin, 0.0, 1.0))

    def estimate_adverse_component(
        self,
        effective_spread_bps: float,
        realised_spread_bps: float,
    ) -> float:
        """Compute adverse selection component of spread.

        Adverse component = effective spread - realised spread.
        This represents the portion of the spread attributable
        to trading against informed counterparties.
        """
        adverse = effective_spread_bps - realised_spread_bps
        return max(0.0, adverse)

    def compute_toxicity_score(
        self,
        returns: pd.Series,
        volumes: pd.Series,
    ) -> float:
        """Compute flow toxicity score.

        High toxicity = large volume on same side as subsequent
        price moves (informed flow). Score in [0, 1].
        """
        if len(returns) < 2 or len(volumes) < 2:
            return 0.0

        # Align lengths
        n = min(len(returns), len(volumes))
        ret = returns.iloc[:n].values
        vol = volumes.iloc[:n].values

        # Forward returns (how price moves after each bar)
        fwd_ret = np.roll(ret, -1)
        fwd_ret[-1] = 0.0

        # Volume-weighted direction agreement
        signed_vol = np.sign(ret) * vol
        agreement = signed_vol * np.sign(fwd_ret)

        total_vol = np.abs(vol).sum()
        if total_vol <= 0:
            return 0.0

        toxicity = np.sum(np.abs(agreement)) / total_vol
        return float(np.clip(toxicity, 0.0, 1.0))

    def estimate(
        self,
        ticker: str,
        trades: Optional[pd.DataFrame] = None,
        effective_spread_bps: float = 0.0,
        realised_spread_bps: float = 0.0,
        returns: Optional[pd.Series] = None,
        volumes: Optional[pd.Series] = None,
    ) -> AdverseSelectionEstimate:
        """Full adverse selection estimation.

        Combines VPIN, spread decomposition, and toxicity scoring.
        """
        pin = 0.0
        if trades is not None and not trades.empty:
            pin = self.estimate_vpin(trades)

        adverse_bps = self.estimate_adverse_component(
            effective_spread_bps, realised_spread_bps
        )

        toxicity = 0.0
        if returns is not None and volumes is not None:
            toxicity = self.compute_toxicity_score(returns, volumes)

        # Permanent impact estimate: adverse component * PIN scaling
        permanent_bps = adverse_bps * (1.0 + pin)

        return AdverseSelectionEstimate(
            ticker=ticker,
            pin_estimate=pin,
            adverse_component_bps=adverse_bps,
            permanent_impact_bps=permanent_bps,
            toxicity_score=toxicity,
            timestamp=pd.Timestamp.now(),
        )
