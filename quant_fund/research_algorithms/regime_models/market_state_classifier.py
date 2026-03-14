"""Market state classifier combining HMM, macro regime, and volatility regime.

Produces a single MarketState enum used by dynamic_strategy_allocator
to adjust strategy weights.
"""

from enum import Enum
from typing import Optional

import pandas as pd

from quant_fund.research_algorithms.regime_models.hidden_markov_regime_model import (
    HiddenMarkovRegimeModel,
)
from quant_fund.research_algorithms.regime_models.volatility_regime_detector import (
    VolatilityRegime,
    VolatilityRegimeDetector,
)


class MarketState(Enum):
    RISK_ON = "risk_on"           # Low vol, positive momentum — full exposure
    RISK_OFF = "risk_off"         # High vol, negative momentum — defensive
    TRANSITION = "transition"     # Mixed signals — moderate exposure
    CRISIS = "crisis"             # Extreme vol, sharp drawdown — minimal exposure


class MarketStateClassifier:
    """Combines multiple regime signals into a single market state.

    Inputs:
    - HMM regime label (from hidden_markov_regime_model)
    - Volatility regime (from volatility_regime_detector)
    - Optional macro regime label (from macro_regime_classifier)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._hmm = HiddenMarkovRegimeModel(cfg)
        self._vol_detector = VolatilityRegimeDetector(cfg)
        self._crisis_vol_percentile = cfg.get("crisis_vol_percentile", 95)
        self._crisis_drawdown_threshold = cfg.get("crisis_drawdown_threshold", -0.10)

    def fit(self, market_returns: pd.Series) -> None:
        """Fit the underlying HMM model."""
        self._hmm.fit(market_returns)

    def classify(
        self,
        market_returns: pd.Series,
        macro_regime: Optional[str] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> MarketState:
        """Classify the current market state.

        Args:
            market_returns: Daily market returns up to (but not including) as_of.
            macro_regime: Optional macro regime string from macro_regime_classifier.
            as_of: Point-in-time boundary (for filtering if needed).

        Returns:
            Current MarketState.
        """
        if as_of is not None:
            market_returns = market_returns[market_returns.index < as_of]

        if len(market_returns) < 20:
            return MarketState.TRANSITION

        vol_regime = self._vol_detector.detect(market_returns)
        vol_percentile = self._vol_detector.get_vol_percentile(market_returns)

        # Check for crisis conditions
        recent_return = market_returns.iloc[-20:].sum()
        if vol_percentile >= self._crisis_vol_percentile and recent_return < self._crisis_drawdown_threshold:
            return MarketState.CRISIS

        # HMM regime
        hmm_regime = 0
        if self._hmm.is_fitted:
            hmm_regime = self._hmm.predict_regime(market_returns)

        # Combine signals
        if vol_regime == VolatilityRegime.HIGH:
            if hmm_regime == 0:  # typically the low-vol regime in fitted model
                return MarketState.TRANSITION
            return MarketState.RISK_OFF

        if vol_regime == VolatilityRegime.LOW:
            return MarketState.RISK_ON

        # Normal vol
        if macro_regime == "stagflation" or macro_regime == "deflation":
            return MarketState.RISK_OFF
        elif macro_regime == "goldilocks":
            return MarketState.RISK_ON

        return MarketState.TRANSITION

    def get_state_weights(self) -> dict:
        """Return default strategy weight multipliers for each state."""
        return {
            MarketState.RISK_ON: 1.2,
            MarketState.TRANSITION: 1.0,
            MarketState.RISK_OFF: 0.6,
            MarketState.CRISIS: 0.2,
        }
