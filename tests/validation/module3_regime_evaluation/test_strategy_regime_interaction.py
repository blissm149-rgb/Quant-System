"""Test strategy-regime interaction and dynamic allocation.

Validates that MarketStateClassifier state weights are correctly
applied and that regime-conditional strategy selection is coherent.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.research_algorithms.regime_models.market_state_classifier import (
    MarketState,
    MarketStateClassifier,
)
from quant_fund.research_algorithms.regime_models.volatility_regime_detector import (
    VolatilityRegime,
    VolatilityRegimeDetector,
)


pytestmark = [pytest.mark.validation]


class TestStrategyRegimeInteraction:
    """Verify that regime signals translate correctly to strategy gating."""

    def test_state_weights_monotonically_decrease_with_risk(self):
        """State weight multipliers must decrease from RISK_ON to CRISIS."""
        classifier = MarketStateClassifier()
        weights = classifier.get_state_weights()

        assert weights[MarketState.RISK_ON] > weights[MarketState.TRANSITION], (
            "RISK_ON weight should exceed TRANSITION"
        )
        assert weights[MarketState.TRANSITION] > weights[MarketState.RISK_OFF], (
            "TRANSITION weight should exceed RISK_OFF"
        )
        assert weights[MarketState.RISK_OFF] > weights[MarketState.CRISIS], (
            "RISK_OFF weight should exceed CRISIS"
        )

    def test_crisis_weight_below_thirty_percent(self):
        """Crisis exposure should be at most 30% of full exposure."""
        classifier = MarketStateClassifier()
        weights = classifier.get_state_weights()

        assert weights[MarketState.CRISIS] <= 0.30, (
            f"Crisis weight = {weights[MarketState.CRISIS]}, expected <= 0.30"
        )

    def test_classifier_returns_crisis_on_extreme_drawdown(self):
        """Feed returns with a 15%+ drawdown and high vol — should classify
        as CRISIS."""
        rng = np.random.default_rng(42)
        # Normal period
        normal = rng.normal(0.0003, 0.01, 200)
        # Extreme crash: large negative returns with very high vol
        crash = rng.normal(-0.04, 0.06, 30)
        combined = np.concatenate([normal, crash])
        dates = pd.bdate_range("2020-01-02", periods=len(combined))
        returns = pd.Series(combined, index=dates)

        classifier = MarketStateClassifier()
        classifier.fit(returns)
        state = classifier.classify(returns)

        assert state in (MarketState.CRISIS, MarketState.RISK_OFF), (
            f"Expected CRISIS or RISK_OFF after extreme drawdown, got {state}"
        )

    def test_classifier_returns_risk_on_in_calm_bull(self):
        """Low-volatility uptrend should classify as RISK_ON."""
        rng = np.random.default_rng(42)
        # Steady positive returns with low vol
        calm_bull = rng.normal(0.001, 0.005, 300)
        dates = pd.bdate_range("2020-01-02", periods=len(calm_bull))
        returns = pd.Series(calm_bull, index=dates)

        classifier = MarketStateClassifier()
        classifier.fit(returns)
        state = classifier.classify(returns)

        assert state in (MarketState.RISK_ON, MarketState.TRANSITION), (
            f"Expected RISK_ON or TRANSITION in calm bull, got {state}"
        )
