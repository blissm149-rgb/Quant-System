"""Test behavior during regime transitions.

Validates that the system handles transitions between regimes
without catastrophic errors, and that multi-regime sequences
produce continuous results.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.research_algorithms.regime_models.volatility_regime_detector import (
    VolatilityRegime,
    VolatilityRegimeDetector,
)
from quant_fund.research_algorithms.regime_models.market_state_classifier import (
    MarketState,
    MarketStateClassifier,
)
from tests.conftest import STANDARD_TICKERS
from tests.generators.market_regime_simulator import MarketRegimeSimulator
from tests.validation.shared.backtest_harness import BacktestHarness
from tests.validation.shared.metrics import max_drawdown


pytestmark = [pytest.mark.validation]


def _extract_market_returns(ohlcv: pd.DataFrame) -> pd.Series:
    """Extract equal-weighted market return from OHLCV."""
    tickers = ohlcv.index.get_level_values("ticker").unique()
    dates = ohlcv.index.get_level_values(0).unique().sort_values()
    daily_rets = []
    for i in range(1, len(dates)):
        rets = []
        for t in tickers:
            try:
                prev = ohlcv.loc[(dates[i - 1], t), "close"]
                curr = ohlcv.loc[(dates[i], t), "close"]
                rets.append((curr - prev) / prev)
            except KeyError:
                continue
        if rets:
            daily_rets.append(np.mean(rets))
        else:
            daily_rets.append(0.0)
    return pd.Series(daily_rets, index=dates[1:])


class TestRegimeTransitionBehavior:
    """Verify correct behavior during regime transitions."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        sim = MarketRegimeSimulator(tickers=STANDARD_TICKERS[:5], seed=42)
        self.multi_regime = sim.regime_sequence(
            [
                ("bull", {"days": 100}),
                ("crash", {"days": 10}),
                ("bear", {"days": 60}),
                ("recovery", {"days": 80}),
                ("sideways", {"days": 100}),
            ]
        )
        self.market_returns = _extract_market_returns(self.multi_regime)

    def test_volatility_regime_changes_during_sequence(self):
        """VolatilityRegimeDetector should detect at least 2 different regimes
        across a bull->crash->bear->recovery->sideways sequence."""
        detector = VolatilityRegimeDetector()

        regimes_seen = set()
        window = 60
        for i in range(window, len(self.market_returns), 10):
            chunk = self.market_returns.iloc[:i]
            regime = detector.detect(chunk)
            regimes_seen.add(regime)

        assert len(regimes_seen) >= 2, (
            f"Expected at least 2 volatility regimes in multi-regime sequence, "
            f"only saw {regimes_seen}"
        )

    def test_market_state_classifier_detects_transitions(self):
        """MarketStateClassifier should produce at least 2 different states
        across the regime sequence."""
        classifier = MarketStateClassifier()
        classifier.fit(self.market_returns)

        states_seen = set()
        window = 60
        for i in range(window, len(self.market_returns), 10):
            chunk = self.market_returns.iloc[:i]
            state = classifier.classify(chunk)
            states_seen.add(state)

        assert len(states_seen) >= 2, (
            f"Expected at least 2 market states, saw: {states_seen}"
        )

    def test_backtest_continuous_through_regime_transitions(self):
        """Backtest should run continuously through regime transitions
        without gaps or errors."""
        from quant_fund.research_algorithms.mean_reversion.zscore_reversion_strategy import (
            ZScoreReversionStrategy,
        )

        harness = BacktestHarness(seed=42)
        strategy = ZScoreReversionStrategy()

        result = harness.run(self.multi_regime, strategy)

        assert len(result.daily_returns) > 50, (
            f"Backtest should produce returns across 350-day multi-regime "
            f"sequence, got {len(result.daily_returns)}"
        )

        # NAV should be continuous (no NaN gaps)
        assert result.nav_series.isna().sum() == 0, (
            "NAV series should have no NaN gaps during regime transitions"
        )

    def test_no_extreme_single_day_return_in_transition(self):
        """During regime transitions, no single-day portfolio return should
        exceed +/- 20% for a diversified long-short strategy."""
        from quant_fund.research_algorithms.factor_models.momentum_factor import (
            MomentumFactor,
        )

        harness = BacktestHarness(seed=42)
        strategy = MomentumFactor()

        result = harness.run(self.multi_regime, strategy)

        if len(result.daily_returns) > 0:
            max_abs_ret = result.daily_returns.abs().max()
            assert max_abs_ret < 0.20, (
                f"Max single-day absolute return = {max_abs_ret:.1%}, "
                f"expected < 20% for diversified long-short"
            )
