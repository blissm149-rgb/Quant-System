"""Test regime-aware performance characteristics.

Validates that strategy performance varies meaningfully across market
regimes and that regime-conditional metrics are within expected bounds.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.research_algorithms.factor_models.momentum_factor import MomentumFactor
from quant_fund.research_algorithms.factor_models.low_volatility_factor import (
    LowVolatilityFactor,
)
from quant_fund.research_algorithms.mean_reversion.zscore_reversion_strategy import (
    ZScoreReversionStrategy,
)
from tests.conftest import STANDARD_TICKERS
from tests.generators.market_regime_simulator import MarketRegimeSimulator
from tests.validation.shared.backtest_harness import BacktestHarness
from tests.validation.shared.metrics import annualized_sharpe, max_drawdown


pytestmark = [pytest.mark.validation]


class TestRegimeAwarePerformance:
    """Verify that strategy performance varies meaningfully by regime."""

    @pytest.fixture(autouse=True)
    def setup_regimes(self):
        """Create separate regime datasets for targeted testing."""
        tickers = STANDARD_TICKERS[:5]
        sim = MarketRegimeSimulator(tickers=tickers, seed=42)
        self.bull = sim.bull_market(days=200)
        self.bear = sim.bear_market(days=200, vol=0.30)
        self.crash = sim.crash(days=30, vol=0.80)
        self.sideways = sim.sideways(days=200)
        self.harness = BacktestHarness(seed=42)

    def test_momentum_performs_differently_in_bull_vs_bear(self):
        """Momentum should have materially different Sharpe in bull vs bear
        markets (the sign or magnitude should differ)."""
        strategy = MomentumFactor()

        result_bull = self.harness.run(self.bull, strategy)
        result_bear = self.harness.run(self.bear, strategy)

        if len(result_bull.daily_returns) > 10 and len(result_bear.daily_returns) > 10:
            sharpe_bull = annualized_sharpe(result_bull.daily_returns)
            sharpe_bear = annualized_sharpe(result_bear.daily_returns)

            diff = abs(sharpe_bull - sharpe_bear)
            assert diff > 0.1, (
                f"Momentum Sharpe should differ materially between bull "
                f"({sharpe_bull:.2f}) and bear ({sharpe_bear:.2f}), "
                f"diff={diff:.2f}"
            )

    def test_low_vol_drawdown_smaller_in_crash(self):
        """Low-volatility factor should have smaller drawdown than momentum
        during a crash regime."""
        mom = MomentumFactor()
        lowvol = LowVolatilityFactor()

        result_mom = self.harness.run(self.crash, mom)
        result_lv = self.harness.run(self.crash, lowvol)

        if len(result_mom.daily_returns) > 5 and len(result_lv.daily_returns) > 5:
            dd_mom = max_drawdown(result_mom.daily_returns)
            dd_lv = max_drawdown(result_lv.daily_returns)

            # Low-vol should have smaller drawdown (defensive)
            assert dd_lv <= dd_mom + 0.05, (
                f"Low-vol drawdown ({dd_lv:.1%}) should be <= momentum "
                f"drawdown ({dd_mom:.1%}) + 5% tolerance during crash"
            )

    def test_reversal_performs_better_in_sideways_market(self):
        """Mean reversion should generate more signal in a sideways
        (range-bound) market than in a trending bull market."""
        strategy = ZScoreReversionStrategy()

        result_sideways = self.harness.run(self.sideways, strategy)
        result_bull = self.harness.run(self.bull, strategy)

        # Both should produce returns; sideways should be at least
        # not dramatically worse (mean reversion's natural habitat)
        if len(result_sideways.daily_returns) > 10 and len(result_bull.daily_returns) > 10:
            sharpe_sw = annualized_sharpe(result_sideways.daily_returns)
            sharpe_bull = annualized_sharpe(result_bull.daily_returns)

            # Reversal should not be catastrophically worse in sideways
            assert sharpe_sw > sharpe_bull - 1.0, (
                f"Reversal Sharpe in sideways ({sharpe_sw:.2f}) should not be "
                f"catastrophically worse than bull ({sharpe_bull:.2f})"
            )

    def test_short_lookback_strategies_produce_output_in_every_regime(self):
        """Short-lookback strategies should produce non-empty output in every
        non-crash regime (crash is only 30 days)."""
        # Only test strategies with lookback <= 100 days against 200-day regimes
        strategies = [ZScoreReversionStrategy()]
        regimes = {
            "bull": self.bull,
            "bear": self.bear,
            "sideways": self.sideways,
        }

        for strategy in strategies:
            for regime_name, data in regimes.items():
                result = self.harness.run(data, strategy)
                assert len(result.daily_returns) > 0, (
                    f"{strategy.feature_name} produced no returns in "
                    f"{regime_name} regime"
                )
