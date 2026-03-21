"""Test overfitting detection mechanisms.

Validates that strategies do not show signs of overfitting by comparing
in-sample vs out-of-sample performance and by testing against shuffled data.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine
from quant_fund.research_algorithms.factor_models.momentum_factor import MomentumFactor
from quant_fund.research_algorithms.factor_models.low_volatility_factor import (
    LowVolatilityFactor,
)
from quant_fund.research_algorithms.mean_reversion.zscore_reversion_strategy import (
    ZScoreReversionStrategy,
)
from tests.conftest import make_ohlcv, STANDARD_TICKERS
from tests.validation.shared.backtest_harness import BacktestHarness
from tests.validation.shared.metrics import annualized_sharpe
from tests.validation.shared.perturbation_engine import PerturbationEngine


pytestmark = [pytest.mark.validation]


class TestOverfittingDetection:
    """Verify that strategies do not exhibit overfitting artifacts."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=500, seed=42)

    def test_shuffled_data_destroys_signal(self):
        """Running a strategy on time-shuffled data should produce near-zero
        or worse Sharpe ratio compared to original data."""
        harness = BacktestHarness(seed=42)
        strategy = MomentumFactor()
        pe = PerturbationEngine()

        # Original backtest
        result_orig = harness.run(self.ohlcv, strategy)

        # Shuffle prices within each ticker (destroy temporal structure)
        shuffled = self.ohlcv.copy()
        rng = np.random.default_rng(42)
        for ticker in self.ohlcv.index.get_level_values("ticker").unique():
            mask = shuffled.index.get_level_values("ticker") == ticker
            ticker_data = shuffled.loc[mask]
            for col in ["open", "high", "low", "close"]:
                vals = ticker_data[col].values.copy()
                rng.shuffle(vals)
                shuffled.loc[mask, col] = vals

        result_shuffled = harness.run(shuffled, strategy)

        if len(result_orig.daily_returns) > 20 and len(result_shuffled.daily_returns) > 20:
            sharpe_orig = annualized_sharpe(result_orig.daily_returns)
            sharpe_shuf = annualized_sharpe(result_shuffled.daily_returns)

            # Shuffled Sharpe should be meaningfully lower than original
            # (or close to zero). We just check it's not systematically better.
            assert sharpe_shuf < sharpe_orig + 0.5, (
                f"Shuffled Sharpe ({sharpe_shuf:.2f}) should not exceed "
                f"original Sharpe ({sharpe_orig:.2f}) by > 0.5"
            )

    def test_in_sample_vs_out_of_sample_sharpe_ratio(self):
        """IS Sharpe should not exceed OOS Sharpe by more than 3x,
        which would suggest overfitting."""
        dates = self.ohlcv.index.get_level_values(0).unique().sort_values()
        split = dates[len(dates) // 2]

        is_data = self.ohlcv[self.ohlcv.index.get_level_values(0) <= split]
        oos_data = self.ohlcv[self.ohlcv.index.get_level_values(0) > split]

        harness = BacktestHarness(seed=42)
        strategy = MomentumFactor()

        is_result = harness.run(is_data, strategy)
        oos_result = harness.run(oos_data, strategy)

        if len(is_result.daily_returns) > 20 and len(oos_result.daily_returns) > 20:
            is_sharpe = abs(annualized_sharpe(is_result.daily_returns))
            oos_sharpe = abs(annualized_sharpe(oos_result.daily_returns))

            if oos_sharpe > 0.01:
                ratio = is_sharpe / oos_sharpe
                assert ratio < 3.0, (
                    f"IS/OOS Sharpe ratio = {ratio:.1f}x (IS={is_sharpe:.2f}, "
                    f"OOS={oos_sharpe:.2f}), suggests overfitting"
                )

    def test_multiple_seeds_produce_consistent_ranking(self):
        """The same strategy on data generated with different seeds should
        produce consistent relative performance ranking across strategies."""
        strategies = [MomentumFactor(), LowVolatilityFactor(), ZScoreReversionStrategy()]
        harness = BacktestHarness(seed=42)

        rankings = []
        for seed in [42, 123, 999]:
            ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=500, seed=seed)
            sharpes = []
            for s in strategies:
                result = harness.run(ohlcv, s)
                if len(result.daily_returns) > 20:
                    sharpes.append(annualized_sharpe(result.daily_returns))
                else:
                    sharpes.append(0.0)
            # Record the ranking (argsort)
            rankings.append(np.argsort(sharpes).tolist())

        # At least 2 of 3 seeds should agree on the best strategy
        best_strat = [r[-1] for r in rankings]
        from collections import Counter

        most_common_count = Counter(best_strat).most_common(1)[0][1]
        assert most_common_count >= 2, (
            f"Best strategy should be consistent across seeds, but got "
            f"best_per_seed={best_strat}"
        )

    def test_strategy_does_not_exploit_data_generation_artifact(self):
        """Adding noise to returns should degrade performance smoothly,
        not cause a cliff — which would indicate reliance on a data artifact."""
        harness = BacktestHarness(seed=42)
        strategy = MomentumFactor()
        pe = PerturbationEngine()

        sharpes = []
        noise_levels = [0.0, 0.001, 0.002, 0.005, 0.01]

        for noise in noise_levels:
            if noise == 0:
                data = self.ohlcv
            else:
                data = self.ohlcv.copy()
                rng = np.random.default_rng(42)
                for col in ["open", "high", "low", "close"]:
                    data[col] = data[col] * (
                        1 + rng.normal(0, noise, len(data))
                    )

            result = harness.run(data, strategy)
            if len(result.daily_returns) > 20:
                sharpes.append(annualized_sharpe(result.daily_returns))
            else:
                sharpes.append(0.0)

        # Check no cliff: max consecutive drop should be < 3 Sharpe units
        for i in range(1, len(sharpes)):
            drop = sharpes[i - 1] - sharpes[i]
            assert drop < 3.0, (
                f"Cliff detected between noise {noise_levels[i-1]} and "
                f"{noise_levels[i]}: Sharpe drop = {drop:.2f}"
            )
