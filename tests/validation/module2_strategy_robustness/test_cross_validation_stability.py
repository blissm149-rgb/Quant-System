"""Test cross-validation stability of factor models.

Validates that strategy performance is stable across time-series splits
and that rolling-window backtests do not show pathological variance.
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
from tests.conftest import make_ohlcv, STANDARD_TICKERS
from tests.validation.shared.backtest_harness import BacktestHarness
from tests.validation.shared.metrics import annualized_sharpe, max_drawdown


pytestmark = [pytest.mark.validation]


class TestCrossValidationStability:
    """Verify that strategy performance is stable across time splits."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        # Use 1000 periods to ensure each fold has enough data for momentum lookback
        self.ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=1000, seed=42)
        self.dates = self.ohlcv.index.get_level_values(0).unique().sort_values()

    def _split_data(self, n_folds=2):
        """Split data into n_folds consecutive time windows."""
        dates = self.dates
        fold_size = len(dates) // n_folds
        folds = []
        for i in range(n_folds):
            start = dates[i * fold_size]
            end = dates[min((i + 1) * fold_size - 1, len(dates) - 1)]
            fold = self.ohlcv[
                (self.ohlcv.index.get_level_values(0) >= start)
                & (self.ohlcv.index.get_level_values(0) <= end)
            ]
            folds.append(fold)
        return folds

    def test_sharpe_variance_across_folds(self):
        """Sharpe ratio across 2 time-folds should have std < 2.0,
        indicating no single-fold dominance."""
        folds = self._split_data(n_folds=2)
        harness = BacktestHarness(seed=42)
        # Use shorter-lookback strategy so each fold has enough data
        strategy = ZScoreReversionStrategy()

        sharpes = []
        for fold in folds:
            result = harness.run(fold, strategy)
            if len(result.daily_returns) > 10:
                sharpes.append(annualized_sharpe(result.daily_returns))

        assert len(sharpes) >= 2, "Need at least 2 folds with returns"
        sharpe_std = np.std(sharpes)
        assert sharpe_std < 2.0, (
            f"Sharpe std across folds = {sharpe_std:.2f}, expected < 2.0. "
            f"Sharpes per fold: {[f'{s:.2f}' for s in sharpes]}"
        )

    def test_no_fold_has_catastrophic_drawdown(self):
        """No single fold should have max drawdown > 50% for standard
        long-short factors."""
        folds = self._split_data(n_folds=3)
        harness = BacktestHarness(seed=42)
        strategies = [MomentumFactor(), LowVolatilityFactor()]

        for strategy in strategies:
            for i, fold in enumerate(folds):
                result = harness.run(fold, strategy)
                if len(result.daily_returns) > 10:
                    mdd = max_drawdown(result.daily_returns)
                    assert mdd < 0.50, (
                        f"{strategy.feature_name} fold {i}: max drawdown = "
                        f"{mdd:.1%}, expected < 50%"
                    )

    def test_rolling_window_performance_not_concentrated(self):
        """Rolling 63-day Sharpe windows should not have > 80% of positive
        values concentrated in a single quarter of the dataset."""
        harness = BacktestHarness(seed=42)
        # Use a shorter-lookback strategy so we get more return observations
        strategy = ZScoreReversionStrategy()
        result = harness.run(self.ohlcv, strategy)

        if len(result.daily_returns) < 126:
            pytest.skip("Not enough returns for rolling analysis")

        rets = result.daily_returns
        rolling_sharpe = (
            rets.rolling(63).mean() / rets.rolling(63).std() * np.sqrt(252)
        ).dropna()

        if len(rolling_sharpe) < 20:
            pytest.skip("Not enough rolling windows")

        positive_sharpe = rolling_sharpe > 0
        n = len(positive_sharpe)
        quarter_size = n // 4

        # Count positive Sharpe windows in each quarter
        quarter_counts = []
        for q in range(4):
            start = q * quarter_size
            end = start + quarter_size if q < 3 else n
            count = positive_sharpe.iloc[start:end].sum()
            quarter_counts.append(count)

        total_positive = sum(quarter_counts)
        if total_positive > 0:
            max_quarter_pct = max(quarter_counts) / total_positive
            assert max_quarter_pct < 0.80, (
                f"Positive Sharpe windows are {max_quarter_pct:.0%} concentrated "
                f"in a single quarter (quarters: {quarter_counts})"
            )
