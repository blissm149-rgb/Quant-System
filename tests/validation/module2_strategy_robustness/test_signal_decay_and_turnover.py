"""Test signal decay and turnover characteristics.

Validates that factor signals exhibit realistic autocorrelation and that
portfolio turnover implied by the signal is within practical bounds.
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


pytestmark = [pytest.mark.validation]


class TestSignalDecayAndTurnover:
    """Verify signal persistence and implied turnover are realistic."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.ohlcv = make_ohlcv(tickers=STANDARD_TICKERS, periods=500, seed=42)
        self.dates = self.ohlcv.index.get_level_values(0).unique().sort_values()
        self.engine = DataAlignmentEngine()

    def _compute_signal_series(self, strategy, start_idx=300, n_days=20):
        """Compute signals at consecutive dates, return dict of date -> scores."""
        signals = {}
        lookback = strategy.lookback_days
        for i in range(start_idx, start_idx + n_days):
            as_of = self.dates[i]
            safe = self.ohlcv[self.ohlcv.index.get_level_values(0) < as_of]
            aligned = self.engine.get_aligned_data(safe, as_of, lookback)
            if aligned.empty:
                continue
            scores = strategy.compute(aligned, as_of)
            if not scores.empty and scores.notna().any():
                signals[as_of] = scores
        return signals

    def test_momentum_signal_has_high_autocorrelation(self):
        """Momentum signal (12-1 month) should have day-to-day rank
        autocorrelation > 0.9 (slow-moving signal)."""
        strategy = MomentumFactor()
        signals = self._compute_signal_series(strategy, start_idx=300, n_days=15)

        signal_dates = sorted(signals.keys())
        autocorrs = []
        for i in range(1, len(signal_dates)):
            prev = signals[signal_dates[i - 1]]
            curr = signals[signal_dates[i]]
            merged = pd.concat([prev, curr], axis=1).dropna()
            if len(merged) >= 3:
                corr = merged.iloc[:, 0].corr(merged.iloc[:, 1], method="spearman")
                autocorrs.append(corr)

        assert len(autocorrs) >= 5, "Need at least 5 consecutive signal pairs"
        mean_autocorr = np.mean(autocorrs)
        assert mean_autocorr > 0.9, (
            f"Momentum day-to-day rank autocorrelation = {mean_autocorr:.3f}, "
            f"expected > 0.9 (slow-moving signal)"
        )

    def test_reversal_signal_has_lower_autocorrelation(self):
        """Short-term reversal signal should have lower autocorrelation than
        momentum (faster-decaying signal)."""
        momentum = MomentumFactor()
        reversal = ZScoreReversionStrategy()

        mom_signals = self._compute_signal_series(momentum, start_idx=300, n_days=15)
        rev_signals = self._compute_signal_series(reversal, start_idx=300, n_days=15)

        def _mean_autocorr(signals):
            dates = sorted(signals.keys())
            corrs = []
            for i in range(1, len(dates)):
                merged = pd.concat(
                    [signals[dates[i - 1]], signals[dates[i]]], axis=1
                ).dropna()
                if len(merged) >= 3:
                    corrs.append(
                        merged.iloc[:, 0].corr(merged.iloc[:, 1], method="spearman")
                    )
            return np.mean(corrs) if corrs else 0.0

        mom_ac = _mean_autocorr(mom_signals)
        rev_ac = _mean_autocorr(rev_signals)

        assert mom_ac > rev_ac, (
            f"Momentum autocorrelation ({mom_ac:.3f}) should exceed "
            f"reversal autocorrelation ({rev_ac:.3f})"
        )

    def test_implied_turnover_within_practical_bounds(self):
        """Implied daily turnover from signal changes should be < 50%
        for slow factors and < 200% for fast factors."""
        strategies = {
            "momentum": (MomentumFactor(), 0.50),
            "low_vol": (LowVolatilityFactor(), 0.50),
            "reversal": (ZScoreReversionStrategy(), 2.00),
        }

        for name, (strategy, max_turnover) in strategies.items():
            signals = self._compute_signal_series(strategy, start_idx=300, n_days=10)
            dates = sorted(signals.keys())

            turnovers = []
            for i in range(1, len(dates)):
                prev = signals[dates[i - 1]]
                curr = signals[dates[i]]
                # Normalise to unit weights
                prev_w = prev / prev.abs().sum() if prev.abs().sum() > 0 else prev
                curr_w = curr / curr.abs().sum() if curr.abs().sum() > 0 else curr
                common = prev_w.index.union(curr_w.index)
                to = (
                    curr_w.reindex(common, fill_value=0)
                    - prev_w.reindex(common, fill_value=0)
                ).abs().sum()
                turnovers.append(float(to))

            if turnovers:
                mean_to = np.mean(turnovers)
                assert mean_to < max_turnover, (
                    f"{name}: mean daily turnover = {mean_to:.3f}, "
                    f"expected < {max_turnover}"
                )

    def test_signal_rank_stability_over_week(self):
        """The top-quintile stocks should have >= 50% overlap between
        day T and day T+5 for momentum (a slow signal)."""
        strategy = MomentumFactor()
        signals = self._compute_signal_series(strategy, start_idx=300, n_days=10)
        dates = sorted(signals.keys())

        if len(dates) < 6:
            pytest.skip("Not enough signal dates")

        day0 = signals[dates[0]].dropna()
        day5 = signals[dates[5]].dropna()

        n_top = max(1, len(day0) // 5)
        top_day0 = set(day0.nlargest(n_top).index)
        top_day5 = set(day5.nlargest(n_top).index)

        overlap = len(top_day0 & top_day5) / max(len(top_day0), 1)
        assert overlap >= 0.5, (
            f"Top-quintile overlap between day 0 and day 5 = {overlap:.0%}, "
            f"expected >= 50% for momentum"
        )
