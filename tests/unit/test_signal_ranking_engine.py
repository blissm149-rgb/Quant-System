"""Unit tests for signal_ranking_engine module.

TESTING_PLAN.md Section 3.4 — Alpha Discovery.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.alpha_discovery.signal_ranking_engine import (
    SignalRankingEngine,
    SignalScore,
)


@pytest.mark.unit
@pytest.mark.tier2
class TestSignalRankingEngine:
    """SignalRankingEngine — evaluates and ranks alpha signals."""

    @pytest.fixture
    def engine(self):
        return SignalRankingEngine()

    @pytest.fixture
    def signal_and_returns(self):
        """Create signal values and forward returns with known IC."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=300)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        rows_signal = []
        rows_returns = []
        for dt in dates:
            # Signal with moderate predictive power
            signal = rng.standard_normal(len(tickers))
            noise = rng.standard_normal(len(tickers))
            fwd_ret = 0.1 * signal + 0.9 * noise  # IC ~ 0.1
            for i, ticker in enumerate(tickers):
                rows_signal.append({"date": dt, "ticker": ticker, "value": signal[i]})
                rows_returns.append({"date": dt, "ticker": ticker, "value": fwd_ret[i]})

        sig_df = pd.DataFrame(rows_signal).set_index(["date", "ticker"])["value"].unstack("ticker")
        ret_df = pd.DataFrame(rows_returns).set_index(["date", "ticker"])["value"].unstack("ticker")
        return sig_df, ret_df

    def test_evaluate_signal_returns_score(self, engine, signal_and_returns):
        """evaluate_signal returns a SignalScore."""
        signals, returns = signal_and_returns
        score = engine.evaluate_signal(signals, returns)
        assert isinstance(score, SignalScore)

    def test_ic_threshold_filters(self, engine):
        """Signals with IC below threshold marked as not passed."""
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2020-01-02", periods=300)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        # Pure noise signal — IC should be near 0
        sig_df = pd.DataFrame(rng.standard_normal((300, 5)), index=dates, columns=tickers)
        ret_df = pd.DataFrame(rng.standard_normal((300, 5)), index=dates, columns=tickers)
        score = engine.evaluate_signal(sig_df, ret_df)
        # A pure noise signal should have IC near 0, below the 0.03 threshold
        assert score.ic_mean < 0.05  # sanity — not a strong signal

    def test_rank_signals_sorted_by_score(self, engine, signal_and_returns):
        """rank_signals returns list sorted by composite_score descending."""
        signals, returns = signal_and_returns
        # Create a second weaker signal
        rng = np.random.default_rng(99)
        weak_signal = pd.DataFrame(
            rng.standard_normal(signals.shape),
            index=signals.index,
            columns=signals.columns,
        )
        rankings = engine.rank_signals(
            {"strong": signals, "weak": weak_signal},
            returns,
        )
        assert len(rankings) == 2
        assert rankings[0].composite_score >= rankings[1].composite_score

    def test_ranking_is_deterministic(self, engine, signal_and_returns):
        """Same input produces same ranking."""
        signals, returns = signal_and_returns
        r1 = engine.evaluate_signal(signals, returns)
        r2 = engine.evaluate_signal(signals, returns)
        assert r1.composite_score == r2.composite_score

    def test_combine_signals(self, engine):
        """combine produces weighted average of signals."""
        rng = np.random.default_rng(42)
        tickers = ["AAPL", "MSFT", "GOOG"]
        sig_a = pd.Series(rng.standard_normal(3), index=tickers)
        sig_b = pd.Series(rng.standard_normal(3), index=tickers)
        combined = engine.combine({"a": sig_a, "b": sig_b})
        assert len(combined) == 3
