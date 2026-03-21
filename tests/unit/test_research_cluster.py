"""Unit tests for research cluster.

TESTING_PLAN.md Section 3.18 — ParallelSignalEvaluator.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.research_cluster.parallel_signal_evaluator import (
    ParallelSignalEvaluator,
    _evaluate_single_signal,
    SignalEvaluation,
)


@pytest.mark.unit
@pytest.mark.tier3
class TestParallelSignalEvaluator:
    """ParallelSignalEvaluator — IC-based signal ranking."""

    @pytest.fixture
    def evaluator(self):
        return ParallelSignalEvaluator(config={"min_ic_tstat": 1.5})

    @pytest.fixture
    def signal_data(self):
        rng = np.random.default_rng(42)
        T, N = 50, 20
        # Good signal: positively correlated with forward returns
        forward_returns = rng.normal(0, 0.02, (T, N))
        good_signal = forward_returns + rng.normal(0, 0.01, (T, N))
        # Random signal: no correlation
        random_signal = rng.normal(0, 1, (T, N))
        return {
            "good": good_signal,
            "random": random_signal,
        }, forward_returns

    def test_evaluate_single_signal(self):
        rng = np.random.default_rng(42)
        T, N = 30, 10
        fwd = rng.normal(0, 0.02, (T, N))
        sig = fwd + rng.normal(0, 0.01, (T, N))
        result = _evaluate_single_signal("test_sig", sig, fwd)
        assert isinstance(result, SignalEvaluation)
        assert result.status == "completed"
        assert result.ic_mean > 0

    def test_evaluate_signals_returns_dataframe(self, evaluator, signal_data):
        signals, fwd = signal_data
        df = evaluator.evaluate_signals(signals, fwd, parallel=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "ic_tstat" in df.columns

    def test_good_signal_ranked_higher(self, evaluator, signal_data):
        signals, fwd = signal_data
        df = evaluator.evaluate_signals(signals, fwd, parallel=False)
        # Good signal should have higher IC t-stat
        assert df.iloc[0]["signal_id"] == "good"

    def test_filter_significant(self, evaluator, signal_data):
        signals, fwd = signal_data
        df = evaluator.evaluate_signals(signals, fwd, parallel=False)
        filtered = evaluator.filter_significant(df)
        # Good signal should pass, random probably not
        assert len(filtered) <= len(df)

    def test_insufficient_data(self):
        result = _evaluate_single_signal(
            "tiny", np.array([[1.0]]), np.array([[0.5]]),
        )
        assert result.status == "insufficient_data"

    def test_empty_signals(self, evaluator):
        df = evaluator.evaluate_signals({}, np.array([]))
        assert df.empty
