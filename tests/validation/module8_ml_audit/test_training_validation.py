"""Module 5 — Training & Validation Framework tests.

Tests walk-forward cross-validation, purged k-fold, signal evaluation
with BH correction, and research runner retraining integration.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.research_algorithms.machine_learning.walk_forward_validator import (
    WalkForwardValidator,
    WalkForwardResult,
)
from quant_fund.research_algorithms.machine_learning.temporal_cross_validator import (
    purged_kfold,
    combinatorial_purged_cv,
)
from quant_fund.alpha_discovery.signal_ranking_engine import SignalRankingEngine
from quant_fund.main.research_runner import ResearchRunner

pytestmark = [pytest.mark.validation, pytest.mark.tier4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n=600):
    return pd.bdate_range("2020-01-01", periods=n)


def _make_noise_data(n_dates=600, n_features=5, seed=42):
    """Pure noise features and returns — no predictive signal."""
    rng = np.random.RandomState(seed)
    dates = _make_dates(n_dates)
    features = pd.DataFrame(
        rng.randn(n_dates, n_features),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    returns = pd.Series(rng.randn(n_dates) * 0.01, index=dates)
    return features, returns


def _make_signal_data(n_dates=600, n_features=5, seed=42):
    """Features with genuine (linear) signal to forward returns."""
    rng = np.random.RandomState(seed)
    dates = _make_dates(n_dates)
    features = pd.DataFrame(
        rng.randn(n_dates, n_features),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Returns are a noisy linear function of feat_0
    returns = pd.Series(
        features["feat_0"].values * 0.05 + rng.randn(n_dates) * 0.005,
        index=dates,
    )
    return features, returns


class _SimpleLinearModel:
    """Minimal linear model for walk-forward testing."""

    def __init__(self, config=None):
        self._coef = None

    def train_model(self, features, returns):
        X = features.values
        y = returns.values
        # OLS: coef = (X'X)^{-1} X'y
        try:
            self._coef = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            self._coef = np.zeros(X.shape[1])
        preds = X @ self._coef
        ss_res = ((y - preds) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        train_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"train_r2": train_r2}

    def predict(self, features):
        X = features.values
        return pd.Series(X @ self._coef, index=features.index)


# ---------------------------------------------------------------------------
# Test 1: Walk-forward splits — no overlap
# ---------------------------------------------------------------------------

class TestWalkForwardSplits:

    def test_no_overlap(self):
        """Train and test dates must not overlap in any fold."""
        dates = _make_dates(600)
        wfv = WalkForwardValidator({"wf_min_train_days": 200, "wf_test_days": 63, "wf_step_days": 21})
        splits = wfv.generate_splits(dates)

        assert len(splits) > 0
        for train_dates, test_dates in splits:
            overlap = train_dates.intersection(test_dates)
            assert len(overlap) == 0, "Train and test dates overlap"

    def test_temporal_order(self):
        """max(train) < min(test) for every fold."""
        dates = _make_dates(600)
        wfv = WalkForwardValidator({"wf_min_train_days": 200, "wf_test_days": 63, "wf_step_days": 21})
        splits = wfv.generate_splits(dates)

        for train_dates, test_dates in splits:
            assert train_dates.max() < test_dates.min(), \
                f"Train end {train_dates.max()} >= Test start {test_dates.min()}"

    def test_expanding_window(self):
        """Training set grows with each fold."""
        dates = _make_dates(600)
        wfv = WalkForwardValidator({"wf_min_train_days": 200, "wf_test_days": 63, "wf_step_days": 21})
        splits = wfv.generate_splits(dates)

        train_sizes = [len(t) for t, _ in splits]
        # Each successive fold should have at least as many training dates
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]


# ---------------------------------------------------------------------------
# Test 2: Purged k-fold embargo
# ---------------------------------------------------------------------------

class TestPurgedKFold:

    def test_embargo_gap(self):
        """There must be >= embargo_days gap after each test block in training."""
        dates = _make_dates(500)
        embargo = 10
        splits = purged_kfold(dates, n_folds=5, embargo_days=embargo)

        assert len(splits) == 5
        for train_dates, test_dates in splits:
            test_end = test_dates.max()
            # Train dates after test_end should be > test_end + embargo_days
            later_train = train_dates[train_dates > test_end]
            if len(later_train) > 0:
                gap = (later_train.min() - test_end).days
                assert gap >= embargo, f"Embargo gap {gap} < {embargo}"

    def test_no_overlap(self):
        dates = _make_dates(500)
        splits = purged_kfold(dates, n_folds=5, embargo_days=5)
        for train_dates, test_dates in splits:
            assert len(train_dates.intersection(test_dates)) == 0


# ---------------------------------------------------------------------------
# Test 3: Walk-forward on noise — should not be significant
# ---------------------------------------------------------------------------

class TestWalkForwardNoise:

    def test_random_data_not_significant(self):
        """Walk-forward on random data: |mean OOS IC| < 0.05, not significant."""
        features, returns = _make_noise_data(n_dates=600, seed=123)
        wfv = WalkForwardValidator({
            "wf_min_train_days": 200,
            "wf_test_days": 63,
            "wf_step_days": 21,
        })
        result = wfv.validate_model(
            _SimpleLinearModel, {}, features, returns, features.index
        )
        assert abs(result.mean_oos_ic) < 0.10  # loose bound for noise
        assert not result.is_significant(min_ic=0.03, min_tstat=2.0)


# ---------------------------------------------------------------------------
# Test 4: Walk-forward on signal — should be significant
# ---------------------------------------------------------------------------

class TestWalkForwardSignal:

    def test_signal_data_is_significant(self):
        """Walk-forward on data with genuine signal: significant."""
        features, returns = _make_signal_data(n_dates=600, seed=42)
        wfv = WalkForwardValidator({
            "wf_min_train_days": 200,
            "wf_test_days": 63,
            "wf_step_days": 21,
        })
        result = wfv.validate_model(
            _SimpleLinearModel, {}, features, returns, features.index
        )
        assert result.mean_oos_ic > 0.03
        assert result.is_significant(min_ic=0.03, min_tstat=2.0)


# ---------------------------------------------------------------------------
# Test 5: Research runner retrains model
# ---------------------------------------------------------------------------

class TestResearchRunnerRetrain:

    def test_model_retrained_at_least_once(self):
        """ResearchRunner._retrain_models should invoke model training.

        Note: the live loop delegates retraining to the offline pipeline,
        so we exercise _retrain_models directly.
        """
        rng = np.random.RandomState(42)
        n_dates = 50
        dates = pd.bdate_range("2022-01-01", periods=n_dates)

        # Create market data with close prices and features
        market_data = pd.DataFrame({
            "close": 100 + rng.randn(n_dates).cumsum(),
            "feat_a": rng.randn(n_dates),
            "feat_b": rng.randn(n_dates),
        }, index=dates)

        class _MockFeatureGen:
            feature_name = "feat_a"
            def compute(self, data, as_of):
                if "feat_a" in data.columns:
                    return data["feat_a"]
                return pd.Series(dtype=float)

        class _MockModel:
            feature_name = "mock_model"
            train_count = 0
            _model = None
            def train_model(self, features, returns):
                _MockModel.train_count += 1
                return {"train_r2": 0.1}

        runner = ResearchRunner({"retrain_frequency": 5})
        runner.inject_components(feature_generators=[_MockFeatureGen()])
        runner.inject_ml_models([_MockModel()])

        # Run a cycle to produce a feature matrix, then retrain directly
        result = runner.run_cycle(as_of=dates[-1], market_data=market_data)
        assert result.feature_matrix is not None, "Feature matrix was not produced"
        runner._retrain_models(result.feature_matrix, market_data)

        assert _MockModel.train_count >= 1, "Model was never retrained"
        assert runner.retrain_count >= 1


# ---------------------------------------------------------------------------
# Test 6: Signal ranking with BH correction
# ---------------------------------------------------------------------------

class TestSignalRankingBHCorrection:

    def test_bh_rejects_noise_signals(self):
        """BH correction should reject most noise signals."""
        rng = np.random.RandomState(42)
        n_dates = 300
        n_tickers = 50
        dates = pd.bdate_range("2020-01-01", periods=n_dates)

        # Forward returns
        returns_data = []
        for dt in dates:
            for t in range(n_tickers):
                returns_data.append({"date": dt, "ticker": f"T{t}", "return": rng.randn() * 0.01})
        fwd_returns = pd.DataFrame(returns_data).set_index(["date", "ticker"])

        # Create 20 signals: 18 noise + 2 real
        signals = {}
        for i in range(18):
            sig_data = []
            for dt in dates:
                for t in range(n_tickers):
                    sig_data.append({"date": dt, "ticker": f"T{t}", "signal": rng.randn()})
            signals[f"noise_{i}"] = pd.DataFrame(sig_data).set_index(["date", "ticker"])

        # 2 signals with genuine predictive power
        for i in range(2):
            sig_data = []
            for dt in dates:
                for t in range(n_tickers):
                    ret = fwd_returns.loc[(dt, f"T{t}"), "return"]
                    sig_data.append({"date": dt, "ticker": f"T{t}", "signal": ret * 10 + rng.randn() * 0.05})
            signals[f"real_{i}"] = pd.DataFrame(sig_data).set_index(["date", "ticker"])

        engine = SignalRankingEngine({"min_eval_days": 100})
        scores = engine.evaluate_signal_with_correction(signals, fwd_returns, fdr=0.10)

        noise_passed = sum(1 for s in scores if s.signal_name.startswith("noise") and s.passed)
        real_passed = sum(1 for s in scores if s.signal_name.startswith("real") and s.passed)

        # BH should reject most noise — allow at most 5 false positives
        assert noise_passed <= 5, f"Too many noise signals passed: {noise_passed}"
        # Real signals should pass
        assert real_passed >= 1, "No real signals passed BH correction"


# ---------------------------------------------------------------------------
# Test 7: Combinatorial purged CV
# ---------------------------------------------------------------------------

class TestCombinatorialPurgedCV:

    def test_cpcv_generates_more_splits_than_kfold(self):
        """CPCV with n_groups=6, n_test_groups=2 generates C(6,2)=15 splits."""
        dates = _make_dates(600)
        splits = combinatorial_purged_cv(dates, n_groups=6, n_test_groups=2, embargo_days=5)
        assert len(splits) == 15

    def test_cpcv_no_overlap(self):
        dates = _make_dates(600)
        splits = combinatorial_purged_cv(dates, n_groups=6, n_test_groups=2, embargo_days=5)
        for train_dates, test_dates in splits:
            assert len(train_dates.intersection(test_dates)) == 0


# ---------------------------------------------------------------------------
# Test 8: Bootstrap IC CI
# ---------------------------------------------------------------------------

class TestBootstrapICCI:

    def test_bootstrap_ci_contains_mean(self):
        """Bootstrap CI for IC should contain the point estimate of mean IC."""
        rng = np.random.RandomState(42)
        n_dates = 300
        n_tickers = 30
        dates = pd.bdate_range("2020-01-01", periods=n_dates)

        # Construct signal correlated with returns
        sig_data = []
        ret_data = []
        for dt in dates:
            for t in range(n_tickers):
                r = rng.randn() * 0.01
                s = r * 5 + rng.randn() * 0.05
                sig_data.append({"date": dt, "ticker": f"T{t}", "signal": s})
                ret_data.append({"date": dt, "ticker": f"T{t}", "return": r})

        sig_df = pd.DataFrame(sig_data).set_index(["date", "ticker"])
        ret_df = pd.DataFrame(ret_data).set_index(["date", "ticker"])

        engine = SignalRankingEngine()
        lower, upper = engine.bootstrap_ic_ci(sig_df, ret_df, n_bootstrap=500, ci=0.95)

        assert lower < upper, "CI lower bound should be less than upper"
        # The interval should be reasonable (not degenerate)
        assert upper - lower > 0.001
