"""Test regime detection accuracy on synthetic labeled data.

Validates that regime models correctly identify known synthetic regimes
and that their outputs are self-consistent.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.research_algorithms.regime_models.hidden_markov_regime_model import (
    HiddenMarkovRegimeModel,
)
from quant_fund.research_algorithms.regime_models.volatility_regime_detector import (
    VolatilityRegime,
    VolatilityRegimeDetector,
)
from quant_fund.research_algorithms.regime_models.market_state_classifier import (
    MarketState,
    MarketStateClassifier,
)


pytestmark = [pytest.mark.validation]


def _make_two_regime_returns(n_low=200, n_high=200, seed=42):
    """Create synthetic returns with two clearly distinct vol regimes."""
    rng = np.random.default_rng(seed)
    low_vol = rng.normal(0.0005, 0.005, n_low)   # ~0.8% annualized vol
    high_vol = rng.normal(-0.001, 0.025, n_high)  # ~4.0% annualized vol
    combined = np.concatenate([low_vol, high_vol])
    dates = pd.bdate_range("2020-01-02", periods=len(combined))
    return pd.Series(combined, index=dates)


class TestRegimeDetectionAccuracy:
    """Verify regime detection models on known synthetic data."""

    def test_hmm_identifies_two_regimes_in_bimodal_data(self):
        """HMM should fit successfully and produce 2 distinct regime means
        on bimodal return data."""
        returns = _make_two_regime_returns()
        hmm = HiddenMarkovRegimeModel(config={"hmm_n_regimes": 2})

        fit_result = hmm.fit(returns)

        assert hmm.is_fitted, "HMM should be fitted after calling fit()"
        assert fit_result["n_regimes"] == 2
        assert len(fit_result["regime_means"]) == 2

        # Regime means should be distinct
        means = fit_result["regime_means"]
        mean_diff = abs(means[0][0] - means[1][0])  # diff in return dimension
        assert mean_diff > 0.0001, (
            f"Regime means should be distinct: {means}"
        )

    def test_hmm_fitted_regime_means_separate_vol_clusters(self):
        """After fitting on bimodal data, the HMM regime means should reflect
        two distinct volatility clusters — one low-vol and one high-vol."""
        returns = _make_two_regime_returns(n_low=300, n_high=300, seed=42)
        hmm = HiddenMarkovRegimeModel(config={"hmm_n_regimes": 2})
        fit_result = hmm.fit(returns)

        # The volatility dimension (index 1) of regime means should differ
        means = np.array(fit_result["regime_means"])
        vol_dim_diff = abs(means[0][1] - means[1][1])

        assert vol_dim_diff > 0.001, (
            f"Regime means volatility dimension should differ meaningfully. "
            f"Means: {means.tolist()}, vol diff: {vol_dim_diff:.6f}"
        )

    def test_vol_detector_identifies_high_vol_in_crash_data(self):
        """VolatilityRegimeDetector should classify crash-like data
        as HIGH volatility."""
        rng = np.random.default_rng(42)
        # Create data: normal period then crash
        normal = rng.normal(0.0003, 0.01, 200)
        crash = rng.normal(-0.03, 0.05, 50)
        combined = np.concatenate([normal, crash])
        dates = pd.bdate_range("2020-01-02", periods=len(combined))
        returns = pd.Series(combined, index=dates)

        detector = VolatilityRegimeDetector()
        regime = detector.detect(returns)

        assert regime == VolatilityRegime.HIGH, (
            f"Expected HIGH volatility regime after crash, got {regime}"
        )

        # Verify percentile is high
        pct = detector.get_vol_percentile(returns)
        assert pct > 70, (
            f"Vol percentile should be > 70 during crash, got {pct:.1f}"
        )
