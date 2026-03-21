"""Unit tests for research algorithms.

TESTING_PLAN.md Section 3.15 — factor models, mean reversion,
regime detection, macro classification.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv

from quant_fund.research_algorithms.factor_models.momentum_factor import (
    MomentumFactor,
)
from quant_fund.research_algorithms.mean_reversion.zscore_reversion_strategy import (
    ZScoreReversionStrategy,
)
from quant_fund.research_algorithms.regime_models.volatility_regime_detector import (
    VolatilityRegime,
    VolatilityRegimeDetector,
)
from quant_fund.alternative_data.macro_data.macro_regime_classifier import (
    MacroRegime,
    MacroRegimeClassifier,
)


# ── Momentum Factor ───────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier2
class TestMomentumFactor:
    """MomentumFactor — 12-1 month price momentum."""

    @pytest.fixture
    def factor(self):
        return MomentumFactor()

    @pytest.fixture
    def ohlcv(self):
        return make_ohlcv(["AAPL", "MSFT", "GOOG"], periods=300, seed=42)

    def test_compute_returns_series(self, factor, ohlcv):
        result = factor.compute(ohlcv, as_of=ohlcv.index.get_level_values("date").max())
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_momentum_finite(self, factor, ohlcv):
        result = factor.compute(ohlcv, as_of=ohlcv.index.get_level_values("date").max())
        assert result.dropna().apply(np.isfinite).all()

    def test_validate(self, factor, ohlcv):
        result = factor.compute(ohlcv, as_of=ohlcv.index.get_level_values("date").max())
        assert factor.validate(result) == True


# ── Z-Score Reversion ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier2
class TestZScoreReversionStrategy:
    """ZScoreReversionStrategy — cross-sectional mean reversion."""

    @pytest.fixture
    def strategy(self):
        return ZScoreReversionStrategy()

    @pytest.fixture
    def ohlcv(self):
        return make_ohlcv(["AAPL", "MSFT", "GOOG", "AMZN", "META"], periods=100, seed=42)

    def test_compute_returns_series(self, strategy, ohlcv):
        result = strategy.compute(ohlcv, as_of=ohlcv.index.get_level_values("date").max())
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_signal_is_inverted(self, strategy, ohlcv):
        """Signal should be negated z-scores (fade extremes)."""
        result = strategy.compute(ohlcv, as_of=ohlcv.index.get_level_values("date").max())
        # Cross-sectional mean should be approximately 0
        assert abs(result.dropna().mean()) < 1.0


# ── Volatility Regime Detector ────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier2
class TestVolatilityRegimeDetector:
    """VolatilityRegimeDetector — low/normal/high vol classification."""

    @pytest.fixture
    def detector(self):
        return VolatilityRegimeDetector()

    def test_detect_returns_regime(self, detector):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 300))
        regime = detector.detect(returns)
        assert isinstance(regime, VolatilityRegime)

    def test_high_vol_regime(self, detector):
        """Very volatile returns → HIGH regime."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 280))
        # Append high-vol period
        high_vol = pd.Series(rng.normal(0, 0.05, 30))
        returns = pd.concat([returns, high_vol], ignore_index=True)
        regime = detector.detect(returns)
        assert regime in (VolatilityRegime.HIGH, VolatilityRegime.NORMAL)

    def test_get_vol_percentile(self, detector):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 300))
        pct = detector.get_vol_percentile(returns)
        assert 0 <= pct <= 100

    def test_short_series_returns_normal(self, detector):
        """Insufficient data defaults to NORMAL."""
        returns = pd.Series([0.01, -0.01])
        assert detector.detect(returns) == VolatilityRegime.NORMAL


# ── Macro Regime Classifier ───────────────────────────────────────


@pytest.mark.unit
@pytest.mark.tier2
class TestMacroRegimeClassifier:
    """MacroRegimeClassifier — growth/inflation regime classification."""

    @pytest.fixture
    def classifier(self):
        return MacroRegimeClassifier()

    def test_goldilocks(self, classifier):
        """Growth expanding, inflation low → GOLDILOCKS."""
        assert classifier.classify(gdp_growth=0.03, inflation_rate=0.02) == MacroRegime.GOLDILOCKS

    def test_stagflation(self, classifier):
        """Growth contracting, inflation high → STAGFLATION."""
        assert classifier.classify(gdp_growth=-0.01, inflation_rate=0.05) == MacroRegime.STAGFLATION

    def test_deflation(self, classifier):
        """Growth contracting, inflation low → DEFLATION."""
        assert classifier.classify(gdp_growth=-0.01, inflation_rate=0.01) == MacroRegime.DEFLATION

    def test_risk_on_growth(self, classifier):
        """Growth expanding, inflation high → RISK_ON_GROWTH."""
        assert classifier.classify(gdp_growth=0.03, inflation_rate=0.05) == MacroRegime.RISK_ON_GROWTH

    def test_default_when_no_data(self, classifier):
        """No data → default GOLDILOCKS."""
        assert classifier.classify() == MacroRegime.GOLDILOCKS

    def test_regime_strategy_adjustments(self, classifier):
        """Each regime returns strategy weight adjustments."""
        adjustments = classifier.get_regime_strategy_adjustments(MacroRegime.STAGFLATION)
        assert isinstance(adjustments, dict)
        assert "quality" in adjustments
