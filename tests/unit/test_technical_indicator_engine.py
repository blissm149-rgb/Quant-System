"""Unit tests for technical_indicator_engine module.

TESTING_PLAN.md Section 3.3 — Feature Factory.
"""

import pandas as pd
import pytest

from quant_fund.feature_factory.technical_indicator_engine import (
    BollingerBandPositionFeature,
    MomentumFeature,
    RelativeVolumeFeature,
    ReturnFeature,
    RSIFeature,
    ShortTermReversalFeature,
    TechnicalIndicatorEngine,
    VolatilityFeature,
)
from tests.conftest import make_ohlcv


@pytest.mark.unit
@pytest.mark.tier2
class TestTechnicalIndicatorEngine:
    """TechnicalIndicatorEngine — computes all technical features."""

    @pytest.fixture
    def engine(self):
        return TechnicalIndicatorEngine()

    @pytest.fixture
    def data(self):
        return make_ohlcv(tickers=["AAPL", "MSFT", "GOOG"], periods=300, seed=42)

    @pytest.fixture
    def as_of(self, data):
        return data.index.get_level_values("date").max() + pd.Timedelta(days=1)

    def test_compute_all_returns_dataframe(self, engine, data, as_of):
        """compute_all returns DataFrame indexed by ticker."""
        result = engine.compute_all(data, as_of=as_of)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_compute_all_has_features(self, engine, data, as_of):
        """Result contains multiple feature columns."""
        result = engine.compute_all(data, as_of=as_of)
        assert result.shape[1] > 0

    def test_generators_list(self, engine):
        """Engine has registered generators."""
        assert len(engine.generators) > 0


@pytest.mark.unit
@pytest.mark.tier2
class TestRSIFeature:
    """RSI must be bounded [0, 100]."""

    def test_rsi_bounded(self):
        """RSI output is in [0, 100] range."""
        data = make_ohlcv(tickers=["AAPL", "MSFT"], periods=100, seed=42)
        as_of = data.index.get_level_values("date").max() + pd.Timedelta(days=1)
        rsi = RSIFeature()
        result = rsi.compute(data, as_of=as_of)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.0
            assert valid.max() <= 100.0


@pytest.mark.unit
@pytest.mark.tier2
class TestVolatilityFeature:
    """Volatility must be non-negative."""

    def test_volatility_non_negative(self):
        """Volatility output is always >= 0."""
        data = make_ohlcv(tickers=["AAPL", "MSFT"], periods=100, seed=42)
        as_of = data.index.get_level_values("date").max() + pd.Timedelta(days=1)
        vol = VolatilityFeature()
        result = vol.compute(data, as_of=as_of)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.0


@pytest.mark.unit
@pytest.mark.tier2
class TestReturnFeature:
    """Return feature computed correctly."""

    def test_return_feature_produces_output(self):
        """Return feature computes for each ticker."""
        data = make_ohlcv(tickers=["AAPL", "MSFT"], periods=100, seed=42)
        as_of = data.index.get_level_values("date").max() + pd.Timedelta(days=1)
        ret = ReturnFeature(window_days=20)
        result = ret.compute(data, as_of=as_of)
        assert len(result) > 0
