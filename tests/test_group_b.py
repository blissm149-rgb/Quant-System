"""Tests for Group B — Feature pipeline.

Validates:
- DataAlignmentEngine enforces point-in-time (no look-ahead)
- Each technical indicator computes without future data
- FeatureNormalizer produces correct distributions
- No look-ahead bias in any feature computation
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator
from quant_fund.feature_factory.data_alignment_engine import (
    DataAlignmentEngine,
    LookAheadError,
)
from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    tickers: list[str],
    start: str = "2019-01-02",
    periods: int = 300,
) -> pd.DataFrame:
    """Create synthetic OHLCV with MultiIndex (date, ticker)."""
    dates = pd.bdate_range(start=start, periods=periods)
    rows = []
    rng = np.random.default_rng(42)
    for ticker in tickers:
        base_price = rng.uniform(20, 200)
        returns = rng.normal(0.0005, 0.02, size=periods)
        prices = base_price * np.cumprod(1 + returns)
        for i, dt in enumerate(dates):
            c = prices[i]
            rows.append({
                "date": dt,
                "ticker": ticker,
                "open": c * rng.uniform(0.99, 1.01),
                "high": c * rng.uniform(1.00, 1.03),
                "low": c * rng.uniform(0.97, 1.00),
                "close": c,
                "volume": int(rng.uniform(1e6, 1e7)),
            })
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df


# ---------------------------------------------------------------------------
# DataAlignmentEngine tests
# ---------------------------------------------------------------------------


class TestDataAlignmentEngine:

    def setup_method(self):
        self.engine = DataAlignmentEngine()

    def test_filters_future_data_strict(self):
        """get_aligned_data raises LookAheadError if future data exists."""
        df = _make_ohlcv(["AAPL"], periods=50)
        as_of = df.index.get_level_values("date")[25]
        with pytest.raises(LookAheadError):
            self.engine.get_aligned_data(df, as_of=as_of, lookback_days=365)

    def test_filters_future_data_permissive(self):
        """get_aligned_data_permissive silently removes future data."""
        df = _make_ohlcv(["AAPL"], periods=50)
        dates = df.index.get_level_values("date").unique()
        as_of = dates[25]
        result = self.engine.get_aligned_data_permissive(df, as_of=as_of, lookback_days=365)
        result_dates = result.index.get_level_values("date")
        assert (result_dates < as_of).all()

    def test_lookback_window(self):
        """Only data within lookback window is returned."""
        df = _make_ohlcv(["AAPL"], periods=100)
        dates = df.index.get_level_values("date").unique()
        as_of = dates[-1] + pd.Timedelta(days=1)
        result = self.engine.get_aligned_data(df, as_of=as_of, lookback_days=30)
        result_dates = result.index.get_level_values("date")
        min_allowed = as_of - pd.Timedelta(days=30)
        assert (result_dates >= min_allowed).all()
        assert (result_dates < as_of).all()

    def test_validate_no_lookahead_clean(self):
        """validate_no_lookahead returns True for clean data."""
        df = _make_ohlcv(["AAPL"], periods=10)
        as_of = pd.Timestamp("2025-01-01")
        assert self.engine.validate_no_lookahead(df, as_of=as_of)

    def test_validate_no_lookahead_dirty(self):
        """validate_no_lookahead returns False when future data present."""
        df = _make_ohlcv(["AAPL"], periods=10)
        dates = df.index.get_level_values("date")
        as_of = dates[5]
        assert not self.engine.validate_no_lookahead(df, as_of=as_of)


# ---------------------------------------------------------------------------
# Technical indicator tests
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]


class TestReturnFeature:

    def test_computes_returns(self):
        df = _make_ohlcv(TICKERS, periods=60)
        as_of = pd.Timestamp("2025-01-01")
        feat = ReturnFeature(window_days=20)
        result = feat.compute(df, as_of=as_of)
        assert len(result) == len(TICKERS)
        assert result.notna().all()
        assert feat.validate(result)

    def test_no_lookahead_in_return(self):
        """Inject future data; verify it doesn't affect the result."""
        df = _make_ohlcv(TICKERS, periods=60)
        dates = df.index.get_level_values("date").unique()
        as_of = dates[40]

        engine = DataAlignmentEngine()
        feat = ReturnFeature(window_days=20)

        aligned = engine.get_aligned_data_permissive(df, as_of=as_of, lookback_days=feat.lookback_days)
        result_clean = feat.compute(aligned, as_of=as_of)

        # Modify future data (after as_of)
        df_modified = df.copy()
        future_mask = df_modified.index.get_level_values("date") >= as_of
        df_modified.loc[future_mask, "close"] = 999999.0

        aligned_modified = engine.get_aligned_data_permissive(df_modified, as_of=as_of, lookback_days=feat.lookback_days)
        result_modified = feat.compute(aligned_modified, as_of=as_of)

        pd.testing.assert_series_equal(result_clean, result_modified)


class TestMomentumFeature:

    def test_computes_momentum(self):
        df = _make_ohlcv(TICKERS, periods=280)
        as_of = pd.Timestamp("2025-01-01")
        feat = MomentumFeature()
        result = feat.compute(df, as_of=as_of)
        assert len(result) == len(TICKERS)
        assert feat.validate(result)


class TestVolatilityFeature:

    def test_computes_volatility(self):
        df = _make_ohlcv(TICKERS, periods=60)
        as_of = pd.Timestamp("2025-01-01")
        feat = VolatilityFeature(window_days=20)
        result = feat.compute(df, as_of=as_of)
        assert len(result) == len(TICKERS)
        assert (result.dropna() > 0).all()
        assert feat.validate(result)


class TestRSIFeature:

    def test_computes_rsi(self):
        df = _make_ohlcv(TICKERS, periods=60)
        as_of = pd.Timestamp("2025-01-01")
        feat = RSIFeature(window_days=14)
        result = feat.compute(df, as_of=as_of)
        assert len(result) == len(TICKERS)
        # RSI should be between 0 and 100
        assert (result.dropna() >= 0).all()
        assert (result.dropna() <= 100).all()
        assert feat.validate(result)


class TestBollingerBandPositionFeature:

    def test_computes_bb_position(self):
        df = _make_ohlcv(TICKERS, periods=60)
        as_of = pd.Timestamp("2025-01-01")
        feat = BollingerBandPositionFeature(window_days=20)
        result = feat.compute(df, as_of=as_of)
        assert len(result) == len(TICKERS)
        assert feat.validate(result)


class TestShortTermReversalFeature:

    def test_computes_reversal(self):
        df = _make_ohlcv(TICKERS, periods=30)
        as_of = pd.Timestamp("2025-01-01")
        feat = ShortTermReversalFeature()
        result = feat.compute(df, as_of=as_of)
        assert len(result) == len(TICKERS)
        assert feat.validate(result)


class TestRelativeVolumeFeature:

    def test_computes_relative_volume(self):
        df = _make_ohlcv(TICKERS, periods=40)
        as_of = pd.Timestamp("2025-01-01")
        feat = RelativeVolumeFeature(window_days=20)
        result = feat.compute(df, as_of=as_of)
        assert len(result) == len(TICKERS)
        assert (result.dropna() > 0).all()
        assert feat.validate(result)


# ---------------------------------------------------------------------------
# TechnicalIndicatorEngine tests
# ---------------------------------------------------------------------------


class TestTechnicalIndicatorEngine:

    def test_compute_all(self):
        """Engine computes all indicators and returns a DataFrame."""
        engine = TechnicalIndicatorEngine()
        df = _make_ohlcv(TICKERS, periods=280)
        dates = df.index.get_level_values("date").unique()
        as_of = dates[-1] + pd.Timedelta(days=1)
        result = engine.compute_all(df, as_of=as_of)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(TICKERS)
        assert len(result.columns) > 5  # should have multiple indicators

    def test_no_lookahead_full_engine(self):
        """Changing future data should not affect engine output."""
        engine = TechnicalIndicatorEngine()
        df = _make_ohlcv(TICKERS, periods=280)
        dates = df.index.get_level_values("date").unique()
        as_of = dates[250]

        # Pre-filter to only past data (strict mode rejects data >= as_of)
        past_mask = df.index.get_level_values("date") < as_of
        df_past = df.loc[past_mask]

        result1 = engine.compute_all(df_past, as_of=as_of)

        # Modify future data and filter again — result should be identical
        df_mod = df.copy()
        future_mask = df_mod.index.get_level_values("date") >= as_of
        df_mod.loc[future_mask, "close"] = 999999.0
        df_mod_past = df_mod.loc[df_mod.index.get_level_values("date") < as_of]
        result2 = engine.compute_all(df_mod_past, as_of=as_of)

        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# FeatureNormalizer tests
# ---------------------------------------------------------------------------


class TestFeatureNormalizer:

    def setup_method(self):
        self.normalizer = FeatureNormalizer()

    def test_zscore_mean_zero(self):
        """Z-scored features should have mean ~0 and std ~1."""
        rng = np.random.default_rng(42)
        raw = pd.Series(rng.normal(50, 10, 100), index=[f"T{i}" for i in range(100)])
        result = self.normalizer.normalize(raw, method="zscore")
        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.1

    def test_zscore_winsorisation(self):
        """Outliers beyond winsorize_std should be clipped."""
        raw = pd.Series([1, 2, 3, 4, 5, 100], index=[f"T{i}" for i in range(6)])
        result = self.normalizer.normalize(raw, method="zscore", winsorize_std=2.0)
        # The extreme outlier (100) should be clipped, so max z-score is bounded
        assert result.max() < 5.0

    def test_rank_normalize(self):
        """Rank normalisation produces values in [-1, 1]."""
        raw = pd.Series([10, 20, 30, 40, 50], index=[f"T{i}" for i in range(5)])
        result = self.normalizer.normalize(raw, method="rank")
        assert result.min() == pytest.approx(-1.0)
        assert result.max() == pytest.approx(1.0)

    def test_percentile_normalize(self):
        """Percentile normalisation produces values in [0, 1]."""
        raw = pd.Series([10, 20, 30, 40, 50], index=[f"T{i}" for i in range(5)])
        result = self.normalizer.normalize(raw, method="percentile")
        assert result.min() == pytest.approx(0.2)
        assert result.max() == pytest.approx(1.0)

    def test_nan_handling(self):
        """NaN values should not break normalisation."""
        raw = pd.Series([1, 2, np.nan, 4, 5], index=[f"T{i}" for i in range(5)])
        result = self.normalizer.normalize(raw, method="zscore")
        assert result.notna().sum() >= 4
        assert np.isnan(result.iloc[2]) or result.notna().all()

    def test_normalize_dataframe(self):
        """DataFrame normalisation applies to each column independently."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "feat_a": rng.normal(0, 1, 50),
            "feat_b": rng.normal(100, 20, 50),
        }, index=[f"T{i}" for i in range(50)])

        result = self.normalizer.normalize_dataframe(df, method="zscore")
        assert result.shape == df.shape
        for col in result.columns:
            assert abs(result[col].mean()) < 0.1
            assert abs(result[col].std() - 1.0) < 0.15

    def test_unknown_method_raises(self):
        raw = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown"):
            self.normalizer.normalize(raw, method="invalid")
