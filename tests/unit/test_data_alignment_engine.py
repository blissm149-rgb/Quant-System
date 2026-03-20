"""Unit tests for data_alignment_engine module.

TESTING_PLAN.md Section 3.3 — CRITICAL: look-ahead bias enforcer.
"""

import pandas as pd
import pytest

from quant_fund.feature_factory.data_alignment_engine import (
    DataAlignmentEngine,
    LookAheadError,
)
from tests.conftest import make_ohlcv


@pytest.mark.unit
@pytest.mark.tier1
class TestDataAlignmentEngine:
    """DataAlignmentEngine — prevents look-ahead bias."""

    @pytest.fixture
    def engine(self):
        return DataAlignmentEngine()

    @pytest.fixture
    def data(self):
        return make_ohlcv(tickers=["AAPL", "MSFT"], periods=100, seed=42)

    def test_strict_mode_raises_on_future_data(self, engine, data):
        """get_aligned_data raises LookAheadError when data has future timestamps."""
        dates = data.index.get_level_values("date").unique()
        as_of = dates[50]  # middle of data — remaining data is "future"
        with pytest.raises(LookAheadError):
            engine.get_aligned_data(data, as_of=as_of, lookback_days=100)

    def test_permissive_mode_removes_future_data(self, engine, data):
        """get_aligned_data_permissive silently filters future data."""
        dates = data.index.get_level_values("date").unique()
        as_of = dates[50]
        aligned = engine.get_aligned_data_permissive(data, as_of=as_of, lookback_days=100)
        max_date = aligned.index.get_level_values("date").max()
        assert max_date < as_of

    def test_as_of_exactly_on_boundary(self, engine, data):
        """Edge case: as_of matches a data timestamp exactly."""
        dates = data.index.get_level_values("date").unique()
        as_of = dates[50]
        aligned = engine.get_aligned_data_permissive(data, as_of=as_of, lookback_days=100)
        max_date = aligned.index.get_level_values("date").max()
        assert max_date < as_of

    def test_validate_no_lookahead_clean_data(self, engine, data):
        """validate_no_lookahead returns True for data entirely before as_of."""
        as_of = pd.Timestamp("2025-01-01")
        assert engine.validate_no_lookahead(data, as_of=as_of)

    def test_validate_no_lookahead_dirty_data(self, engine, data):
        """validate_no_lookahead returns False for data with future timestamps."""
        dates = data.index.get_level_values("date").unique()
        as_of = dates[50]
        assert not engine.validate_no_lookahead(data, as_of=as_of)

    def test_lookback_window_limits_data(self, engine, data):
        """Lookback window truncates old data."""
        as_of = pd.Timestamp("2025-01-01")
        aligned = engine.get_aligned_data_permissive(data, as_of=as_of, lookback_days=30)
        dates = aligned.index.get_level_values("date").unique()
        if len(dates) > 0:
            span = (dates.max() - dates.min()).days
            assert span <= 60  # some slack for business days

    def test_empty_result_for_no_matching_data(self, engine, data):
        """Returns empty DataFrame when no data falls in window."""
        as_of = pd.Timestamp("2010-01-01")  # way before data starts
        aligned = engine.get_aligned_data_permissive(data, as_of=as_of, lookback_days=30)
        assert len(aligned) == 0

    def test_preserves_multiindex_format(self, engine, data):
        """Aligned data keeps MultiIndex(date, ticker)."""
        as_of = pd.Timestamp("2025-01-01")
        aligned = engine.get_aligned_data_permissive(data, as_of=as_of, lookback_days=300)
        assert isinstance(aligned.index, pd.MultiIndex)
        assert aligned.index.names == ["date", "ticker"]
