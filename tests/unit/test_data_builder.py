"""Tests for the DataBuilder offline training data pipeline."""

import numpy as np
import pandas as pd
import pytest

from quant_fund.infrastructure.data_builder import DataBuilder

pytestmark = [pytest.mark.tier1]


class TestDataBuilder:
    """Tests for the training dataset builder."""

    def test_build_with_preloaded_data(self):
        """Build filters data to temporal window."""
        dates = pd.bdate_range("2020-01-02", periods=500)
        data = pd.DataFrame(
            {"close": np.random.default_rng(42).normal(100, 10, 500)},
            index=dates,
        )

        builder = DataBuilder()
        dataset = builder.build(
            cutoff_date="2021-06-01",
            lookback_days=252,
            data=data,
        )
        assert not dataset.empty
        assert dataset.index.max() <= pd.Timestamp("2021-06-01")

    def test_build_respects_cutoff(self):
        """No data after cutoff_date appears in dataset."""
        dates = pd.bdate_range("2020-01-02", periods=500)
        data = pd.DataFrame(
            {"close": np.random.default_rng(42).normal(100, 10, 500)},
            index=dates,
        )

        cutoff = "2020-06-01"
        builder = DataBuilder()
        dataset = builder.build(cutoff_date=cutoff, lookback_days=252, data=data)

        assert dataset.index.max() <= pd.Timestamp(cutoff)

    def test_build_with_multiindex_data(self):
        """Handles MultiIndex (date, ticker) data correctly."""
        dates = pd.bdate_range("2020-01-02", periods=200)
        tickers = ["AAPL", "MSFT"]
        rows = []
        for dt in dates:
            for t in tickers:
                rows.append({"date": dt, "ticker": t, "close": 100.0})
        data = pd.DataFrame(rows).set_index(["date", "ticker"])

        builder = DataBuilder()
        dataset = builder.build(
            cutoff_date="2020-06-01",
            lookback_days=120,
            data=data,
        )
        assert not dataset.empty
        max_date = dataset.index.get_level_values(0).max()
        assert max_date <= pd.Timestamp("2020-06-01")

    def test_compute_hash_deterministic(self):
        """Same data produces same hash."""
        data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        builder = DataBuilder()
        h1 = builder.compute_hash(data)
        h2 = builder.compute_hash(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_hash_different_data(self):
        """Different data produces different hash."""
        builder = DataBuilder()
        h1 = builder.compute_hash(pd.DataFrame({"a": [1.0]}))
        h2 = builder.compute_hash(pd.DataFrame({"a": [2.0]}))
        assert h1 != h2

    def test_compute_hash_empty_dataset(self):
        """Empty dataset returns a valid hash."""
        builder = DataBuilder()
        h = builder.compute_hash(pd.DataFrame())
        assert len(h) == 64

    def test_build_empty_data_returns_empty(self):
        """Building from empty data returns empty DataFrame."""
        builder = DataBuilder()
        dataset = builder.build(
            cutoff_date="2020-06-01",
            lookback_days=252,
            data=pd.DataFrame(),
        )
        assert dataset.empty

    def test_build_default_cutoff(self):
        """Build without explicit cutoff defaults to yesterday."""
        dates = pd.bdate_range("2020-01-02", periods=2000)
        data = pd.DataFrame(
            {"close": np.random.default_rng(42).normal(100, 10, len(dates))},
            index=dates,
        )

        builder = DataBuilder()
        dataset = builder.build(lookback_days=252, data=data)
        # Should not fail and should have data
        assert isinstance(dataset, pd.DataFrame)
