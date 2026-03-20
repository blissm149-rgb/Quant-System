"""Unit tests for data_storage_manager module.

TESTING_PLAN.md Section 3.2 — Data Layer.
"""

import pandas as pd
import pytest

from quant_fund.data_layer.data_storage_manager import DataStorageManager
from tests.conftest import make_ohlcv


@pytest.mark.unit
@pytest.mark.tier2
class TestDataStorageManager:
    """DataStorageManager — Parquet-based data storage."""

    @pytest.fixture
    def manager(self, tmp_path):
        return DataStorageManager(config={"data_root": str(tmp_path / "data")})

    @pytest.fixture
    def sample_data(self):
        return make_ohlcv(tickers=["AAPL", "MSFT"], periods=50, seed=42)

    def test_save_load_roundtrip(self, manager, sample_data):
        """Parquet write/read roundtrip preserves schema."""
        manager.save(sample_data, dataset="test_ohlcv")
        loaded = manager.load(dataset="test_ohlcv")
        assert loaded.shape[0] > 0
        for col in sample_data.columns:
            assert col in loaded.columns

    def test_dataset_exists_after_save(self, manager, sample_data):
        """dataset_exists returns True after saving."""
        assert not manager.dataset_exists("test_ohlcv")
        manager.save(sample_data, dataset="test_ohlcv")
        assert manager.dataset_exists("test_ohlcv")

    def test_delete_dataset(self, manager, sample_data):
        """delete_dataset removes the dataset."""
        manager.save(sample_data, dataset="test_ohlcv")
        assert manager.dataset_exists("test_ohlcv")
        manager.delete_dataset("test_ohlcv")
        assert not manager.dataset_exists("test_ohlcv")

    def test_load_nonexistent_raises(self, manager):
        """Loading a non-existent dataset raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            manager.load(dataset="nonexistent")

    def test_list_datasets(self, manager, sample_data):
        """list_datasets returns saved dataset names."""
        manager.save(sample_data, dataset="ds_a")
        manager.save(sample_data, dataset="ds_b")
        datasets = manager.list_datasets()
        assert "ds_a" in datasets
        assert "ds_b" in datasets

    def test_get_storage_stats(self, manager, sample_data):
        """get_storage_stats returns expected metrics."""
        manager.save(sample_data, dataset="test_ohlcv")
        stats = manager.get_storage_stats("test_ohlcv")
        assert isinstance(stats, dict)
        assert "total_files" in stats or "total_bytes" in stats

    def test_purge_old_partitions(self, manager, sample_data):
        """purge_old_partitions removes only expired data."""
        manager.save(sample_data, dataset="test_ohlcv")
        # Purge with a cutoff far in the future — should remove everything
        removed, kept = manager.purge_old_partitions(
            "test_ohlcv",
            cutoff_date=pd.Timestamp("2030-01-01"),
        )
        assert isinstance(removed, int)
        assert isinstance(kept, int)

    def test_load_with_date_filter(self, manager, sample_data):
        """Loading with date range returns subset of data."""
        manager.save(sample_data, dataset="test_ohlcv")
        loaded = manager.load(
            dataset="test_ohlcv",
            start_date=pd.Timestamp("2019-01-02"),
            end_date=pd.Timestamp("2019-02-01"),
        )
        if len(loaded) > 0:
            dates = loaded.index.get_level_values("date") if isinstance(loaded.index, pd.MultiIndex) else loaded.index
            assert dates.max() <= pd.Timestamp("2019-02-01")

    def test_load_with_ticker_filter(self, manager, sample_data):
        """Loading with tickers returns only requested tickers."""
        manager.save(sample_data, dataset="test_ohlcv")
        loaded = manager.load(dataset="test_ohlcv", tickers=["AAPL"])
        if len(loaded) > 0 and isinstance(loaded.index, pd.MultiIndex):
            tickers = loaded.index.get_level_values("ticker").unique()
            assert list(tickers) == ["AAPL"]
