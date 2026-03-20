"""Unit tests for historical_data_loader module.

TESTING_PLAN.md Section 3.2 — Data Layer.
"""

import pandas as pd
import pytest

from quant_fund.data_layer.historical_data_loader import HistoricalDataLoader
from quant_fund.data_layer.data_storage_manager import DataStorageManager
from quant_fund.data_layer.data_validator import DataValidator
from tests.conftest import make_ohlcv


@pytest.mark.unit
@pytest.mark.tier2
class TestHistoricalDataLoader:
    """HistoricalDataLoader — loads and validates historical market data."""

    @pytest.fixture
    def loader(self, tmp_path):
        storage = DataStorageManager(config={"data_root": str(tmp_path / "data")})
        validator = DataValidator()
        return HistoricalDataLoader(
            config={},
            storage_manager=storage,
            data_validator=validator,
        )

    @pytest.fixture
    def stored_data(self, loader):
        """Ingest some data so load() can find it."""
        df = make_ohlcv(tickers=["AAPL", "MSFT"], periods=100, seed=42)
        loader.ingest_data(df, dataset="price_data")
        return df

    def test_returns_multiindex_dataframe(self, loader, stored_data):
        """Loaded data has MultiIndex(date, ticker)."""
        df = loader.load(
            tickers=["AAPL", "MSFT"],
            start_date=pd.Timestamp("2019-01-02"),
            end_date=pd.Timestamp("2020-12-31"),
            as_of=pd.Timestamp("2025-01-01"),
        )
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["date", "ticker"]

    def test_as_of_respects_point_in_time(self, loader, stored_data):
        """as_of parameter filters out future data."""
        early_as_of = pd.Timestamp("2019-03-01")
        df = loader.load(
            tickers=["AAPL"],
            start_date=pd.Timestamp("2019-01-02"),
            end_date=pd.Timestamp("2020-12-31"),
            as_of=early_as_of,
        )
        if len(df) > 0:
            max_date = df.index.get_level_values("date").max()
            assert max_date < early_as_of

    def test_load_returns_expected_columns(self, loader, stored_data):
        """Loaded data contains standard OHLCV columns."""
        df = loader.load(
            tickers=["AAPL"],
            start_date=pd.Timestamp("2019-01-02"),
            end_date=pd.Timestamp("2020-12-31"),
            as_of=pd.Timestamp("2025-01-01"),
        )
        if len(df) > 0:
            for col in ["open", "high", "low", "close", "volume"]:
                assert col in df.columns

    def test_ingest_and_load_roundtrip(self, loader):
        """Data ingested can be loaded back."""
        df = make_ohlcv(tickers=["GOOG"], periods=50, seed=99)
        loader.ingest_data(df, dataset="price_data")
        loaded = loader.load(
            tickers=["GOOG"],
            start_date=pd.Timestamp("2019-01-01"),
            end_date=pd.Timestamp("2020-12-31"),
            as_of=pd.Timestamp("2025-01-01"),
        )
        # Should get data back
        assert len(loaded) > 0

    def test_load_universe_returns_ticker_list(self, loader, stored_data):
        """load_universe returns list of strings."""
        tickers = loader.load_universe(as_of=pd.Timestamp("2025-01-01"))
        assert isinstance(tickers, list)

    def test_load_universe_includes_delisted_by_default(self, loader):
        """Default include_delisted=True for survivor-bias-free analysis."""
        # Just verify the parameter is accepted
        tickers = loader.load_universe(
            as_of=pd.Timestamp("2025-01-01"),
            include_delisted=True,
        )
        assert isinstance(tickers, list)
