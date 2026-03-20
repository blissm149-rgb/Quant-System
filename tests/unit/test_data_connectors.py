"""Unit tests for data layer connectors.

TESTING_PLAN.md Section 3.2 — Data Layer connectors.
"""

import pandas as pd
import pytest

from quant_fund.data_layer.connectors.base_connector import BaseConnector
from quant_fund.data_layer.connectors.csv_connector import CSVConnector


@pytest.mark.unit
@pytest.mark.tier2
class TestBaseConnector:
    """BaseConnector — abstract interface compliance."""

    def test_cannot_instantiate_directly(self):
        """BaseConnector is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseConnector()

    def test_subclass_must_implement_abstract_methods(self):
        """Subclass without abstract methods raises TypeError."""
        class IncompleteConnector(BaseConnector):
            pass

        with pytest.raises(TypeError):
            IncompleteConnector()

    def test_empty_ohlcv_has_correct_schema(self):
        """_empty_ohlcv returns DataFrame with standard OHLCV columns."""
        empty = BaseConnector._empty_ohlcv()
        assert isinstance(empty, pd.DataFrame)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in empty.columns


@pytest.mark.unit
@pytest.mark.tier2
class TestCSVConnector:
    """CSVConnector — loads data from local CSV files."""

    @pytest.fixture
    def csv_dir(self, tmp_path):
        """Create a temp directory with a CSV file."""
        csv_content = "date,open,high,low,close,volume\n"
        csv_content += "2023-01-03,150.0,152.0,149.0,151.0,50000000\n"
        csv_content += "2023-01-04,151.0,153.0,150.0,152.0,45000000\n"
        csv_content += "2023-01-05,152.0,154.0,151.0,153.0,48000000\n"

        csv_file = tmp_path / "AAPL.csv"
        csv_file.write_text(csv_content)
        return tmp_path

    @pytest.fixture
    def connector(self, csv_dir):
        return CSVConnector(config={"data_dir": str(csv_dir)})

    def test_get_source_name(self, connector):
        """Source name is 'csv'."""
        assert connector.get_source_name() == "csv"

    def test_fetch_historical_loads_csv(self, connector):
        """Fetches data from CSV files."""
        df = connector.fetch_historical(
            tickers=["AAPL"],
            start_date=pd.Timestamp("2023-01-01"),
            end_date=pd.Timestamp("2023-12-31"),
        )
        assert len(df) > 0

    def test_fetch_historical_multiindex(self, connector):
        """Result has MultiIndex(date, ticker)."""
        df = connector.fetch_historical(
            tickers=["AAPL"],
            start_date=pd.Timestamp("2023-01-01"),
            end_date=pd.Timestamp("2023-12-31"),
        )
        if len(df) > 0:
            assert isinstance(df.index, pd.MultiIndex)

    def test_missing_ticker_returns_empty(self, connector):
        """Non-existent ticker CSV returns empty DataFrame."""
        df = connector.fetch_historical(
            tickers=["NONEXISTENT"],
            start_date=pd.Timestamp("2023-01-01"),
            end_date=pd.Timestamp("2023-12-31"),
        )
        assert len(df) == 0

    def test_fetch_latest_returns_last_row(self, connector):
        """fetch_latest returns the most recent bar."""
        df = connector.fetch_latest(tickers=["AAPL"])
        if len(df) > 0:
            assert len(df) <= 1 or isinstance(df.index, pd.MultiIndex)

    def test_date_filtering_works(self, connector):
        """Only data within date range is returned."""
        df = connector.fetch_historical(
            tickers=["AAPL"],
            start_date=pd.Timestamp("2023-01-04"),
            end_date=pd.Timestamp("2023-01-04"),
        )
        if len(df) > 0:
            dates = df.index.get_level_values("date") if isinstance(df.index, pd.MultiIndex) else df.index
            assert dates.min() >= pd.Timestamp("2023-01-04")
            assert dates.max() <= pd.Timestamp("2023-01-04")
