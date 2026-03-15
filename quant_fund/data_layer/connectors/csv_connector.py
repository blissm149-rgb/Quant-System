"""CSV file data connector.

Loads historical OHLCV data from local CSV files. Useful for offline
research, testing, and environments without internet access. Expects
CSV files with columns ``date, open, high, low, close, volume`` and
optionally ``ticker``.

No external dependencies beyond pandas.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from quant_fund.data_layer.connectors.base_connector import BaseConnector

logger = logging.getLogger(__name__)


class CSVConnector(BaseConnector):
    """Loads market data from local CSV files.

    Supports two directory layouts:
    1. **Per-ticker files**: ``<data_dir>/<TICKER>.csv``
    2. **Single file**: ``<data_dir>/all_data.csv`` with a ``ticker`` column

    Usage:
        connector = CSVConnector(config={"data_dir": "./data/csv"})
        df = connector.fetch_historical(
            tickers=["AAPL", "MSFT"],
            start_date=pd.Timestamp("2023-01-01"),
            end_date=pd.Timestamp("2024-01-01"),
        )
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._data_dir = Path(self._config.get("data_dir", "./data/csv"))

    def get_source_name(self) -> str:
        return "csv"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load historical bars from CSV files.

        Args:
            tickers: Ticker symbols to load.
            start_date: Start date (inclusive).
            end_date: End date (exclusive).
            fields: Columns to keep (default: OHLCV).

        Returns:
            DataFrame with MultiIndex ``(date, ticker)`` and OHLCV columns.
        """
        frames: list[pd.DataFrame] = []

        # Try single combined file first
        combined = self._data_dir / "all_data.csv"
        if combined.exists():
            df = self._load_combined(combined, tickers, start_date, end_date)
            if not df.empty:
                frames.append(df)
        else:
            # Per-ticker files
            for ticker in tickers:
                df = self._load_ticker_file(ticker, start_date, end_date)
                if not df.empty:
                    frames.append(df)

        if not frames:
            logger.warning("No CSV data found for %d tickers in %s", len(tickers), self._data_dir)
            return self._empty_ohlcv()

        result = pd.concat(frames)

        fields = fields or ["open", "high", "low", "close", "volume"]
        available = [f for f in fields if f in result.columns]
        if available:
            result = result[available]

        meta = self._track_ingestion(tickers, len(result), start_date, end_date)
        logger.info("CSV fetch complete: %s", meta)
        return result.sort_index()

    def fetch_latest(self, tickers: List[str]) -> pd.DataFrame:
        """Return the most recent bar per ticker from CSV files.

        Args:
            tickers: Ticker symbols to load.

        Returns:
            DataFrame with MultiIndex ``(date, ticker)`` and OHLCV columns.
        """
        # Load everything and take last row per ticker
        df = self.fetch_historical(
            tickers,
            start_date=pd.Timestamp("1900-01-01"),
            end_date=pd.Timestamp("2100-01-01"),
        )
        if df.empty:
            return df
        return df.groupby(level="ticker").tail(1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_ticker_file(
        self,
        ticker: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Load a single per-ticker CSV file."""
        candidates = [
            self._data_dir / f"{ticker}.csv",
            self._data_dir / f"{ticker.lower()}.csv",
            self._data_dir / f"{ticker.upper()}.csv",
        ]

        path = None
        for c in candidates:
            if c.exists():
                path = c
                break

        if path is None:
            logger.debug("No CSV file found for ticker %s", ticker)
            return self._empty_ohlcv()

        df = pd.read_csv(path, parse_dates=["date"])
        df.columns = [c.lower().strip() for c in df.columns]
        df["ticker"] = ticker
        df = df.set_index(["date", "ticker"])
        return self._filter_dates(df, start_date, end_date)

    def _load_combined(
        self,
        path: Path,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Load a combined CSV with a ticker column."""
        df = pd.read_csv(path, parse_dates=["date"])
        df.columns = [c.lower().strip() for c in df.columns]

        if "ticker" not in df.columns:
            logger.warning("Combined CSV at %s has no 'ticker' column", path)
            return self._empty_ohlcv()

        df = df[df["ticker"].isin(tickers)]
        df = df.set_index(["date", "ticker"])
        return self._filter_dates(df, start_date, end_date)

    @staticmethod
    def _filter_dates(
        df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if df.empty:
            return df
        dates = df.index.get_level_values("date")
        mask = (dates >= start_date) & (dates < end_date)
        return df[mask]
