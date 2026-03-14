"""Data storage manager for Parquet-based data persistence.

Handles reading and writing of market data in Parquet format with
date-based partitioning. Raw data is never modified; adjusted series
are stored separately.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataStorageManager:
    """Manages persistent storage of market and feature data in Parquet format.

    Supports date-based partitioning and maintains separation between raw
    and adjusted data series.
    """

    RAW_SUBDIR = "raw"
    ADJUSTED_SUBDIR = "adjusted"
    FEATURES_SUBDIR = "features"
    METADATA_SUBDIR = "metadata"

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._data_root = Path(self._config.get("data_root", "./data"))
        self._format = self._config.get("format", "parquet")
        self._partitioning = self._config.get("partitioning", "by_date")
        self._lookback_years = self._config.get("lookback_years", 10)

    @classmethod
    def from_config_file(cls, config_path: str) -> "DataStorageManager":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        storage_config = config.get("storage", {})
        storage_config["data_root"] = config.get("data_root", "./data")
        return cls(config=storage_config)

    def _ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def _dataset_path(self, dataset: str, category: str = RAW_SUBDIR) -> Path:
        return self._data_root / category / dataset

    def save(
        self,
        df: pd.DataFrame,
        dataset: str,
        category: str = RAW_SUBDIR,
        partition_col: Optional[str] = None,
    ) -> str:
        """Save a DataFrame to storage.

        Args:
            df: Data to persist.
            dataset: Logical dataset name (e.g., "price_data", "fundamentals").
            category: Storage category — "raw", "adjusted", "features", "metadata".
            partition_col: Column name to partition by (default: date-based).

        Returns:
            Path where data was written.
        """
        base_path = self._dataset_path(dataset, category)
        self._ensure_dir(base_path)

        if self._partitioning == "by_date" and partition_col:
            return self._save_partitioned(df, base_path, partition_col)
        else:
            file_path = base_path / f"{dataset}.parquet"
            df.to_parquet(file_path, engine="pyarrow", index=True)
            logger.info("Saved %d rows to %s", len(df), file_path)
            return str(file_path)

    def _save_partitioned(
        self, df: pd.DataFrame, base_path: Path, partition_col: str
    ) -> str:
        """Save data partitioned by a date column."""
        if partition_col in df.columns:
            dates = df[partition_col]
        elif isinstance(df.index, pd.MultiIndex) and partition_col in df.index.names:
            dates = df.index.get_level_values(partition_col)
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        else:
            file_path = base_path / "data.parquet"
            df.to_parquet(file_path, engine="pyarrow", index=True)
            return str(file_path)

        unique_dates = pd.Series(dates).dt.date.unique()
        for dt in unique_dates:
            date_str = str(dt)
            partition_path = base_path / f"date={date_str}"
            self._ensure_dir(partition_path)

            if isinstance(dates, pd.DatetimeIndex) or hasattr(dates, "dt"):
                mask = pd.Series(dates).dt.date == dt
                if isinstance(df.index, pd.MultiIndex):
                    mask = mask.values
            else:
                mask = dates.dt.date == dt

            partition_df = df[mask]
            file_path = partition_path / "data.parquet"
            partition_df.to_parquet(file_path, engine="pyarrow", index=True)

        logger.info("Saved %d rows across %d partitions to %s", len(df), len(unique_dates), base_path)
        return str(base_path)

    def load(
        self,
        dataset: str,
        category: str = RAW_SUBDIR,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load data from storage.

        Args:
            dataset: Logical dataset name.
            category: Storage category.
            start_date: Filter rows on or after this date.
            end_date: Filter rows before this date.
            tickers: Filter to these tickers only.

        Returns:
            DataFrame with loaded data.
        """
        base_path = self._dataset_path(dataset, category)

        if not base_path.exists():
            raise FileNotFoundError(f"Dataset not found: {base_path}")

        single_file = base_path / f"{dataset}.parquet"
        if single_file.exists():
            df = pd.read_parquet(single_file, engine="pyarrow")
        else:
            df = self._load_partitioned(base_path, start_date, end_date)

        if df.empty:
            return df

        df = self._apply_filters(df, start_date, end_date, tickers)
        logger.info("Loaded %d rows from %s/%s", len(df), category, dataset)
        return df

    def _load_partitioned(
        self,
        base_path: Path,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        """Load data from date-partitioned directories."""
        parquet_files = sorted(base_path.rglob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        frames = []
        for pf in parquet_files:
            date_part = None
            for part in pf.parts:
                if part.startswith("date="):
                    date_part = part.replace("date=", "")
                    break

            if date_part and start_date is not None:
                part_date = pd.Timestamp(date_part)
                if part_date < start_date:
                    continue
            if date_part and end_date is not None:
                part_date = pd.Timestamp(date_part)
                if part_date >= end_date:
                    continue

            frames.append(pd.read_parquet(pf, engine="pyarrow"))

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames)

    def _apply_filters(
        self,
        df: pd.DataFrame,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
        tickers: Optional[List[str]],
    ) -> pd.DataFrame:
        """Apply date and ticker filters to loaded data."""
        dates = None
        ticker_vals = None

        if isinstance(df.index, pd.MultiIndex):
            if "date" in df.index.names:
                dates = df.index.get_level_values("date")
            if "ticker" in df.index.names:
                ticker_vals = df.index.get_level_values("ticker")
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index

        if dates is None and "date" in df.columns:
            dates = df["date"]
        if ticker_vals is None and "ticker" in df.columns:
            ticker_vals = df["ticker"]

        mask = pd.Series(True, index=df.index)

        if dates is not None and start_date is not None:
            mask &= dates >= start_date
        if dates is not None and end_date is not None:
            mask &= dates < end_date
        if ticker_vals is not None and tickers is not None:
            mask &= ticker_vals.isin(tickers)

        return df[mask]

    def list_datasets(self, category: str = RAW_SUBDIR) -> List[str]:
        """List available datasets in a category."""
        cat_path = self._data_root / category
        if not cat_path.exists():
            return []
        return [d.name for d in cat_path.iterdir() if d.is_dir()]

    def dataset_exists(self, dataset: str, category: str = RAW_SUBDIR) -> bool:
        """Check if a dataset exists."""
        return self._dataset_path(dataset, category).exists()

    def delete_dataset(self, dataset: str, category: str = RAW_SUBDIR) -> None:
        """Delete a dataset and all its partitions."""
        import shutil
        path = self._dataset_path(dataset, category)
        if path.exists():
            shutil.rmtree(path)
            logger.info("Deleted dataset: %s/%s", category, dataset)
