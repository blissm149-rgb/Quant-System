"""Historical data loader for OHLCV, fundamental, and reference data.

Loads point-in-time snapshots indexed by (date, ticker). Includes delisted
tickers to avoid survivor bias. All data is returned as-of the requested
time boundary.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from quant_fund.data_layer.corporate_action_adjuster import CorporateActionAdjuster
from quant_fund.data_layer.data_storage_manager import DataStorageManager
from quant_fund.data_layer.data_validator import DataValidator

logger = logging.getLogger(__name__)

DEFAULT_FIELDS = ["open", "high", "low", "close", "volume"]


class HistoricalDataLoader:
    """Loads historical market data with point-in-time integrity.

    Supports loading OHLCV price data, fundamental data, and reference data
    for a configurable universe. Includes delisted tickers for survivor-bias-free
    backtesting.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        storage_manager: Optional[DataStorageManager] = None,
        data_validator: Optional[DataValidator] = None,
        corporate_action_adjuster: Optional[CorporateActionAdjuster] = None,
    ):
        self._config = config or {}
        self._storage = storage_manager or DataStorageManager(config)
        self._validator = data_validator or DataValidator(config)
        self._adjuster = corporate_action_adjuster or CorporateActionAdjuster(config)

        self._data_source = self._config.get("data_sources", {}).get(
            "price_data", "polygon_io"
        )
        self._lookback_years = self._config.get("lookback_years", 10)

    @classmethod
    def from_config_file(cls, config_path: str) -> "HistoricalDataLoader":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config)

    def load(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fields: Optional[List[str]] = None,
        adjusted: bool = True,
        as_of: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Load historical data for the specified tickers and date range.

        Args:
            tickers: List of ticker symbols to load.
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (exclusive, acts as as_of if as_of not set).
            fields: List of fields to load (default: OHLCV).
            adjusted: If True, return corporate-action-adjusted prices.
            as_of: Point-in-time boundary for data availability. Defaults to end_date.

        Returns:
            DataFrame with MultiIndex (date, ticker) and requested fields as columns.

        Raises:
            ValueError: If validation fails with critical errors.
        """
        fields = fields or DEFAULT_FIELDS
        as_of = as_of or end_date

        category = DataStorageManager.ADJUSTED_SUBDIR if adjusted else DataStorageManager.RAW_SUBDIR
        dataset = "price_data"

        try:
            df = self._storage.load(
                dataset=dataset,
                category=category,
                start_date=start_date,
                end_date=as_of,
                tickers=tickers,
            )
        except FileNotFoundError:
            df = self._load_from_raw(tickers, start_date, as_of, adjusted)

        if df.empty:
            logger.warning(
                "No data loaded for %d tickers between %s and %s",
                len(tickers), start_date, end_date,
            )
            return df

        available_fields = [f for f in fields if f in df.columns]
        if available_fields:
            df = df[available_fields]

        validation = self._validator.validate(df, as_of=as_of)
        if not validation.is_valid:
            for error in validation.errors:
                logger.error("Validation error: %s", error)
            raise ValueError(
                f"Data validation failed with {len(validation.errors)} errors: "
                + "; ".join(validation.errors)
            )
        for warning in validation.warnings:
            logger.warning("Validation warning: %s", warning)

        return df

    def _load_from_raw(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        adjusted: bool,
    ) -> pd.DataFrame:
        """Load raw data and optionally adjust for corporate actions."""
        try:
            raw_df = self._storage.load(
                dataset="price_data",
                category=DataStorageManager.RAW_SUBDIR,
                start_date=start_date,
                end_date=end_date,
                tickers=tickers,
            )
        except FileNotFoundError:
            logger.warning("No raw data found for price_data")
            return pd.DataFrame()

        if raw_df.empty:
            return raw_df

        if adjusted:
            return self._adjuster.adjust_dataframe(raw_df)
        return raw_df

    def load_universe(
        self,
        as_of: pd.Timestamp,
        min_adv_usd: Optional[float] = None,
        min_price: Optional[float] = None,
        include_delisted: bool = True,
    ) -> List[str]:
        """Load the tradeable universe as of a specific date.

        Args:
            as_of: Point-in-time for universe membership.
            min_adv_usd: Minimum average daily volume in USD.
            min_price: Minimum stock price.
            include_delisted: Include delisted tickers (for survivor-bias-free backtest).

        Returns:
            List of ticker symbols in the universe.
        """
        liquidity_config = self._config.get("universe", {}).get("liquidity_filter", {})
        min_adv_usd = min_adv_usd or liquidity_config.get("min_adv_usd", 5_000_000)
        min_price = min_price or liquidity_config.get("min_price", 5.0)

        try:
            universe_df = self._storage.load(
                dataset="universe",
                category=DataStorageManager.METADATA_SUBDIR,
            )
        except FileNotFoundError:
            logger.warning("No universe metadata found")
            return []

        if universe_df.empty:
            return []

        dates = None
        if isinstance(universe_df.index, pd.MultiIndex) and "date" in universe_df.index.names:
            dates = universe_df.index.get_level_values("date")
        elif "date" in universe_df.columns:
            dates = universe_df["date"]

        if dates is not None:
            mask = dates < as_of
            universe_df = universe_df[mask]

        if not include_delisted and "is_active" in universe_df.columns:
            universe_df = universe_df[universe_df["is_active"]]

        if "adv_usd" in universe_df.columns:
            universe_df = universe_df[universe_df["adv_usd"] >= min_adv_usd]

        if "price" in universe_df.columns:
            universe_df = universe_df[universe_df["price"] >= min_price]

        if "ticker" in universe_df.columns:
            return sorted(universe_df["ticker"].unique().tolist())
        elif isinstance(universe_df.index, pd.MultiIndex) and "ticker" in universe_df.index.names:
            return sorted(universe_df.index.get_level_values("ticker").unique().tolist())

        return []

    def ingest_data(
        self,
        df: pd.DataFrame,
        dataset: str = "price_data",
        adjusted: bool = False,
    ) -> str:
        """Ingest external data into storage.

        Args:
            df: DataFrame to store. Should have MultiIndex (date, ticker) or
                columns 'date' and 'ticker'.
            dataset: Logical dataset name.
            adjusted: If True, store in adjusted category; otherwise raw.

        Returns:
            Storage path.
        """
        category = (
            DataStorageManager.ADJUSTED_SUBDIR
            if adjusted
            else DataStorageManager.RAW_SUBDIR
        )
        partition_col = None
        if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
            partition_col = "date"
        elif "date" in df.columns:
            partition_col = "date"

        return self._storage.save(
            df=df,
            dataset=dataset,
            category=category,
            partition_col=partition_col,
        )
