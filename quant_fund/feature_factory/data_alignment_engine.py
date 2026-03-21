"""Data alignment engine — the look-ahead bias enforcer.

All feature generators route through this module to obtain data that is
strictly point-in-time safe. Any attempt to access data at or after the
as_of timestamp raises a LookAheadError.
"""

import warnings
from typing import Optional

import pandas as pd
import yaml


class LookAheadError(Exception):
    """Raised when data contains timestamps at or after the as_of boundary."""


class DataAlignmentEngine:
    """Enforces point-in-time data alignment for all feature computations.

    This is the single enforcement point for look-ahead bias prevention.
    Every feature pipeline must obtain its input data through this engine.
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}

    @classmethod
    def from_config_file(cls, config_path: str) -> "DataAlignmentEngine":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config)

    def get_aligned_data(
        self,
        df: pd.DataFrame,
        as_of: pd.Timestamp,
        lookback_days: int,
    ) -> pd.DataFrame:
        """Return only rows where timestamp < as_of and within the lookback window.

        Args:
            df: DataFrame with date information in the index or columns.
                Supports MultiIndex (date, ticker), DatetimeIndex, or a 'date' column.
            as_of: Point-in-time boundary. All returned data will have
                timestamps strictly before this value.
            lookback_days: Maximum number of calendar days to look back from as_of.

        Returns:
            Filtered DataFrame containing only point-in-time safe data
            within the lookback window.

        Raises:
            LookAheadError: If the input data contains timestamps >= as_of,
                indicating a potential look-ahead bias violation upstream.
        """
        dates = self._extract_dates(df)

        self._check_no_future_data(dates, as_of)

        lookback_start = as_of - pd.Timedelta(days=lookback_days)

        mask = (dates < as_of) & (dates >= lookback_start)

        return df.loc[mask]

    def get_aligned_data_permissive(
        self,
        df: pd.DataFrame,
        as_of: pd.Timestamp,
        lookback_days: int,
    ) -> pd.DataFrame:
        """Like get_aligned_data but silently filters future data instead of raising.

        .. deprecated::
            Use :meth:`get_aligned_data` for strict point-in-time enforcement.
            This method silently masks look-ahead bias violations.
        """
        warnings.warn(
            "get_aligned_data_permissive() silently filters future data. "
            "Use get_aligned_data() for strict point-in-time enforcement.",
            DeprecationWarning,
            stacklevel=2,
        )
        dates = self._extract_dates(df)

        lookback_start = as_of - pd.Timedelta(days=lookback_days)
        mask = (dates < as_of) & (dates >= lookback_start)

        return df.loc[mask]

    def validate_no_lookahead(
        self, df: pd.DataFrame, as_of: pd.Timestamp
    ) -> bool:
        """Check that a DataFrame contains no data at or after as_of.

        Args:
            df: DataFrame to check.
            as_of: Point-in-time boundary.

        Returns:
            True if all data is strictly before as_of.
        """
        dates = self._extract_dates(df)
        return not (dates >= as_of).any()

    def _extract_dates(self, df: pd.DataFrame) -> pd.Series:
        """Extract date values as a Series aligned with df's index."""
        if isinstance(df.index, pd.MultiIndex):
            if "date" in df.index.names:
                level_values = df.index.get_level_values("date")
            else:
                level_values = df.index.get_level_values(0)
            return pd.Series(level_values, index=df.index)
        elif isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(df.index, index=df.index)
        elif "date" in df.columns:
            return df["date"]
        else:
            raise ValueError(
                "Cannot extract dates: DataFrame must have a DatetimeIndex, "
                "MultiIndex with 'date' level, or a 'date' column."
            )

    def _check_no_future_data(
        self, dates: pd.Series, as_of: pd.Timestamp
    ) -> None:
        """Raise LookAheadError if any dates are at or after as_of."""
        future_count = (dates >= as_of).sum()
        if future_count > 0:
            max_date = dates.max()
            raise LookAheadError(
                f"Data contains {future_count} rows with timestamps >= as_of "
                f"({as_of}). Latest timestamp: {max_date}. "
                f"This indicates a look-ahead bias risk."
            )
