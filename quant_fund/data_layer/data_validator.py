"""Data validation module enforcing data quality and point-in-time integrity.

Validates every data batch before it enters the feature pipeline. Checks for
look-ahead bias, missing data, invalid prices, and duplicate records.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import yaml


@dataclass
class ValidationResult:
    """Result of a data validation pass."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    offending_rows: Optional[pd.DataFrame] = None

    def __bool__(self) -> bool:
        return self.is_valid


class DataValidator:
    """Validates data batches for quality, consistency, and point-in-time integrity.

    All data must pass validation before entering the feature pipeline.
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._max_nan_ratio = self._config.get("max_nan_ratio", 0.05)
        self._min_price = self._config.get("min_price", 0.0)

    @classmethod
    def from_config_file(cls, config_path: str) -> "DataValidator":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config.get("validation", {}))

    def validate(self, df: pd.DataFrame, as_of: pd.Timestamp) -> ValidationResult:
        """Run all validation checks on the data batch.

        Args:
            df: DataFrame with a 'date' column (or DatetimeIndex level) and
                price/volume columns. Expected MultiIndex (date, ticker) or
                columns including 'date' and 'ticker'.
            as_of: Point-in-time boundary. No data at or after this timestamp
                is permitted.

        Returns:
            ValidationResult with pass/fail and details of any violations.
        """
        errors: List[str] = []
        warnings: List[str] = []
        offending_frames: List[pd.DataFrame] = []

        dates = self._extract_dates(df)
        if dates is None:
            return ValidationResult(
                is_valid=False,
                errors=["Cannot extract date index from DataFrame"],
            )

        self._check_no_future_timestamps(df, dates, as_of, errors, offending_frames)
        self._check_no_nan_prices(df, errors, warnings, offending_frames)
        self._check_no_negative_prices(df, errors, offending_frames)
        self._check_positive_volume(df, warnings, offending_frames)
        self._check_no_duplicates(df, errors, offending_frames)

        offending = pd.concat(offending_frames) if offending_frames else None
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            offending_rows=offending,
        )

    def _extract_dates(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract the date series from a DataFrame, handling various index formats."""
        if isinstance(df.index, pd.MultiIndex):
            if "date" in df.index.names:
                return df.index.get_level_values("date")
            return df.index.get_level_values(0)
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.to_series()
        if "date" in df.columns:
            return df["date"]
        return None

    def _check_no_future_timestamps(
        self,
        df: pd.DataFrame,
        dates: pd.Series,
        as_of: pd.Timestamp,
        errors: List[str],
        offending: List[pd.DataFrame],
    ) -> None:
        """No timestamps at or after as_of (look-ahead guard)."""
        future_mask = dates >= as_of
        if future_mask.any():
            n_future = future_mask.sum()
            errors.append(
                f"Look-ahead violation: {n_future} rows have timestamps >= as_of ({as_of})"
            )
            offending.append(df.loc[future_mask] if not isinstance(df.index, pd.MultiIndex) else df[future_mask])

    def _check_no_nan_prices(
        self,
        df: pd.DataFrame,
        errors: List[str],
        warnings: List[str],
        offending: List[pd.DataFrame],
    ) -> None:
        """Close prices must not be NaN. Other price columns raise warnings."""
        price_cols = [c for c in df.columns if c in ("close", "open", "high", "low")]
        if "close" in df.columns:
            close_nan_mask = df["close"].isna()
            if close_nan_mask.any():
                n_nan = close_nan_mask.sum()
                errors.append(f"NaN in close prices: {n_nan} rows")
                offending.append(df[close_nan_mask])

        for col in price_cols:
            if col == "close":
                continue
            nan_ratio = df[col].isna().mean()
            if nan_ratio > self._max_nan_ratio:
                warnings.append(
                    f"High NaN ratio in {col}: {nan_ratio:.2%} (threshold: {self._max_nan_ratio:.2%})"
                )

    def _check_no_negative_prices(
        self,
        df: pd.DataFrame,
        errors: List[str],
        offending: List[pd.DataFrame],
    ) -> None:
        """No negative prices in any price column."""
        price_cols = [c for c in df.columns if c in ("close", "open", "high", "low")]
        for col in price_cols:
            col_data = df[col].dropna()
            neg_mask = col_data < 0
            if neg_mask.any():
                n_neg = neg_mask.sum()
                errors.append(f"Negative values in {col}: {n_neg} rows")
                offending.append(df.loc[col_data[neg_mask].index])

    def _check_positive_volume(
        self,
        df: pd.DataFrame,
        warnings: List[str],
        offending: List[pd.DataFrame],
    ) -> None:
        """Volume should be > 0 on trading days."""
        if "volume" not in df.columns:
            return
        zero_vol_mask = df["volume"] <= 0
        zero_vol_data = df["volume"].dropna()
        zero_mask = zero_vol_data <= 0
        if zero_mask.any():
            n_zero = zero_mask.sum()
            ratio = n_zero / len(zero_vol_data)
            if ratio > self._max_nan_ratio:
                warnings.append(
                    f"Zero/negative volume: {n_zero} rows ({ratio:.2%})"
                )

    def _check_no_duplicates(
        self,
        df: pd.DataFrame,
        errors: List[str],
        offending: List[pd.DataFrame],
    ) -> None:
        """No duplicate (date, ticker) rows."""
        if isinstance(df.index, pd.MultiIndex):
            dup_mask = df.index.duplicated(keep=False)
            if dup_mask.any():
                n_dup = dup_mask.sum()
                errors.append(f"Duplicate index entries: {n_dup} rows")
                offending.append(df[dup_mask])
        elif "date" in df.columns and "ticker" in df.columns:
            dup_mask = df.duplicated(subset=["date", "ticker"], keep=False)
            if dup_mask.any():
                n_dup = dup_mask.sum()
                errors.append(f"Duplicate (date, ticker) rows: {n_dup}")
                offending.append(df[dup_mask])
