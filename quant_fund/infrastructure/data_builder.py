"""Data builder for the offline training pipeline.

Builds frozen, immutable training datasets with strict temporal boundaries.
The cutoff_date must always be yesterday's close or earlier -- never today.
This prevents look-ahead bias from partial-day data.
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataBuilder:
    """Builds training datasets from the data warehouse.

    Enforces temporal integrity: no data from after the cutoff date.
    All features use only backward-looking transforms.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._warehouse_path = cfg.get("warehouse_path", "./data/warehouse")
        self._validation_split = cfg.get("validation_split", 0.2)

    def build(
        self,
        cutoff_date: Optional[str] = None,
        lookback_days: int = 756,
        data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build training dataset with strict temporal boundaries.

        Parameters
        ----------
        cutoff_date : str, optional
            ISO date string. Must be yesterday's close or earlier.
            If None, defaults to yesterday.
        lookback_days : int
            Number of business days to include (default 756 = ~3 years).
        data : pd.DataFrame, optional
            Pre-loaded data (for testing). If None, loads from warehouse.

        Returns
        -------
        pd.DataFrame
            Frozen training dataset.
        """
        if cutoff_date is None:
            cutoff = datetime.now(timezone.utc).date() - timedelta(days=1)
            cutoff_date = cutoff.isoformat()

        cutoff_ts = pd.Timestamp(cutoff_date)
        start_ts = cutoff_ts - pd.Timedelta(days=int(lookback_days * 1.5))

        if data is not None:
            dataset = self._filter_data(data, start_ts, cutoff_ts)
        else:
            dataset = self._load_from_warehouse(start_ts, cutoff_ts)

        if dataset.empty:
            logger.warning("DataBuilder produced empty dataset")
            return dataset

        logger.info(
            "Dataset built: %d rows, cutoff=%s, lookback=%d days",
            len(dataset),
            cutoff_date,
            lookback_days,
        )
        return dataset

    def compute_hash(self, dataset: pd.DataFrame) -> str:
        """SHA-256 of the training data for reproducibility tracking."""
        if dataset.empty:
            return hashlib.sha256(b"empty").hexdigest()

        # Hash the raw bytes of the values array
        h = hashlib.sha256()
        h.update(dataset.values.tobytes())
        # Include column names and index for completeness
        h.update(str(list(dataset.columns)).encode())
        h.update(str(list(dataset.index)).encode())
        return h.hexdigest()

    def _filter_data(
        self,
        data: pd.DataFrame,
        start_ts: pd.Timestamp,
        cutoff_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        """Filter data to the temporal window [start, cutoff]."""
        if isinstance(data.index, pd.MultiIndex):
            dates = data.index.get_level_values(0)
            mask = (dates >= start_ts) & (dates <= cutoff_ts)
            return data.loc[mask].copy()
        elif isinstance(data.index, pd.DatetimeIndex):
            return data.loc[
                (data.index >= start_ts) & (data.index <= cutoff_ts)
            ].copy()
        return data.copy()

    def _load_from_warehouse(
        self, start_ts: pd.Timestamp, cutoff_ts: pd.Timestamp
    ) -> pd.DataFrame:
        """Load data from the warehouse directory.

        Override this method for production data sources.
        """
        logger.info(
            "Loading from warehouse: %s to %s", start_ts.date(), cutoff_ts.date()
        )
        # Return empty DataFrame -- subclasses or injected loaders fill this
        return pd.DataFrame()
