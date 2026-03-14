"""Abstract base class for all feature generators.

Every feature in the system — technical, alternative, factor — inherits from
BaseFeatureGenerator. This enforces a uniform interface for compute, validate,
and metadata, enabling the feature pipeline to treat all features generically.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import yaml


class BaseFeatureGenerator(ABC):
    """Abstract base class that all feature generators must inherit.

    Subclasses must implement compute() and validate(). The compute method
    receives pre-aligned data (only rows with timestamp < as_of) and returns
    a cross-sectional Series indexed by ticker.

    Attributes:
        feature_name: Unique identifier for this feature.
        lookback_days: Number of historical days required for computation.
            Enforced by data_alignment_engine.
        recompute_frequency: How often the feature is recomputed —
            "daily", "hourly", or "on_data".
    """

    feature_name: str
    lookback_days: int
    recompute_frequency: str  # "daily" | "hourly" | "on_data"

    def __init__(
        self,
        feature_name: str,
        lookback_days: int,
        recompute_frequency: str = "daily",
        config: Optional[dict] = None,
    ):
        self.feature_name = feature_name
        self.lookback_days = lookback_days
        self.recompute_frequency = recompute_frequency
        self._config = config or {}

    @abstractmethod
    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Compute cross-sectional feature values for all assets as of `as_of`.

        Args:
            data: DataFrame with MultiIndex (date, ticker) containing only rows
                with timestamp < as_of. Pre-aligned by data_alignment_engine.
            as_of: Point-in-time boundary. data must not contain any rows at or
                after this timestamp.

        Returns:
            pd.Series indexed by ticker with the computed feature values.
        """

    @abstractmethod
    def validate(self, feature_output: pd.Series) -> bool:
        """Validate the output of compute().

        Args:
            feature_output: Series returned by compute().

        Returns:
            True if output passes sanity checks (NaN ratio below threshold,
            values within expected range, etc.).
        """

    def get_metadata(self) -> dict:
        """Return metadata about this feature generator."""
        return {
            "feature_name": self.feature_name,
            "lookback_days": self.lookback_days,
            "recompute_frequency": self.recompute_frequency,
        }

    def _validate_nan_ratio(
        self, series: pd.Series, max_nan_ratio: float = 0.10
    ) -> bool:
        """Helper: check that NaN ratio is below threshold."""
        if len(series) == 0:
            return True
        nan_ratio = series.isna().mean()
        return nan_ratio <= max_nan_ratio

    def _validate_range(
        self,
        series: pd.Series,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> bool:
        """Helper: check that non-NaN values are within expected range."""
        clean = series.dropna()
        if len(clean) == 0:
            return True
        if min_val is not None and (clean < min_val).any():
            return False
        if max_val is not None and (clean > max_val).any():
            return False
        return True
