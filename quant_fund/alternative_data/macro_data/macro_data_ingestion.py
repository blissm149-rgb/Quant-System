"""Macro data ingestion from FRED and similar sources."""

import logging
from typing import Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class MacroDataIngestion:
    """Ingests macroeconomic data series from FRED or configured sources."""

    # Common FRED series IDs
    DEFAULT_SERIES = {
        "gdp_growth": "A191RL1Q225SBEA",
        "cpi_yoy": "CPIAUCSL",
        "unemployment": "UNRATE",
        "fed_funds": "FEDFUNDS",
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "vix": "VIXCLS",
    }

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._data_source = cfg.get("macro", "fred")
        self._series_map = cfg.get("macro_series", self.DEFAULT_SERIES)

    @classmethod
    def from_config_file(cls, config_path: str) -> "MacroDataIngestion":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config.get("data_sources", {}))

    def load_series(
        self,
        series_name: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.Series:
        """Load a single macro data series.

        Args:
            series_name: Logical name (e.g. "gdp_growth", "cpi_yoy").
            start_date: Start of range.
            end_date: End of range (point-in-time boundary).

        Returns:
            Time series of macro data.
        """
        logger.info("Loading macro series %s from %s to %s",
                     series_name, start_date, end_date)
        return pd.Series(dtype=float, name=series_name)

    def load_all(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Load all configured macro series."""
        series_dict = {}
        for name in self._series_map:
            s = self.load_series(name, start_date, end_date)
            if not s.empty:
                series_dict[name] = s
        if not series_dict:
            return pd.DataFrame()
        return pd.DataFrame(series_dict)
