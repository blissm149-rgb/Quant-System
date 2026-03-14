"""Macro feature generation.

Transforms raw macro data into features usable by the strategy allocator
and regime classifier.
"""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class MacroFeatureGeneration(BaseFeatureGenerator):
    """Generates cross-asset features from macroeconomic data."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        super().__init__(
            feature_name="macro_composite",
            lookback_days=cfg.get("macro_lookback_days", 365),
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Compute macro features. Returns single-row Series of macro signals.

        For cross-sectional features (per-ticker), macro data is the same
        for all tickers and acts as a regime indicator rather than a
        ticker-level signal. Returns empty series by default.
        """
        return pd.Series(dtype=float, name=self.feature_name)

    def compute_yield_curve_slope(self, macro_df: pd.DataFrame) -> pd.Series:
        """Compute yield curve slope (10Y - 2Y spread)."""
        if "treasury_10y" in macro_df.columns and "treasury_2y" in macro_df.columns:
            return macro_df["treasury_10y"] - macro_df["treasury_2y"]
        return pd.Series(dtype=float, name="yield_curve_slope")

    def compute_growth_momentum(
        self, gdp_series: pd.Series, lookback: int = 4
    ) -> float:
        """Compute GDP growth momentum (change over last N quarters)."""
        if len(gdp_series) < lookback + 1:
            return 0.0
        return float(gdp_series.iloc[-1] - gdp_series.iloc[-lookback - 1])

    def compute_inflation_trend(
        self, cpi_series: pd.Series, lookback: int = 12
    ) -> float:
        """Compute CPI year-over-year change trend."""
        if len(cpi_series) < lookback + 1:
            return 0.0
        yoy = (cpi_series.iloc[-1] - cpi_series.iloc[-lookback - 1]) / cpi_series.iloc[-lookback - 1]
        return float(yoy)

    def validate(self, feature_output: pd.Series) -> bool:
        return True
