"""Factor exposure estimator via cross-sectional regression.

Estimates per-stock factor exposures (betas) to market, momentum, value,
quality, low-vol, size, and sector factors using cross-sectional regression
of stock returns on factor returns.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FACTORS = ["market", "momentum", "value", "quality", "low_vol", "size"]


class FactorExposureEstimator:
    """Estimates factor exposures for each stock in the universe.

    Uses cross-sectional regression: for each date, regress stock returns
    on factor characteristics to obtain factor exposures (betas).
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._estimation_window = cfg.get("factor_estimation_window", 252)
        self._min_observations = cfg.get("factor_min_observations", 60)
        self._factors = cfg.get("factors", DEFAULT_FACTORS)
        self._include_sectors = cfg.get("include_sector_factors", True)

    def estimate(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        as_of: pd.Timestamp,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Estimate factor exposures for all stocks.

        Args:
            returns: DataFrame (dates x tickers) of daily stock returns.
            factor_returns: DataFrame (dates x factors) of daily factor returns.
            as_of: Point-in-time boundary. Only data before as_of is used.
            sector_map: Optional dict mapping ticker -> GICS sector.

        Returns:
            DataFrame (tickers x factors) of factor exposures (betas).
        """
        # Filter to data before as_of
        returns = returns[returns.index < as_of]
        factor_returns = factor_returns[factor_returns.index < as_of]

        # Use most recent estimation_window days
        returns = returns.iloc[-self._estimation_window :]
        factor_returns = factor_returns.iloc[-self._estimation_window :]

        # Align dates
        common_dates = returns.index.intersection(factor_returns.index)
        if len(common_dates) < self._min_observations:
            logger.warning(
                "Insufficient data for factor estimation: %d < %d",
                len(common_dates),
                self._min_observations,
            )
            return pd.DataFrame(
                0.0, index=returns.columns, columns=factor_returns.columns
            )

        R = returns.loc[common_dates]
        F = factor_returns.loc[common_dates]

        # Time-series regression for each stock: R_i = alpha_i + B_i * F + eps
        tickers = R.columns
        exposures = {}

        for ticker in tickers:
            y = R[ticker].values
            valid = np.isfinite(y)
            if valid.sum() < self._min_observations:
                exposures[ticker] = {f: 0.0 for f in F.columns}
                continue

            X = F.values[valid]
            y_clean = y[valid]

            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])

            try:
                betas, _, _, _ = np.linalg.lstsq(X_with_intercept, y_clean, rcond=None)
                # Skip intercept (betas[0]), take factor betas
                exposures[ticker] = {
                    f: betas[i + 1] for i, f in enumerate(F.columns)
                }
            except np.linalg.LinAlgError:
                exposures[ticker] = {f: 0.0 for f in F.columns}

        result = pd.DataFrame(exposures).T

        # Add sector dummy exposures if requested
        if self._include_sectors and sector_map:
            result = self._add_sector_exposures(result, sector_map)

        return result

    def estimate_from_characteristics(
        self,
        characteristics: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Estimate exposures directly from stock characteristics.

        Simple approach: standardise characteristics cross-sectionally.
        Each characteristic column becomes a factor exposure.

        Args:
            characteristics: DataFrame (tickers x characteristics).
            factor_names: Optional list of columns to use as factors.

        Returns:
            DataFrame (tickers x factors) of standardised exposures.
        """
        if factor_names:
            chars = characteristics[factor_names]
        else:
            chars = characteristics

        # Cross-sectional standardisation
        result = pd.DataFrame(index=chars.index)
        for col in chars.columns:
            series = chars[col].dropna()
            if len(series) < 2 or series.std() == 0:
                result[col] = 0.0
            else:
                result[col] = (chars[col] - series.mean()) / series.std()

        return result.fillna(0.0)

    def _add_sector_exposures(
        self, exposures: pd.DataFrame, sector_map: Dict[str, str]
    ) -> pd.DataFrame:
        """Add sector dummy variable exposures."""
        sectors = sorted(set(sector_map.values()))
        for sector in sectors:
            col_name = f"sector_{sector}"
            exposures[col_name] = [
                1.0 if sector_map.get(t) == sector else 0.0
                for t in exposures.index
            ]
        return exposures
