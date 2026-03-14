"""Analyst estimate data ingestion.

Loads analyst consensus estimates and individual estimates with
point-in-time timestamps. Ensures estimates are not available before
their announcement date.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class AnalystEstimate:
    """A single analyst estimate record."""

    ticker: str
    analyst_id: str
    metric: str  # "eps", "revenue", "ebitda"
    period: str  # "FY1", "FY2", "Q1", etc.
    estimate_value: float
    estimate_date: pd.Timestamp  # when estimate was published
    period_end_date: pd.Timestamp


class AnalystDataIngestion:
    """Ingests and manages analyst estimate data with point-in-time integrity."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._data_source = cfg.get("analyst_estimates", "ibes")
        self._min_analysts = cfg.get("min_analysts", 3)

    @classmethod
    def from_config_file(cls, config_path: str) -> "AnalystDataIngestion":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config.get("data_sources", {}))

    def load_estimates(
        self,
        tickers: List[str],
        as_of: pd.Timestamp,
        metric: str = "eps",
    ) -> pd.DataFrame:
        """Load analyst estimates available as of a specific date.

        Args:
            tickers: Tickers to load estimates for.
            as_of: Only estimates published before this date are returned.
            metric: Estimate metric (eps, revenue, ebitda).

        Returns:
            DataFrame with columns: ticker, analyst_id, metric, period,
            estimate_value, estimate_date, period_end_date.
        """
        # In production, this would query IBES or similar database.
        # Returns empty DataFrame as placeholder for data source integration.
        logger.info("Loading %s estimates for %d tickers as of %s",
                     metric, len(tickers), as_of)
        return pd.DataFrame(columns=[
            "ticker", "analyst_id", "metric", "period",
            "estimate_value", "estimate_date", "period_end_date",
        ])

    def compute_consensus(
        self,
        estimates_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """Compute consensus estimates as of a specific date.

        Args:
            estimates_df: Raw analyst estimates.
            as_of: Point-in-time boundary.

        Returns:
            DataFrame with columns: ticker, period, consensus_mean,
            consensus_median, n_analysts, std_dev.
        """
        if estimates_df.empty:
            return pd.DataFrame(columns=[
                "ticker", "period", "consensus_mean", "consensus_median",
                "n_analysts", "std_dev",
            ])

        # Filter to estimates published before as_of
        mask = estimates_df["estimate_date"] < as_of
        valid = estimates_df[mask]

        if valid.empty:
            return pd.DataFrame(columns=[
                "ticker", "period", "consensus_mean", "consensus_median",
                "n_analysts", "std_dev",
            ])

        # Take the most recent estimate per analyst per ticker per period
        valid = valid.sort_values("estimate_date")
        latest = valid.groupby(["ticker", "period", "analyst_id"]).last().reset_index()

        result = latest.groupby(["ticker", "period"]).agg(
            consensus_mean=("estimate_value", "mean"),
            consensus_median=("estimate_value", "median"),
            n_analysts=("estimate_value", "count"),
            std_dev=("estimate_value", "std"),
        ).reset_index()

        # Filter by minimum analyst coverage
        result = result[result["n_analysts"] >= self._min_analysts]

        return result

    def load_actuals(
        self,
        tickers: List[str],
        as_of: pd.Timestamp,
        metric: str = "eps",
    ) -> pd.DataFrame:
        """Load actual reported values available as of a specific date.

        Actual earnings are only available after the announcement date,
        lagged by at least 1 trading day.

        Returns:
            DataFrame with columns: ticker, period, actual_value,
            announcement_date.
        """
        logger.info("Loading %s actuals for %d tickers as of %s",
                     metric, len(tickers), as_of)
        return pd.DataFrame(columns=[
            "ticker", "period", "actual_value", "announcement_date",
        ])
