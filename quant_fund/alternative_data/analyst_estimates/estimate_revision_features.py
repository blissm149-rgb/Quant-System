"""Analyst estimate revision features.

Produces:
- SUE (Standardised Unexpected Earnings)
- Revision momentum
- Estimate dispersion
"""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class EstimateRevisionFeatures(BaseFeatureGenerator):
    """Generates alpha signals from analyst estimate revisions.

    Features:
    - SUE: (actual - consensus) / std_dev
    - Revision momentum: % of estimates revised up over last 4 weeks
    - Estimate dispersion: std_dev of estimates / |consensus|
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._revision_window_days = cfg.get("revision_window_days", 28)
        self._min_analysts = cfg.get("min_analysts", 3)
        super().__init__(
            feature_name="estimate_revision",
            lookback_days=cfg.get("estimate_lookback_days", 90),
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Compute estimate revision features from analyst data.

        Expects data to contain columns: ticker, estimate_value, estimate_date,
        revision_direction (optional). Falls back to dispersion if available.
        """
        tickers = data.index.get_level_values("ticker").unique() if isinstance(
            data.index, pd.MultiIndex
        ) else data.get("ticker", pd.Series(dtype=str)).unique()

        if "estimate_value" not in data.columns:
            return pd.Series(0.0, index=tickers, name=self.feature_name)

        result = {}
        for ticker in tickers:
            try:
                td = data.xs(ticker, level="ticker") if isinstance(
                    data.index, pd.MultiIndex
                ) else data[data["ticker"] == ticker]
            except KeyError:
                result[ticker] = np.nan
                continue

            if len(td) < self._min_analysts:
                result[ticker] = np.nan
                continue

            estimates = td["estimate_value"].dropna()
            if len(estimates) < 2:
                result[ticker] = np.nan
                continue

            # Dispersion as default feature
            mean_est = estimates.mean()
            if abs(mean_est) > 1e-10:
                dispersion = estimates.std() / abs(mean_est)
                # Lower dispersion = higher conviction = higher score
                result[ticker] = -dispersion
            else:
                result[ticker] = 0.0

        return pd.Series(result, name=self.feature_name)

    def compute_sue(
        self,
        actual: float,
        consensus_mean: float,
        consensus_std: float,
    ) -> float:
        """Compute Standardised Unexpected Earnings.

        Args:
            actual: Actual reported EPS.
            consensus_mean: Mean consensus estimate.
            consensus_std: Standard deviation of estimates.

        Returns:
            SUE score.
        """
        if consensus_std == 0 or np.isnan(consensus_std):
            return 0.0
        return (actual - consensus_mean) / consensus_std

    def compute_revision_momentum(
        self,
        estimates_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Compute revision momentum: % of estimates revised up over recent window.

        Args:
            estimates_df: DataFrame with ticker, estimate_value, estimate_date.
            as_of: Point-in-time boundary.

        Returns:
            Series indexed by ticker with revision momentum scores.
        """
        if estimates_df.empty:
            return pd.Series(dtype=float, name="revision_momentum")

        window_start = as_of - pd.Timedelta(days=self._revision_window_days)
        recent = estimates_df[
            (estimates_df["estimate_date"] >= window_start)
            & (estimates_df["estimate_date"] < as_of)
        ]

        if recent.empty:
            return pd.Series(dtype=float, name="revision_momentum")

        result = {}
        for ticker, group in recent.groupby("ticker"):
            if len(group) < 2:
                result[ticker] = 0.0
                continue
            sorted_g = group.sort_values("estimate_date")
            # Compare each estimate to the prior one by the same analyst
            revisions = sorted_g.groupby("analyst_id").apply(
                lambda x: (x["estimate_value"].diff().dropna() > 0).mean()
                if len(x) > 1 else 0.5
            )
            result[ticker] = revisions.mean() - 0.5  # center at 0

        return pd.Series(result, name="revision_momentum")

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.50)
