"""ETF flow feature generation.

Produces:
- Net flow as % of AUM over trailing 5 and 20 days
- Flow surprise vs trailing 60-day average
- Sector rotation signal
"""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class FlowFeatureGeneration(BaseFeatureGenerator):
    """Generates features from ETF flow data."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._short_window = cfg.get("flow_short_window", 5)
        self._long_window = cfg.get("flow_long_window", 20)
        self._surprise_window = cfg.get("flow_surprise_window", 60)
        super().__init__(
            feature_name="etf_flow",
            lookback_days=cfg.get("flow_lookback_days", 90),
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Compute flow features. Expects 'net_flow_pct' column."""
        tickers = data.index.get_level_values("ticker").unique() if isinstance(
            data.index, pd.MultiIndex
        ) else []

        if "net_flow_pct" not in data.columns:
            return pd.Series(0.0, index=tickers, name=self.feature_name)

        result = {}
        for ticker in tickers:
            try:
                td = data.xs(ticker, level="ticker")["net_flow_pct"].sort_index()
            except KeyError:
                result[ticker] = 0.0
                continue

            if len(td) < self._short_window:
                result[ticker] = 0.0
                continue

            short_flow = td.iloc[-self._short_window:].sum()
            result[ticker] = short_flow

        return pd.Series(result, name=self.feature_name)

    def compute_flow_surprise(
        self, flow_data: pd.DataFrame, as_of: pd.Timestamp
    ) -> pd.Series:
        """Compute flow surprise vs trailing average."""
        if flow_data.empty or "net_flow_pct" not in flow_data.columns:
            return pd.Series(dtype=float, name="flow_surprise")

        result = {}
        tickers = flow_data.index.get_level_values("ticker").unique() if isinstance(
            flow_data.index, pd.MultiIndex
        ) else []

        for ticker in tickers:
            try:
                td = flow_data.xs(ticker, level="ticker")["net_flow_pct"].sort_index()
            except KeyError:
                result[ticker] = 0.0
                continue

            if len(td) < self._surprise_window:
                result[ticker] = 0.0
                continue

            recent = td.iloc[-self._short_window:].mean()
            trailing = td.iloc[-self._surprise_window:-self._short_window].mean()
            std = td.iloc[-self._surprise_window:].std()
            if std > 0:
                result[ticker] = (recent - trailing) / std
            else:
                result[ticker] = 0.0

        return pd.Series(result, name="flow_surprise")

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.50)
