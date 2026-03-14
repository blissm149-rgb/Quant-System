"""Momentum factor: 12-1 month return (skip last month to avoid reversal)."""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class MomentumFactor(BaseFeatureGenerator):
    """12-1 month price momentum factor.

    Computes trailing 12-month return excluding the most recent month
    to avoid short-term reversal contamination.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._long_window = cfg.get("momentum_long_window", 252)
        self._skip_window = cfg.get("momentum_skip_window", 21)
        super().__init__(
            feature_name="momentum",
            lookback_days=self._long_window + 10,
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        if "close" not in data.columns:
            return pd.Series(dtype=float)

        result = {}
        tickers = data.index.get_level_values("ticker").unique()
        for ticker in tickers:
            try:
                prices = data.xs(ticker, level="ticker")["close"].sort_index()
            except KeyError:
                continue
            if len(prices) < self._skip_window + 2:
                result[ticker] = np.nan
                continue
            # Price at T - skip_window (end of non-skip period)
            end_price = prices.iloc[-(self._skip_window + 1)]
            # Price at T - long_window (start of momentum window)
            start_idx = max(0, len(prices) - self._long_window)
            start_price = prices.iloc[start_idx]
            if start_price != 0:
                result[ticker] = (end_price - start_price) / start_price
            else:
                result[ticker] = np.nan

        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.20)
