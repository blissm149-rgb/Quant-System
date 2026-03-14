"""Size factor: log market cap, inverted (small-cap tilt)."""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class SizeFactor(BaseFeatureGenerator):
    """Size factor — inverted log market cap (small-cap tilt).

    When market_cap is available, uses log(market_cap) inverted.
    Falls back to inverted log(price * volume) as a proxy.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(
            feature_name="size",
            lookback_days=30,
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        tickers = data.index.get_level_values("ticker").unique()

        if "market_cap" in data.columns:
            return self._compute_from_market_cap(data, tickers)
        return self._compute_proxy(data, tickers)

    def _compute_from_market_cap(
        self, data: pd.DataFrame, tickers: pd.Index
    ) -> pd.Series:
        result = {}
        for ticker in tickers:
            try:
                mcap = data.xs(ticker, level="ticker")["market_cap"].dropna().sort_index()
            except KeyError:
                result[ticker] = np.nan
                continue
            if len(mcap) == 0 or mcap.iloc[-1] <= 0:
                result[ticker] = np.nan
            else:
                # Invert: smaller cap = higher score
                result[ticker] = -np.log(mcap.iloc[-1])
        return pd.Series(result, name=self.feature_name)

    def _compute_proxy(
        self, data: pd.DataFrame, tickers: pd.Index
    ) -> pd.Series:
        """Use price * average volume as a market cap proxy."""
        if "close" not in data.columns or "volume" not in data.columns:
            return pd.Series(dtype=float)

        result = {}
        for ticker in tickers:
            try:
                td = data.xs(ticker, level="ticker").sort_index()
            except KeyError:
                continue
            if len(td) == 0:
                result[ticker] = np.nan
                continue
            price = td["close"].iloc[-1]
            avg_vol = td["volume"].iloc[-20:].mean() if len(td) >= 20 else td["volume"].mean()
            proxy = price * avg_vol
            if proxy > 0:
                result[ticker] = -np.log(proxy)
            else:
                result[ticker] = np.nan

        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.20)
