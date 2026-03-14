"""Value factor: book-to-market, earnings yield, or composite."""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class ValueFactor(BaseFeatureGenerator):
    """Value factor based on earnings yield or book-to-market ratio.

    When fundamental data (earnings_yield or book_to_market columns) is
    available, uses those directly. Otherwise falls back to a price-based
    value proxy (inverse of trailing return — contrarian tilt).
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._value_metric = cfg.get("value_metric", "earnings_yield")
        self._lookback = cfg.get("value_lookback_days", 63)
        super().__init__(
            feature_name="value",
            lookback_days=self._lookback + 10,
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        tickers = data.index.get_level_values("ticker").unique()

        # Try fundamental columns first
        if self._value_metric in data.columns:
            return self._compute_from_fundamental(data, tickers)

        # Fallback: inverse of trailing return as a value proxy
        return self._compute_price_proxy(data, tickers)

    def _compute_from_fundamental(
        self, data: pd.DataFrame, tickers: pd.Index
    ) -> pd.Series:
        result = {}
        for ticker in tickers:
            try:
                ticker_data = data.xs(ticker, level="ticker")[self._value_metric].sort_index()
            except KeyError:
                result[ticker] = np.nan
                continue
            valid = ticker_data.dropna()
            if len(valid) == 0:
                result[ticker] = np.nan
            else:
                result[ticker] = valid.iloc[-1]
        return pd.Series(result, name=self.feature_name)

    def _compute_price_proxy(
        self, data: pd.DataFrame, tickers: pd.Index
    ) -> pd.Series:
        """Use negative trailing return as a value proxy (cheaper = higher score)."""
        if "close" not in data.columns:
            return pd.Series(dtype=float)

        result = {}
        for ticker in tickers:
            try:
                prices = data.xs(ticker, level="ticker")["close"].sort_index()
            except KeyError:
                continue
            if len(prices) < self._lookback:
                result[ticker] = np.nan
                continue
            start_idx = max(0, len(prices) - self._lookback)
            ret = (prices.iloc[-1] - prices.iloc[start_idx]) / prices.iloc[start_idx]
            # Invert: stocks that dropped more get higher value scores
            result[ticker] = -ret

        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.30)
