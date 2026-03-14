"""Low volatility factor: trailing realised vol, inverted."""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class LowVolatilityFactor(BaseFeatureGenerator):
    """Low volatility factor — inverted trailing realised volatility.

    Lower historical volatility receives a higher factor score.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._vol_window = cfg.get("low_vol_window", 252)
        super().__init__(
            feature_name="low_volatility",
            lookback_days=self._vol_window + 10,
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
            if len(prices) < 3:
                result[ticker] = np.nan
                continue
            log_returns = np.log(prices / prices.shift(1)).dropna()
            window = min(self._vol_window, len(log_returns))
            vol = log_returns.iloc[-window:].std() * np.sqrt(252)
            # Invert: lower vol = higher score
            result[ticker] = -vol if vol > 0 else np.nan

        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.20)
