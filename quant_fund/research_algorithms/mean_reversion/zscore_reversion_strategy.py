"""Z-score mean reversion strategy.

Cross-sectional z-score of short-term return. Fade extremes:
long bottom decile, short top decile. Hold 1-5 days.
"""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class ZScoreReversionStrategy(BaseFeatureGenerator):
    """Mean reversion signal based on cross-sectional z-score of recent returns.

    Computes the trailing N-day return for each ticker, z-scores cross-sectionally,
    and inverts the signal (negative z-score = long, positive z-score = short).
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._return_window = cfg.get("reversion_return_window", 5)
        self._zscore_window = cfg.get("reversion_zscore_window", 60)
        super().__init__(
            feature_name="zscore_reversion",
            lookback_days=self._zscore_window + self._return_window + 10,
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        if "close" not in data.columns:
            return pd.Series(dtype=float)

        tickers = data.index.get_level_values("ticker").unique()
        returns = {}

        for ticker in tickers:
            try:
                prices = data.xs(ticker, level="ticker")["close"].sort_index()
            except KeyError:
                continue
            if len(prices) < self._return_window + 1:
                returns[ticker] = np.nan
                continue
            ret = (prices.iloc[-1] - prices.iloc[-self._return_window - 1]) / prices.iloc[-self._return_window - 1]
            returns[ticker] = ret

        ret_series = pd.Series(returns)
        clean = ret_series.dropna()
        if len(clean) < 3:
            return pd.Series(0.0, index=ret_series.index, name=self.feature_name)

        mean = clean.mean()
        std = clean.std()
        if std == 0:
            return pd.Series(0.0, index=ret_series.index, name=self.feature_name)

        z_scores = (ret_series - mean) / std
        # Invert: fade extremes (losers become positive signal)
        signal = -z_scores
        signal.name = self.feature_name
        return signal

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.20)
