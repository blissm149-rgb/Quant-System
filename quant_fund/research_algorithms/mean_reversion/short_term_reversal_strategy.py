"""Short-term reversal strategy with volume filter.

1-week reversal: only trade reversals accompanied by above-average volume
to reduce noise.
"""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class ShortTermReversalStrategy(BaseFeatureGenerator):
    """Short-term reversal signal with volume confirmation.

    Computes 1-week return, inverts it (reversal), and masks out signals
    where volume is below the trailing average (noise filter).
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._return_days = cfg.get("reversal_return_days", 5)
        self._volume_avg_window = cfg.get("reversal_volume_window", 20)
        self._volume_threshold = cfg.get("reversal_volume_threshold", 1.0)
        super().__init__(
            feature_name="short_term_reversal",
            lookback_days=max(self._return_days, self._volume_avg_window) + 10,
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        if "close" not in data.columns:
            return pd.Series(dtype=float)

        tickers = data.index.get_level_values("ticker").unique()
        result = {}

        for ticker in tickers:
            try:
                td = data.xs(ticker, level="ticker").sort_index()
            except KeyError:
                continue

            prices = td["close"]
            if len(prices) < self._return_days + 1:
                result[ticker] = np.nan
                continue

            # 1-week return, inverted
            ret = (prices.iloc[-1] - prices.iloc[-self._return_days - 1]) / prices.iloc[-self._return_days - 1]
            reversal_signal = -ret

            # Volume filter
            if "volume" in td.columns and len(td) >= self._volume_avg_window:
                current_vol = td["volume"].iloc[-1]
                avg_vol = td["volume"].iloc[-self._volume_avg_window:].mean()
                if avg_vol > 0 and current_vol / avg_vol < self._volume_threshold:
                    reversal_signal = 0.0  # suppress signal on low volume

            result[ticker] = reversal_signal

        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.20)
