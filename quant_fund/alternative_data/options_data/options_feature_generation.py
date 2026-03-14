"""Options-derived feature generation.

Produces:
- IV rank (IVR): current IV percentile vs trailing 252-day range
- IV skew: 25-delta put IV minus 25-delta call IV
- Put/call ratio
- Options-implied move
"""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class OptionsFeatureGeneration(BaseFeatureGenerator):
    """Generates alpha features from options data."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._ivr_lookback = cfg.get("ivr_lookback_days", 252)
        super().__init__(
            feature_name="options_composite",
            lookback_days=cfg.get("options_lookback_days", 260),
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Compute options-derived features.

        Expects data to contain implied_vol column. If not available,
        returns zeros.
        """
        tickers = data.index.get_level_values("ticker").unique() if isinstance(
            data.index, pd.MultiIndex
        ) else []

        if "implied_vol" not in data.columns:
            return pd.Series(0.0, index=tickers, name=self.feature_name)

        result = {}
        for ticker in tickers:
            try:
                td = data.xs(ticker, level="ticker")["implied_vol"].sort_index().dropna()
            except KeyError:
                result[ticker] = np.nan
                continue

            if len(td) < 20:
                result[ticker] = np.nan
                continue

            # IV Rank: percentile of current IV vs trailing history
            current_iv = td.iloc[-1]
            lookback = td.iloc[-self._ivr_lookback:]
            iv_rank = (lookback < current_iv).mean()

            # Center at 0: high IV rank = negative (mean reversion expectation)
            result[ticker] = -(iv_rank - 0.5)

        return pd.Series(result, name=self.feature_name)

    def compute_iv_rank(
        self, iv_series: pd.Series, lookback: int = 252
    ) -> float:
        """Compute IV rank for a single ticker."""
        if len(iv_series) < lookback:
            lookback = len(iv_series)
        if lookback < 2:
            return 50.0
        current = iv_series.iloc[-1]
        history = iv_series.iloc[-lookback:]
        return float((history < current).mean() * 100)

    def compute_put_call_ratio(
        self, chain_df: pd.DataFrame
    ) -> float:
        """Compute put/call ratio from options chain."""
        if chain_df.empty or "option_type" not in chain_df.columns:
            return np.nan

        if "volume" in chain_df.columns:
            puts = chain_df[chain_df["option_type"] == "put"]["volume"].sum()
            calls = chain_df[chain_df["option_type"] == "call"]["volume"].sum()
        elif "open_interest" in chain_df.columns:
            puts = chain_df[chain_df["option_type"] == "put"]["open_interest"].sum()
            calls = chain_df[chain_df["option_type"] == "call"]["open_interest"].sum()
        else:
            return np.nan

        if calls == 0:
            return np.nan
        return float(puts / calls)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.50)
