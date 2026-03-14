"""Sentiment feature generation.

Aggregates per-article sentiment scores to per-ticker daily signals:
- Volume-weighted average sentiment (1-day and 5-day windows)
- Sentiment momentum: change vs prior 5-day average
- Sentiment divergence: difference from sector average
"""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class SentimentFeatureGeneration(BaseFeatureGenerator):
    """Generates cross-sectional sentiment features from news scores."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._short_window = cfg.get("sentiment_short_window", 1)
        self._long_window = cfg.get("sentiment_long_window", 5)
        super().__init__(
            feature_name="news_sentiment",
            lookback_days=cfg.get("sentiment_lookback_days", 30),
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Compute sentiment features.

        Expects data to contain a 'sentiment_score' column.
        If not available, returns zeros.
        """
        tickers = data.index.get_level_values("ticker").unique() if isinstance(
            data.index, pd.MultiIndex
        ) else []

        if "sentiment_score" not in data.columns:
            return pd.Series(0.0, index=tickers, name=self.feature_name)

        result = {}
        for ticker in tickers:
            try:
                td = data.xs(ticker, level="ticker")["sentiment_score"].sort_index()
            except KeyError:
                result[ticker] = 0.0
                continue

            if len(td) == 0:
                result[ticker] = 0.0
                continue

            # Short-window average
            short_avg = td.iloc[-self._short_window:].mean()
            result[ticker] = short_avg

        return pd.Series(result, name=self.feature_name)

    def compute_sentiment_momentum(
        self, sentiment_df: pd.DataFrame, as_of: pd.Timestamp
    ) -> pd.Series:
        """Compute sentiment momentum: recent vs trailing average."""
        if sentiment_df.empty or "sentiment_score" not in sentiment_df.columns:
            return pd.Series(dtype=float, name="sentiment_momentum")

        result = {}
        tickers = sentiment_df.index.get_level_values("ticker").unique() if isinstance(
            sentiment_df.index, pd.MultiIndex
        ) else sentiment_df["ticker"].unique() if "ticker" in sentiment_df.columns else []

        for ticker in tickers:
            try:
                td = sentiment_df.xs(ticker, level="ticker")["sentiment_score"].sort_index()
            except (KeyError, TypeError):
                result[ticker] = 0.0
                continue

            if len(td) < self._long_window + 1:
                result[ticker] = 0.0
                continue

            recent = td.iloc[-self._short_window:].mean()
            trailing = td.iloc[-(self._long_window + self._short_window):-self._short_window].mean()
            result[ticker] = recent - trailing

        return pd.Series(result, name="sentiment_momentum")

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.50)
