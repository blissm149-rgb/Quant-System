"""Quality factor: ROE, earnings stability, low accruals composite."""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


class QualityFactor(BaseFeatureGenerator):
    """Quality factor based on profitability and earnings stability.

    Uses fundamental columns (roe, earnings_stability, accruals) when available.
    Falls back to a volatility-based quality proxy (lower volatility = higher quality).
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._vol_window = cfg.get("quality_vol_window", 60)
        self._roe_weight = cfg.get("quality_roe_weight", 0.4)
        self._stability_weight = cfg.get("quality_stability_weight", 0.3)
        self._accruals_weight = cfg.get("quality_accruals_weight", 0.3)
        super().__init__(
            feature_name="quality",
            lookback_days=max(252, self._vol_window + 10),
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        tickers = data.index.get_level_values("ticker").unique()

        has_fundamentals = any(
            col in data.columns for col in ("roe", "earnings_stability", "accruals")
        )

        if has_fundamentals:
            return self._compute_from_fundamentals(data, tickers)
        return self._compute_proxy(data, tickers)

    def _compute_from_fundamentals(
        self, data: pd.DataFrame, tickers: pd.Index
    ) -> pd.Series:
        result = {}
        for ticker in tickers:
            try:
                ticker_data = data.xs(ticker, level="ticker")
            except KeyError:
                result[ticker] = np.nan
                continue

            score = 0.0
            weight_sum = 0.0

            if "roe" in ticker_data.columns:
                roe = ticker_data["roe"].dropna()
                if len(roe) > 0:
                    score += self._roe_weight * roe.iloc[-1]
                    weight_sum += self._roe_weight

            if "earnings_stability" in ticker_data.columns:
                stab = ticker_data["earnings_stability"].dropna()
                if len(stab) > 0:
                    score += self._stability_weight * stab.iloc[-1]
                    weight_sum += self._stability_weight

            if "accruals" in ticker_data.columns:
                acc = ticker_data["accruals"].dropna()
                if len(acc) > 0:
                    # Lower accruals = higher quality
                    score += self._accruals_weight * (-acc.iloc[-1])
                    weight_sum += self._accruals_weight

            result[ticker] = score / weight_sum if weight_sum > 0 else np.nan

        return pd.Series(result, name=self.feature_name)

    def _compute_proxy(
        self, data: pd.DataFrame, tickers: pd.Index
    ) -> pd.Series:
        """Use inverse volatility as a quality proxy."""
        if "close" not in data.columns:
            return pd.Series(dtype=float)

        result = {}
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
            vol = log_returns.iloc[-window:].std()
            # Higher quality = lower volatility
            result[ticker] = -vol if vol > 0 else np.nan

        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.30)
