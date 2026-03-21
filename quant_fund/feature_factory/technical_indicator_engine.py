"""Technical indicator engine computing price/volume based features.

All indicators inherit from BaseFeatureGenerator and are computed strictly
point-in-time using data aligned by data_alignment_engine.
"""

from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator
from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine


class ReturnFeature(BaseFeatureGenerator):
    """Cross-sectional return over a configurable lookback period."""

    def __init__(self, window_days: int = 20, config: Optional[dict] = None):
        self._window_days = window_days
        super().__init__(
            feature_name=f"return_{window_days}d",
            lookback_days=window_days + 5,  # buffer for weekends/holidays
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
                ticker_data = data.xs(ticker, level="ticker")["close"].sort_index()
            except KeyError:
                continue
            if len(ticker_data) < 2:
                result[ticker] = np.nan
                continue
            current = ticker_data.iloc[-1]
            lookback_idx = max(0, len(ticker_data) - self._window_days)
            past = ticker_data.iloc[lookback_idx]
            if past != 0:
                result[ticker] = (current - past) / past
            else:
                result[ticker] = np.nan
        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.20)


class MomentumFeature(BaseFeatureGenerator):
    """12-1 month momentum: 12-month return excluding the most recent month."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(
            feature_name="momentum_12_1",
            lookback_days=270,  # ~12 months of trading days + buffer
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
                ticker_data = data.xs(ticker, level="ticker")["close"].sort_index()
            except KeyError:
                continue
            if len(ticker_data) < 22:
                result[ticker] = np.nan
                continue
            # Skip last ~21 trading days (1 month)
            current = ticker_data.iloc[-22]
            past_idx = max(0, len(ticker_data) - 252)
            past = ticker_data.iloc[past_idx]
            if past != 0:
                result[ticker] = (current - past) / past
            else:
                result[ticker] = np.nan
        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.20)


class VolatilityFeature(BaseFeatureGenerator):
    """Rolling realised volatility over a configurable window."""

    def __init__(self, window_days: int = 20, config: Optional[dict] = None):
        self._window_days = window_days
        super().__init__(
            feature_name=f"volatility_{window_days}d",
            lookback_days=window_days + 10,
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
                ticker_data = data.xs(ticker, level="ticker")["close"].sort_index()
            except KeyError:
                continue
            if len(ticker_data) < 3:
                result[ticker] = np.nan
                continue
            log_returns = np.log(ticker_data / ticker_data.shift(1)).dropna()
            window = min(self._window_days, len(log_returns))
            vol = log_returns.iloc[-window:].std() * np.sqrt(252)
            result[ticker] = vol
        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        if not self._validate_nan_ratio(feature_output, max_nan_ratio=0.20):
            return False
        return self._validate_range(feature_output, min_val=0.0)


class RelativeVolumeFeature(BaseFeatureGenerator):
    """Relative volume: current volume vs trailing average."""

    def __init__(self, window_days: int = 20, config: Optional[dict] = None):
        self._window_days = window_days
        super().__init__(
            feature_name=f"relative_volume_{window_days}d",
            lookback_days=window_days + 5,
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        if "volume" not in data.columns:
            return pd.Series(dtype=float)

        result = {}
        tickers = data.index.get_level_values("ticker").unique()
        for ticker in tickers:
            try:
                ticker_data = data.xs(ticker, level="ticker")["volume"].sort_index()
            except KeyError:
                continue
            if len(ticker_data) < 2:
                result[ticker] = np.nan
                continue
            current_vol = ticker_data.iloc[-1]
            avg_vol = ticker_data.iloc[-self._window_days:].mean()
            if avg_vol > 0:
                result[ticker] = current_vol / avg_vol
            else:
                result[ticker] = np.nan
        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        if not self._validate_nan_ratio(feature_output, max_nan_ratio=0.20):
            return False
        return self._validate_range(feature_output, min_val=0.0)


class ShortTermReversalFeature(BaseFeatureGenerator):
    """Short-term reversal: 1-week return (inverted for reversal signal)."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(
            feature_name="short_term_reversal",
            lookback_days=10,
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
                ticker_data = data.xs(ticker, level="ticker")["close"].sort_index()
            except KeyError:
                continue
            if len(ticker_data) < 5:
                result[ticker] = np.nan
                continue
            current = ticker_data.iloc[-1]
            week_ago = ticker_data.iloc[-5]
            if week_ago != 0:
                # Negate for reversal: losers become positive signal
                result[ticker] = -(current - week_ago) / week_ago
            else:
                result[ticker] = np.nan
        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.20)


class RSIFeature(BaseFeatureGenerator):
    """Relative Strength Index over a configurable window."""

    def __init__(self, window_days: int = 14, config: Optional[dict] = None):
        self._window_days = window_days
        super().__init__(
            feature_name=f"rsi_{window_days}d",
            lookback_days=window_days + 10,
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
                ticker_data = data.xs(ticker, level="ticker")["close"].sort_index()
            except KeyError:
                continue
            if len(ticker_data) < self._window_days + 1:
                result[ticker] = np.nan
                continue
            delta = ticker_data.diff().dropna()
            window = min(self._window_days, len(delta))
            recent = delta.iloc[-window:]
            gains = recent.clip(lower=0).mean()
            losses = (-recent.clip(upper=0)).mean()
            if losses == 0:
                result[ticker] = 100.0
            else:
                rs = gains / losses
                result[ticker] = 100.0 - (100.0 / (1.0 + rs))
        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        if not self._validate_nan_ratio(feature_output, max_nan_ratio=0.20):
            return False
        return self._validate_range(feature_output, min_val=0.0, max_val=100.0)


class BollingerBandPositionFeature(BaseFeatureGenerator):
    """Position within Bollinger Bands: (price - lower) / (upper - lower)."""

    def __init__(self, window_days: int = 20, num_std: float = 2.0, config: Optional[dict] = None):
        self._window_days = window_days
        self._num_std = num_std
        super().__init__(
            feature_name=f"bollinger_position_{window_days}d",
            lookback_days=window_days + 5,
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
                ticker_data = data.xs(ticker, level="ticker")["close"].sort_index()
            except KeyError:
                continue
            if len(ticker_data) < self._window_days:
                result[ticker] = np.nan
                continue
            window_data = ticker_data.iloc[-self._window_days:]
            sma = window_data.mean()
            std = window_data.std()
            if std == 0:
                result[ticker] = 0.5
                continue
            upper = sma + self._num_std * std
            lower = sma - self._num_std * std
            current = ticker_data.iloc[-1]
            band_width = upper - lower
            if band_width > 0:
                result[ticker] = (current - lower) / band_width
            else:
                result[ticker] = 0.5
        return pd.Series(result, name=self.feature_name)

    def validate(self, feature_output: pd.Series) -> bool:
        return self._validate_nan_ratio(feature_output, max_nan_ratio=0.20)


class TechnicalIndicatorEngine:
    """Orchestrates computation of all technical indicators.

    Creates and manages a set of technical feature generators. Each generator
    is computed using point-in-time aligned data from the DataAlignmentEngine.
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._alignment_engine = DataAlignmentEngine(config)
        self._generators = self._build_generators()

    def _build_generators(self) -> list[BaseFeatureGenerator]:
        """Build the default set of technical indicators from config."""
        indicators_config = self._config.get("technical_indicators", {})

        return_windows = indicators_config.get("return_windows", [1, 5, 20, 60, 252])
        vol_windows = indicators_config.get("volatility_windows", [20, 60])
        rsi_window = indicators_config.get("rsi_window", 14)
        bb_window = indicators_config.get("bollinger_window", 20)
        bb_std = indicators_config.get("bollinger_std", 2.0)
        rv_window = indicators_config.get("relative_volume_window", 20)

        generators: list[BaseFeatureGenerator] = []

        for w in return_windows:
            generators.append(ReturnFeature(window_days=w, config=self._config))

        generators.append(MomentumFeature(config=self._config))

        for w in vol_windows:
            generators.append(VolatilityFeature(window_days=w, config=self._config))

        generators.append(RelativeVolumeFeature(window_days=rv_window, config=self._config))
        generators.append(ShortTermReversalFeature(config=self._config))
        generators.append(RSIFeature(window_days=rsi_window, config=self._config))
        generators.append(BollingerBandPositionFeature(window_days=bb_window, num_std=bb_std, config=self._config))

        return generators

    @property
    def generators(self) -> list[BaseFeatureGenerator]:
        return self._generators

    def compute_all(
        self, data: pd.DataFrame, as_of: pd.Timestamp
    ) -> pd.DataFrame:
        """Compute all technical indicators for the given data.

        Args:
            data: Full historical data (will be aligned per-generator).
            as_of: Point-in-time boundary.

        Returns:
            DataFrame indexed by ticker with one column per feature.
        """
        features = {}
        for gen in self._generators:
            aligned = self._alignment_engine.get_aligned_data(
                data, as_of=as_of, lookback_days=gen.lookback_days
            )
            if aligned.empty:
                continue
            feature_values = gen.compute(aligned, as_of=as_of)
            if gen.validate(feature_values):
                features[gen.feature_name] = feature_values

        if not features:
            return pd.DataFrame()
        return pd.DataFrame(features)
