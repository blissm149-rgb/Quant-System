"""Corporate action adjuster for historical price data.

Adjusts prices for splits, dividends, and spin-offs. Raw unadjusted data
is always preserved. Adjusted series are stored separately. Adjustment
factors are applied backward from the most recent date.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

AdjustmentType = Literal["split", "dividend", "spinoff"]


@dataclass
class CorporateAction:
    """Represents a single corporate action event."""

    ticker: str
    ex_date: pd.Timestamp
    action_type: AdjustmentType
    adjustment_factor: float  # multiplicative for price; divisive for shares


class CorporateActionAdjuster:
    """Adjusts historical prices for corporate actions.

    Raw data is never modified. Adjusted series are computed by applying
    cumulative adjustment factors backward from the most recent date.
    """

    PRICE_COLUMNS = ("open", "high", "low", "close")
    VOLUME_COLUMN = "volume"

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._actions: List[CorporateAction] = []

    @classmethod
    def from_config_file(cls, config_path: str) -> "CorporateActionAdjuster":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config)

    def register_actions(self, actions: List[CorporateAction]) -> None:
        """Register corporate actions to apply during adjustment."""
        self._actions.extend(actions)
        self._actions.sort(key=lambda a: (a.ticker, a.ex_date))

    def clear_actions(self) -> None:
        """Clear all registered corporate actions."""
        self._actions.clear()

    def load_actions_from_dataframe(self, df: pd.DataFrame) -> None:
        """Load corporate actions from a DataFrame.

        Expected columns: ticker, ex_date, action_type, adjustment_factor
        """
        for _, row in df.iterrows():
            action = CorporateAction(
                ticker=row["ticker"],
                ex_date=pd.Timestamp(row["ex_date"]),
                action_type=row["action_type"],
                adjustment_factor=float(row["adjustment_factor"]),
            )
            self._actions.append(action)
        self._actions.sort(key=lambda a: (a.ticker, a.ex_date))

    def compute_adjustment_factors(
        self, ticker: str, dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Compute cumulative backward-looking adjustment factors for a ticker.

        Factors are applied backward from the most recent date: prices before
        a corporate action's ex_date are multiplied by the cumulative factor.

        Args:
            ticker: Stock ticker symbol.
            dates: DatetimeIndex of trading dates.

        Returns:
            Series of cumulative adjustment factors indexed by date.
        """
        ticker_actions = [a for a in self._actions if a.ticker == ticker]

        factors = pd.Series(1.0, index=dates, dtype=np.float64)

        for action in ticker_actions:
            mask = dates < action.ex_date
            factors[mask] *= action.adjustment_factor

        return factors

    def adjust_series(
        self, ticker: str, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Adjust price and volume data for a single ticker.

        Args:
            ticker: Stock ticker symbol.
            prices: DataFrame with DatetimeIndex and OHLCV columns.

        Returns:
            New DataFrame with adjusted prices and volumes.
        """
        if prices.empty:
            return prices.copy()

        dates = prices.index
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        factors = self.compute_adjustment_factors(ticker, dates)

        adjusted = prices.copy()

        for col in self.PRICE_COLUMNS:
            if col in adjusted.columns:
                adjusted[col] = adjusted[col] * factors

        if self.VOLUME_COLUMN in adjusted.columns:
            # Volume is adjusted inversely — a 2:1 split doubles shares,
            # so historical volume is divided by the split factor
            volume_factors = factors.replace(0, np.nan)
            adjusted[self.VOLUME_COLUMN] = (
                adjusted[self.VOLUME_COLUMN] / volume_factors
            )

        return adjusted

    def adjust_dataframe(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adjust a MultiIndex (date, ticker) DataFrame for corporate actions.

        Args:
            df: DataFrame with MultiIndex (date, ticker) and OHLCV columns.

        Returns:
            New DataFrame with adjusted data.
        """
        if df.empty:
            return df.copy()

        if isinstance(df.index, pd.MultiIndex):
            return self._adjust_multiindex(df)
        elif "ticker" in df.columns:
            return self._adjust_with_ticker_column(df)
        else:
            logger.warning("Cannot determine ticker grouping; returning unadjusted")
            return df.copy()

    def _adjust_multiindex(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust a MultiIndex DataFrame."""
        ticker_level = "ticker" if "ticker" in df.index.names else 1
        tickers = df.index.get_level_values(ticker_level).unique()

        frames = []
        for ticker in tickers:
            try:
                ticker_data = df.xs(ticker, level=ticker_level)
            except KeyError:
                continue

            adjusted = self.adjust_series(ticker, ticker_data)

            adjusted_mi = adjusted.copy()
            adjusted_mi["ticker"] = ticker
            adjusted_mi = adjusted_mi.set_index("ticker", append=True)
            if adjusted_mi.index.names != df.index.names:
                adjusted_mi.index = adjusted_mi.index.reorder_levels(df.index.names)

            frames.append(adjusted_mi)

        if not frames:
            return df.copy()
        return pd.concat(frames).sort_index()

    def _adjust_with_ticker_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust a DataFrame with 'ticker' as a column."""
        frames = []
        for ticker, group in df.groupby("ticker"):
            adjusted = self.adjust_series(str(ticker), group.set_index("date") if "date" in group.columns else group)
            if "date" not in adjusted.columns and adjusted.index.name == "date":
                adjusted = adjusted.reset_index()
            adjusted["ticker"] = ticker
            frames.append(adjusted)

        if not frames:
            return df.copy()
        return pd.concat(frames, ignore_index=True)

    def get_actions_for_ticker(self, ticker: str) -> List[CorporateAction]:
        """Return all registered corporate actions for a ticker."""
        return [a for a in self._actions if a.ticker == ticker]

    def get_known_split(
        self, ticker: str, ex_date: pd.Timestamp
    ) -> Optional[CorporateAction]:
        """Look up a specific corporate action."""
        for a in self._actions:
            if a.ticker == ticker and a.ex_date == ex_date:
                return a
        return None
