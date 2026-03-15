"""Base data connector interface.

Defines the abstract contract that all vendor-specific data connectors
must implement. Connectors handle authentication, rate limiting, and
data format normalisation so that downstream consumers always receive
the same MultiIndex ``(date, ticker)`` DataFrame schema.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Abstract interface for external market data sources.

    Every concrete connector must implement:
    - ``fetch_historical``: bulk historical OHLCV bars
    - ``fetch_latest``: most recent bars for live/polling use
    - ``get_source_name``: identifier for logging and metadata
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or {}
        self._rate_limit_delay: float = self._config.get(
            "rate_limit_delay_sec", 0.25
        )
        self._max_retries: int = self._config.get("max_retries", 3)

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_historical(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars for a list of tickers.

        Returns:
            DataFrame with MultiIndex ``(date, ticker)`` and columns
            ``open, high, low, close, volume`` at minimum.
        """

    @abstractmethod
    def fetch_latest(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch the most recent bar for each ticker.

        Returns:
            DataFrame with MultiIndex ``(date, ticker)`` and OHLCV columns.
        """

    @abstractmethod
    def get_source_name(self) -> str:
        """Return a human-readable name for this data source."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _retry_with_backoff(self, fn, *args, **kwargs):
        """Call *fn* with exponential backoff on failure.

        Returns the result of *fn* on success.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as exc:
                last_exc = exc
                wait = self._rate_limit_delay * (2 ** attempt)
                logger.warning(
                    "%s attempt %d/%d failed: %s — retrying in %.1fs",
                    self.get_source_name(),
                    attempt + 1,
                    self._max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has ``(date, ticker)`` MultiIndex.

        If the DataFrame already has the correct index, it is returned
        unchanged. Otherwise columns ``date`` and ``ticker`` are used.
        """
        if isinstance(df.index, pd.MultiIndex):
            if list(df.index.names) == ["date", "ticker"]:
                return df

        if "date" in df.columns and "ticker" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index(["date", "ticker"])

        return df

    @staticmethod
    def _empty_ohlcv() -> pd.DataFrame:
        """Return an empty DataFrame with the canonical schema."""
        index = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return pd.DataFrame(
            index=index,
            columns=["open", "high", "low", "close", "volume"],
            dtype=float,
        )

    def _track_ingestion(
        self,
        tickers: List[str],
        row_count: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> Dict:
        """Return metadata dict for an ingestion event."""
        return {
            "source": self.get_source_name(),
            "tickers_requested": len(tickers),
            "rows_fetched": row_count,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
        }
