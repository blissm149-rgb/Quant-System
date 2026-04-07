"""YFinance feed provider — wraps YFinanceConnector for live data streaming.

Implements the FeedProvider interface used by LiveDataStreamAdapter.
Yahoo Finance data has a ~15-minute delay which is acceptable for paper
trading and strategy development.
"""

import logging
from typing import List, Optional

import pandas as pd

from quant_fund.data_layer.connectors.yfinance_connector import YFinanceConnector
from quant_fund.data_layer.live_data_stream_adapter import FeedProvider

logger = logging.getLogger(__name__)


class YFinanceFeedProvider(FeedProvider):
    """Wraps YFinanceConnector to implement the FeedProvider interface."""

    def __init__(self, config: Optional[dict] = None):
        self._connector = YFinanceConnector(config=config)
        self._connected = False
        self._consecutive_failures = 0
        self._failure_warning_threshold = 3
        self._failure_critical_threshold = 10

    def connect(self) -> None:
        self._connected = True
        logger.info("YFinanceFeedProvider connected")

    def disconnect(self) -> None:
        self._connected = False
        logger.info("YFinanceFeedProvider disconnected")

    def fetch_latest_bars(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch latest bars from Yahoo Finance."""
        if not tickers:
            return pd.DataFrame()
        try:
            df = self._connector.fetch_latest(tickers)
            if df.empty:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_critical_threshold:
                    logger.critical(
                        "YFinance returned empty data %d consecutive times — "
                        "possible rate limit or outage",
                        self._consecutive_failures,
                    )
                elif self._consecutive_failures >= self._failure_warning_threshold:
                    logger.warning(
                        "YFinance returned empty data %d consecutive times",
                        self._consecutive_failures,
                    )
            else:
                if self._consecutive_failures > 0:
                    logger.info(
                        "YFinance recovered after %d empty responses",
                        self._consecutive_failures,
                    )
                self._consecutive_failures = 0
            return df
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_critical_threshold:
                logger.critical(
                    "YFinance fetch failed %d consecutive times — "
                    "possible rate limit or network outage",
                    self._consecutive_failures,
                )
            else:
                logger.warning("YFinance fetch_latest_bars failed", exc_info=True)
            return pd.DataFrame()

    def fetch_snapshot(
        self, tickers: List[str], fields: List[str]
    ) -> pd.DataFrame:
        """Fetch snapshot — delegates to fetch_latest_bars."""
        return self.fetch_latest_bars(tickers)
