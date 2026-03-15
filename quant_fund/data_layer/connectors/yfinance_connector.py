"""Yahoo Finance data connector via the yfinance library.

Provides free historical and latest-bar data for US equities. This is
suitable for research and paper-trading but should not be relied upon
for latency-sensitive live trading (Yahoo's data has a ~15-minute delay
and no guaranteed SLA).

Requires: ``pip install yfinance``
"""

import logging
from typing import List, Optional

import pandas as pd

from quant_fund.data_layer.connectors.base_connector import BaseConnector

logger = logging.getLogger(__name__)


class YFinanceConnector(BaseConnector):
    """Fetches market data from Yahoo Finance via the ``yfinance`` library.

    Usage:
        connector = YFinanceConnector(config={"rate_limit_delay_sec": 0.5})
        df = connector.fetch_historical(
            tickers=["AAPL", "MSFT"],
            start_date=pd.Timestamp("2023-01-01"),
            end_date=pd.Timestamp("2024-01-01"),
        )
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._batch_size: int = self._config.get("batch_size", 50)

    def get_source_name(self) -> str:
        return "yfinance"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars from Yahoo Finance.

        Downloads data in batches to respect rate limits. Normalises
        output to the ``(date, ticker)`` MultiIndex schema.

        Args:
            tickers: Ticker symbols to fetch.
            start_date: Start date (inclusive).
            end_date: End date (exclusive).
            fields: Columns to keep (default: OHLCV).

        Returns:
            DataFrame with MultiIndex ``(date, ticker)`` and OHLCV columns.
        """
        try:
            import yfinance as yf  # noqa: F401 — imported here to keep dep optional
        except ImportError:
            logger.error(
                "yfinance is not installed. Run: pip install yfinance"
            )
            return self._empty_ohlcv()

        all_frames: list[pd.DataFrame] = []

        for batch_start in range(0, len(tickers), self._batch_size):
            batch = tickers[batch_start : batch_start + self._batch_size]
            batch_str = " ".join(batch)

            try:
                raw = self._retry_with_backoff(
                    yf.download,
                    batch_str,
                    start=str(start_date.date()),
                    end=str(end_date.date()),
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception:
                logger.exception(
                    "Failed to download batch starting at index %d",
                    batch_start,
                )
                continue

            if raw.empty:
                continue

            df = self._reshape_yfinance(raw, batch)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            logger.warning("No data fetched from yfinance for %d tickers", len(tickers))
            return self._empty_ohlcv()

        result = pd.concat(all_frames)

        fields = fields or ["open", "high", "low", "close", "volume"]
        available = [f for f in fields if f in result.columns]
        if available:
            result = result[available]

        meta = self._track_ingestion(tickers, len(result), start_date, end_date)
        logger.info("yfinance fetch complete: %s", meta)
        return result.sort_index()

    def fetch_latest(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch the most recent bar for each ticker.

        Uses yfinance's ``period="1d"`` to get the last trading day's
        data. Suitable for daily polling, not real-time.

        Args:
            tickers: Ticker symbols to fetch.

        Returns:
            DataFrame with MultiIndex ``(date, ticker)`` and OHLCV columns.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance is not installed.")
            return self._empty_ohlcv()

        batch_str = " ".join(tickers)

        try:
            raw = self._retry_with_backoff(
                yf.download,
                batch_str,
                period="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception:
            logger.exception("Failed to fetch latest bars from yfinance")
            return self._empty_ohlcv()

        if raw.empty:
            return self._empty_ohlcv()

        return self._reshape_yfinance(raw, tickers).sort_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_yfinance(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Reshape yfinance output into ``(date, ticker)`` MultiIndex.

        yfinance returns different shapes depending on whether one or
        multiple tickers were requested:
        - Single ticker: simple DatetimeIndex with OHLCV columns
        - Multiple tickers: MultiIndex columns ``(ticker, field)``

        This method normalises both cases.
        """
        frames: list[pd.DataFrame] = []

        if len(tickers) == 1:
            # Single-ticker download: flat DatetimeIndex
            ticker = tickers[0]
            df = raw.copy()
            df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
            # Handle MultiIndex columns from newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            df["ticker"] = ticker
            df.index.name = "date"
            df = df.reset_index().set_index(["date", "ticker"])
            frames.append(df)
        else:
            # Multi-ticker download: MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in tickers:
                    try:
                        ticker_df = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else None
                    except (KeyError, TypeError):
                        ticker_df = None

                    if ticker_df is None or ticker_df.empty:
                        continue

                    ticker_df.columns = [c.lower() for c in ticker_df.columns]
                    ticker_df["ticker"] = ticker
                    ticker_df.index.name = "date"
                    ticker_df = ticker_df.reset_index().set_index(["date", "ticker"])
                    ticker_df = ticker_df.dropna(subset=["close"])
                    frames.append(ticker_df)
            else:
                # Fallback: flat columns (single ticker returned despite multi-request)
                df = raw.copy()
                df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
                if len(tickers) == 1:
                    df["ticker"] = tickers[0]
                else:
                    df["ticker"] = tickers[0]  # best effort
                df.index.name = "date"
                df = df.reset_index().set_index(["date", "ticker"])
                frames.append(df)

        if not frames:
            return BaseConnector._empty_ohlcv()

        result = pd.concat(frames)

        # Normalise column names
        col_map = {
            "adj close": "close",
            "adj_close": "close",
        }
        result = result.rename(columns=col_map)

        # Keep only OHLCV columns that exist
        ohlcv = ["open", "high", "low", "close", "volume"]
        available = [c for c in ohlcv if c in result.columns]
        return result[available] if available else result
