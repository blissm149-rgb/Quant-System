"""News data ingestion module.

Loads news articles and headlines with timestamps for sentiment analysis.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """A single news article record."""

    article_id: str
    ticker: str
    headline: str
    summary: str
    source: str
    published_at: pd.Timestamp
    relevance_score: float  # 0-1


class NewsIngestion:
    """Ingests news data from configured sources."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._data_source = cfg.get("news", "refinitiv")
        self._min_relevance = cfg.get("min_news_relevance", 0.5)

    @classmethod
    def from_config_file(cls, config_path: str) -> "NewsIngestion":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config.get("data_sources", {}))

    def load_articles(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Load news articles for given tickers and date range.

        Args:
            tickers: Tickers to load news for.
            start_date: Start of date range.
            end_date: End of date range (exclusive, point-in-time boundary).

        Returns:
            DataFrame with columns matching NewsArticle fields.
        """
        logger.info("Loading news for %d tickers from %s to %s",
                     len(tickers), start_date, end_date)
        return pd.DataFrame(columns=[
            "article_id", "ticker", "headline", "summary",
            "source", "published_at", "relevance_score",
        ])
