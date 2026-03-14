"""Sentiment model for news text.

Produces per-article sentiment scores in [-1, +1].
Uses a simple lexicon-based approach as fallback; FinBERT for production.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Simple financial sentiment lexicon
POSITIVE_WORDS = {
    "upgrade", "beat", "exceeded", "strong", "growth", "profit", "gain",
    "surge", "rally", "bullish", "outperform", "positive", "higher",
    "record", "boost", "improve", "optimistic", "upside", "recovery",
}

NEGATIVE_WORDS = {
    "downgrade", "miss", "missed", "weak", "loss", "decline", "drop",
    "fall", "bearish", "underperform", "negative", "lower", "cut",
    "warning", "risk", "pessimistic", "downside", "recession", "crash",
}


class SentimentModel:
    """Produces sentiment scores for news text.

    Uses a lexicon-based approach as a lightweight default.
    For production, replace with a fine-tuned FinBERT model.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._model_type = cfg.get("sentiment_model", "lexicon")

    def score_text(self, text: str) -> float:
        """Score a single text string for sentiment.

        Args:
            text: Cleaned text string.

        Returns:
            Sentiment score in [-1, +1].
        """
        if not text:
            return 0.0

        words = set(text.lower().split())
        pos_count = len(words & POSITIVE_WORDS)
        neg_count = len(words & NEGATIVE_WORDS)
        total = pos_count + neg_count

        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score multiple texts.

        Args:
            texts: List of cleaned text strings.

        Returns:
            List of sentiment scores.
        """
        return [self.score_text(t) for t in texts]

    def score_dataframe(self, df: pd.DataFrame, text_column: str = "headline") -> pd.Series:
        """Score all articles in a DataFrame.

        Args:
            df: DataFrame with a text column.
            text_column: Name of the column containing text.

        Returns:
            Series of sentiment scores aligned with DataFrame index.
        """
        if df.empty or text_column not in df.columns:
            return pd.Series(dtype=float)

        return df[text_column].apply(self.score_text)
