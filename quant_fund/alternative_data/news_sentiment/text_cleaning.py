"""Text cleaning utilities for news sentiment pipeline."""

import re
from typing import Optional


class TextCleaner:
    """Cleans and preprocesses news text for sentiment analysis."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._max_length = cfg.get("max_text_length", 512)
        self._remove_urls = cfg.get("remove_urls", True)

    def clean(self, text: str) -> str:
        """Clean a text string for sentiment model input.

        Args:
            text: Raw text string.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        # Remove URLs
        if self._remove_urls:
            text = re.sub(r"https?://\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Truncate
        if len(text) > self._max_length:
            text = text[: self._max_length]

        return text

    def clean_headline(self, headline: str) -> str:
        """Clean a headline for sentiment analysis."""
        headline = self.clean(headline)
        # Remove common prefixes like "UPDATE 1 -"
        headline = re.sub(r"^(UPDATE \d+ -|BRIEF-|RPT-)", "", headline).strip()
        return headline
