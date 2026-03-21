"""Unit tests for alternative data modules.

TESTING_PLAN.md Section 3.16 — sentiment model, text processing.
"""

import pandas as pd
import pytest

from quant_fund.alternative_data.news_sentiment.sentiment_model import (
    SentimentModel,
)


@pytest.mark.unit
@pytest.mark.tier2
class TestSentimentModel:
    """SentimentModel — lexicon-based sentiment scoring."""

    @pytest.fixture
    def model(self):
        return SentimentModel()

    def test_positive_text(self, model):
        score = model.score_text("AAPL beat earnings with strong growth and profit surge")
        assert score > 0

    def test_negative_text(self, model):
        score = model.score_text("Company miss warning weak decline crash")
        assert score < 0

    def test_neutral_text(self, model):
        score = model.score_text("The meeting was held on Tuesday afternoon")
        assert score == 0.0

    def test_empty_text(self, model):
        assert model.score_text("") == 0.0

    def test_score_bounded(self, model):
        score = model.score_text("upgrade beat exceeded strong growth")
        assert -1 <= score <= 1

    def test_score_batch(self, model):
        texts = ["strong growth rally", "crash decline loss", "neutral sentence here"]
        scores = model.score_batch(texts)
        assert len(scores) == 3
        assert scores[0] > 0
        assert scores[1] < 0

    def test_score_dataframe(self, model):
        df = pd.DataFrame({"headline": ["beat expectations", "missed estimate"]})
        scores = model.score_dataframe(df, text_column="headline")
        assert len(scores) == 2
