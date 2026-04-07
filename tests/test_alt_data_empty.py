"""Tests verifying the system runs without errors when alternative data
modules (news, analyst estimates, options, ETF flows, macro) return
empty DataFrames — which is the current state of all ingestion stubs.

These tests exercise the full chain:
  ingestion (returns empty) → feature generation (returns zeros/empty)
  → research runner (completes with features or graceful skip)
  → paper trading runner (no crash, no orders generated)
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.alternative_data.news_sentiment.news_ingestion import NewsIngestion
from quant_fund.alternative_data.news_sentiment.sentiment_feature_generation import (

    SentimentFeatureGeneration,
)
from quant_fund.alternative_data.news_sentiment.sentiment_model import SentimentModel
from quant_fund.alternative_data.news_sentiment.text_cleaning import TextCleaner
from quant_fund.alternative_data.analyst_estimates.analyst_data_ingestion import (
    AnalystDataIngestion,
)

pytestmark = [pytest.mark.tier2]
from quant_fund.alternative_data.analyst_estimates.estimate_revision_features import (
    EstimateRevisionFeatures,
)
from quant_fund.alternative_data.options_data.options_chain_ingestion import (
    OptionsChainIngestion,
)
from quant_fund.alternative_data.options_data.options_feature_generation import (
    OptionsFeatureGeneration,
)
from quant_fund.alternative_data.etf_flows.etf_flow_ingestion import ETFFlowIngestion
from quant_fund.alternative_data.etf_flows.flow_feature_generation import (
    FlowFeatureGeneration,
)
from quant_fund.alternative_data.macro_data.macro_data_ingestion import (
    MacroDataIngestion,
)
from quant_fund.alternative_data.macro_data.macro_feature_generation import (
    MacroFeatureGeneration,
)
from quant_fund.main.research_runner import ResearchRunner


# ── Helpers ────────────────────────────────────────────────────────


def _make_ohlcv(tickers=None, n_days=60, seed=42):
    """Create synthetic OHLCV data with (date, ticker) MultiIndex."""
    rng = np.random.RandomState(seed)
    tickers = tickers or ["AAPL", "MSFT", "GOOG"]
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    rows = []
    for d in dates:
        for t in tickers:
            c = 100 + rng.randn() * 5
            rows.append({
                "date": d, "ticker": t,
                "open": c - 1, "high": c + 2, "low": c - 2,
                "close": c, "volume": rng.randint(500_000, 5_000_000),
            })
    return pd.DataFrame(rows).set_index(["date", "ticker"])


# ═══════════════════════════════════════════════════════════════════
# Ingestion modules return empty without error
# ═══════════════════════════════════════════════════════════════════


class TestIngestionStubsReturnEmpty:
    """All ingestion stubs must return empty DataFrames/Series, no exceptions."""

    def test_news_ingestion_returns_empty(self):
        ingestor = NewsIngestion()
        df = ingestor.load_articles(
            ["AAPL", "MSFT"],
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-06-01"),
        )
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_analyst_data_ingestion_returns_empty(self):
        ingestor = AnalystDataIngestion()
        df = ingestor.load_estimates(
            ["AAPL", "MSFT"],
            as_of=pd.Timestamp("2024-06-01"),
        )
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_options_chain_ingestion_returns_empty(self):
        ingestor = OptionsChainIngestion()
        df = ingestor.load_chain("AAPL", as_of=pd.Timestamp("2024-06-01"))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_etf_flow_ingestion_returns_empty(self):
        ingestor = ETFFlowIngestion()
        df = ingestor.load_flows(
            ["SPY", "QQQ"],
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-06-01"),
        )
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_macro_data_ingestion_returns_empty(self):
        ingestor = MacroDataIngestion()
        series = ingestor.load_series(
            "gdp_growth",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-06-01"),
        )
        assert isinstance(series, pd.Series)
        assert series.empty

    def test_macro_load_all_returns_empty(self):
        ingestor = MacroDataIngestion()
        df = ingestor.load_all(
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-06-01"),
        )
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ═══════════════════════════════════════════════════════════════════
# Feature generators handle empty/missing data gracefully
# ═══════════════════════════════════════════════════════════════════


class TestFeatureGeneratorsEmptyInput:
    """Feature generators must not crash on empty or column-missing data."""

    @pytest.fixture
    def ohlcv(self):
        """Standard OHLCV data — no sentiment/estimate/options columns."""
        return _make_ohlcv()

    @pytest.fixture
    def empty_df(self):
        idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return pd.DataFrame(index=idx)

    @pytest.fixture
    def as_of(self):
        return pd.Timestamp("2023-03-15")

    def test_sentiment_on_ohlcv_returns_zeros(self, ohlcv, as_of):
        """No 'sentiment_score' column → returns zeros, not exception."""
        gen = SentimentFeatureGeneration()
        result = gen.compute(ohlcv, as_of)
        assert isinstance(result, pd.Series)
        # All zeros because column is missing
        assert (result == 0.0).all()

    def test_sentiment_on_empty_df(self, empty_df, as_of):
        gen = SentimentFeatureGeneration()
        result = gen.compute(empty_df, as_of)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_sentiment_momentum_empty(self, as_of):
        gen = SentimentFeatureGeneration()
        result = gen.compute_sentiment_momentum(pd.DataFrame(), as_of)
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_estimate_revision_on_ohlcv_returns_zeros(self, ohlcv, as_of):
        """No 'estimate_value' column → returns zeros."""
        gen = EstimateRevisionFeatures()
        result = gen.compute(ohlcv, as_of)
        assert isinstance(result, pd.Series)
        assert (result == 0.0).all()

    def test_estimate_revision_momentum_empty(self, as_of):
        gen = EstimateRevisionFeatures()
        result = gen.compute_revision_momentum(pd.DataFrame(), as_of)
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_options_feature_on_ohlcv_returns_zeros(self, ohlcv, as_of):
        """No 'implied_vol' column → returns zeros."""
        gen = OptionsFeatureGeneration()
        result = gen.compute(ohlcv, as_of)
        assert isinstance(result, pd.Series)
        assert (result == 0.0).all()

    def test_flow_feature_on_ohlcv_returns_zeros(self, ohlcv, as_of):
        """No 'net_flow_pct' column → returns zeros."""
        gen = FlowFeatureGeneration()
        result = gen.compute(ohlcv, as_of)
        assert isinstance(result, pd.Series)
        assert (result == 0.0).all()

    def test_macro_feature_on_ohlcv_returns_empty(self, ohlcv, as_of):
        gen = MacroFeatureGeneration()
        result = gen.compute(ohlcv, as_of)
        assert isinstance(result, pd.Series)

    def test_all_generators_validate_empty(self):
        """validate() on empty output should not crash."""
        generators = [
            SentimentFeatureGeneration(),
            EstimateRevisionFeatures(),
            OptionsFeatureGeneration(),
            FlowFeatureGeneration(),
            MacroFeatureGeneration(),
        ]
        empty = pd.Series(dtype=float)
        for gen in generators:
            # Should not raise
            gen.validate(empty)


# ═══════════════════════════════════════════════════════════════════
# Research runner completes with alternative data generators
# ═══════════════════════════════════════════════════════════════════


class TestResearchRunnerWithAltData:
    """Research runner must complete when alternative data generators
    are wired in but their underlying data sources return empty."""

    def test_research_cycle_with_all_alt_generators(self):
        """Wire ALL alternative data feature generators into research runner.
        All will receive OHLCV data (no alt-data columns) and should
        return zeros or empty — research runner should still complete."""
        runner = ResearchRunner()
        alt_generators = [
            SentimentFeatureGeneration(),
            EstimateRevisionFeatures(),
            OptionsFeatureGeneration(),
            FlowFeatureGeneration(),
            MacroFeatureGeneration(),
        ]
        runner.inject_components(feature_generators=alt_generators)

        ohlcv = _make_ohlcv(n_days=60)
        as_of = pd.Timestamp("2023-03-15")

        result = runner.run_cycle(as_of=as_of, market_data=ohlcv)

        # Must complete, not crash
        assert result.status == "completed"
        # Feature matrix should exist (generators returned zeros)
        assert result.feature_matrix is not None
        # Should have columns for each generator
        assert "news_sentiment" in result.feature_matrix.columns
        assert "estimate_revision" in result.feature_matrix.columns

    def test_research_cycle_empty_market_data(self):
        """Empty market data should return 'no_data' status, not crash."""
        runner = ResearchRunner()
        runner.inject_components(feature_generators=[
            SentimentFeatureGeneration(),
        ])

        empty = pd.DataFrame()
        result = runner.run_cycle(
            as_of=pd.Timestamp("2024-01-01"),
            market_data=empty,
        )
        assert result.status == "no_data"

    def test_research_cycle_none_market_data_no_loader(self):
        """No data loader, no market data → no_data status."""
        runner = ResearchRunner()
        runner.inject_components(feature_generators=[
            SentimentFeatureGeneration(),
        ])

        result = runner.run_cycle(
            as_of=pd.Timestamp("2024-01-01"),
            market_data=None,
        )
        assert result.status == "no_data"


# ═══════════════════════════════════════════════════════════════════
# Text cleaning and sentiment model handle empty input
# ═══════════════════════════════════════════════════════════════════


class TestSentimentPipelineEmpty:
    """The NLP sub-pipeline must handle empty text gracefully."""

    def test_text_cleaner_empty_string(self):
        cleaner = TextCleaner()
        result = cleaner.clean("")
        assert isinstance(result, str)

    def test_text_cleaner_none_like(self):
        cleaner = TextCleaner()
        result = cleaner.clean("   ")
        assert isinstance(result, str)

    def test_sentiment_model_empty_text(self):
        model = SentimentModel()
        score = model.score_text("")
        assert isinstance(score, float)

    def test_sentiment_model_batch_empty_list(self):
        model = SentimentModel()
        scores = model.score_batch([])
        assert isinstance(scores, list)
        assert len(scores) == 0

    def test_sentiment_model_batch_empty_strings(self):
        model = SentimentModel()
        scores = model.score_batch(["", "  ", "no content"])
        assert isinstance(scores, list)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)


# ═══════════════════════════════════════════════════════════════════
# Full pipeline: paper trading with alt data generators
# ═══════════════════════════════════════════════════════════════════


class TestPaperTradingWithAltData:
    """Paper trading runner must complete a multi-day run when
    alternative data generators are present but return empty."""

    def test_paper_trading_30_days_with_alt_generators(self):
        """Run 30-day paper trading with all alt-data generators wired in.
        Since ingestion returns empty, generators return zeros, and the
        pipeline should still produce valid (zero-alpha) results."""
        from quant_fund.main.paper_trading_runner import PaperTradingRunner
        from quant_fund.broker_interface.simulation_broker import SimulationBroker

        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        ohlcv = _make_ohlcv(tickers=tickers, n_days=90)

        runner = PaperTradingRunner()
        broker = SimulationBroker(config={"initial_cash": 1_000_000})
        research = ResearchRunner()

        alt_generators = [
            SentimentFeatureGeneration(),
            EstimateRevisionFeatures(),
            OptionsFeatureGeneration(),
            FlowFeatureGeneration(),
            MacroFeatureGeneration(),
        ]
        research.inject_components(feature_generators=alt_generators)
        runner.inject_components(
            research_runner=research,
            broker=broker,
        )

        dates = list(pd.bdate_range("2023-02-01", periods=30))
        # Build market_data_by_date so research runner gets data each day
        market_data_by_date = {d: ohlcv for d in dates}
        results = runner.run(dates, market_data_by_date=market_data_by_date)

        # Must complete all 30 days without error
        assert len(results.daily_results) == 30
        for day_result in results.daily_results:
            assert day_result.status != "error", (
                f"Day {day_result.date} failed: {getattr(day_result, 'error_message', '')}"
            )

        # NAV should be unchanged (no alpha → no trades → NAV = initial)
        final_nav = broker.get_account_value()
        assert final_nav == pytest.approx(1_000_000, rel=0.01)
