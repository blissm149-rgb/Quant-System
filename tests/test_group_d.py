"""Tests for Group D — Alternative data."""

import numpy as np
import pandas as pd
import pytest

from quant_fund.alternative_data.analyst_estimates.analyst_data_ingestion import AnalystDataIngestion
from quant_fund.alternative_data.analyst_estimates.estimate_revision_features import EstimateRevisionFeatures
from quant_fund.alternative_data.options_data.options_chain_ingestion import OptionsChainIngestion
from quant_fund.alternative_data.options_data.implied_volatility_surface_builder import ImpliedVolatilitySurfaceBuilder
from quant_fund.alternative_data.options_data.options_feature_generation import OptionsFeatureGeneration
from quant_fund.alternative_data.news_sentiment.news_ingestion import NewsIngestion
from quant_fund.alternative_data.news_sentiment.text_cleaning import TextCleaner
from quant_fund.alternative_data.news_sentiment.sentiment_model import SentimentModel
from quant_fund.alternative_data.news_sentiment.sentiment_feature_generation import SentimentFeatureGeneration
from quant_fund.alternative_data.etf_flows.etf_flow_ingestion import ETFFlowIngestion
from quant_fund.alternative_data.etf_flows.flow_feature_generation import FlowFeatureGeneration
from quant_fund.alternative_data.macro_data.macro_data_ingestion import MacroDataIngestion
from quant_fund.alternative_data.macro_data.macro_feature_generation import MacroFeatureGeneration
from quant_fund.alternative_data.macro_data.macro_regime_classifier import MacroRegimeClassifier, MacroRegime

pytestmark = [pytest.mark.tier2]


class TestAnalystEstimates:

    def test_consensus_computation(self):
        """Consensus should only use estimates published before as_of."""
        ingestion = AnalystDataIngestion()
        estimates = pd.DataFrame({
            "ticker": ["AAPL"] * 5,
            "analyst_id": ["A1", "A2", "A3", "A4", "A5"],
            "metric": ["eps"] * 5,
            "period": ["FY1"] * 5,
            "estimate_value": [5.0, 5.5, 4.8, 5.2, 5.1],
            "estimate_date": pd.date_range("2020-01-01", periods=5),
            "period_end_date": [pd.Timestamp("2020-12-31")] * 5,
        })

        as_of = pd.Timestamp("2020-01-04")  # only 3 estimates available
        consensus = ingestion.compute_consensus(estimates, as_of=as_of)
        assert len(consensus) == 1
        assert consensus.iloc[0]["n_analysts"] == 3

    def test_consensus_excludes_future(self):
        """Estimates after as_of must not be included."""
        ingestion = AnalystDataIngestion()
        estimates = pd.DataFrame({
            "ticker": ["AAPL"] * 4,
            "analyst_id": ["A1", "A2", "A3", "A4"],
            "metric": ["eps"] * 4,
            "period": ["FY1"] * 4,
            "estimate_value": [5.0, 5.5, 4.8, 100.0],  # 100 is future
            "estimate_date": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
                pd.Timestamp("2020-06-01"),  # future
            ],
            "period_end_date": [pd.Timestamp("2020-12-31")] * 4,
        })

        as_of = pd.Timestamp("2020-02-01")
        consensus = ingestion.compute_consensus(estimates, as_of=as_of)
        assert consensus.iloc[0]["n_analysts"] == 3
        # Mean should not include the 100.0 outlier
        assert consensus.iloc[0]["consensus_mean"] < 10

    def test_sue_computation(self):
        features = EstimateRevisionFeatures()
        sue = features.compute_sue(actual=5.5, consensus_mean=5.0, consensus_std=0.3)
        assert abs(sue - 1.667) < 0.01


class TestOptionsData:

    def test_iv_surface_builder(self):
        builder = ImpliedVolatilitySurfaceBuilder()
        chain = pd.DataFrame({
            "strike": [90, 95, 100, 105, 110],
            "expiry": ["2020-03-20"] * 5,
            "implied_vol": [0.30, 0.25, 0.22, 0.24, 0.28],
        })
        surface = builder.build_surface(chain, spot_price=100.0, as_of=pd.Timestamp("2020-01-15"))
        assert len(surface["moneyness"]) == 5
        atm_iv = builder.get_atm_iv(surface)
        assert 0.20 < atm_iv < 0.35

    def test_put_call_ratio(self):
        features = OptionsFeatureGeneration()
        chain = pd.DataFrame({
            "option_type": ["call", "call", "put", "put", "put"],
            "volume": [1000, 500, 800, 600, 400],
        })
        pcr = features.compute_put_call_ratio(chain)
        assert pcr == pytest.approx(1800 / 1500, rel=0.01)


class TestNewsSentiment:

    def test_text_cleaning(self):
        cleaner = TextCleaner()
        raw = "UPDATE 1 - Stock <b>rises</b> https://example.com on earnings"
        clean = cleaner.clean_headline(raw)
        assert "https" not in clean
        assert "<b>" not in clean
        assert "UPDATE" not in clean

    def test_sentiment_scoring(self):
        model = SentimentModel()
        pos = model.score_text("Company beat earnings expectations with strong growth")
        neg = model.score_text("Stock crash amid recession fears and weak outlook")
        neutral = model.score_text("Trading volume was average today")
        assert pos > 0
        assert neg < 0
        assert abs(neutral) < abs(pos)

    def test_sentiment_batch(self):
        model = SentimentModel()
        scores = model.score_batch(["positive upgrade", "negative downgrade", ""])
        assert len(scores) == 3
        assert scores[0] > 0
        assert scores[1] < 0


class TestETFFlows:

    def test_flow_feature_generation(self):
        features = FlowFeatureGeneration()
        assert features.feature_name == "etf_flow"


class TestMacroData:

    def test_macro_regime_goldilocks(self):
        classifier = MacroRegimeClassifier()
        regime = classifier.classify(gdp_growth=0.03, inflation_rate=0.02)
        assert regime == MacroRegime.GOLDILOCKS

    def test_macro_regime_stagflation(self):
        classifier = MacroRegimeClassifier()
        regime = classifier.classify(gdp_growth=-0.01, inflation_rate=0.05)
        assert regime == MacroRegime.STAGFLATION

    def test_macro_regime_deflation(self):
        classifier = MacroRegimeClassifier()
        regime = classifier.classify(gdp_growth=-0.02, inflation_rate=0.01)
        assert regime == MacroRegime.DEFLATION

    def test_regime_strategy_adjustments(self):
        classifier = MacroRegimeClassifier()
        adj = classifier.get_regime_strategy_adjustments(MacroRegime.STAGFLATION)
        assert adj["quality"] > 1.0  # quality favoured in stagflation
        assert adj["momentum"] < 1.0  # momentum reduced

    def test_yield_curve_slope(self):
        gen = MacroFeatureGeneration()
        df = pd.DataFrame({
            "treasury_10y": [3.0, 3.1, 3.2],
            "treasury_2y": [2.5, 2.6, 2.7],
        })
        slope = gen.compute_yield_curve_slope(df)
        assert len(slope) == 3
        assert (slope == 0.5).all()
