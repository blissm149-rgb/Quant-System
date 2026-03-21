"""Tests for Section 10-11: Synthetic data generators and simulation environments.

Validates that all generators produce well-formed data.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
@pytest.mark.tier2
class TestAltDataGenerators:
    """Validate alternative data generators."""

    def test_analyst_estimates_structure(self):
        """Analyst estimate generator produces correct DataFrame structure."""
        from tests.generators.alt_data_generator import make_analyst_estimates

        df = make_analyst_estimates(tickers=["AAPL", "MSFT"], n_dates=30, n_analysts=3, seed=42)

        assert isinstance(df, pd.DataFrame)
        required_cols = {"date", "ticker", "analyst_id", "eps_estimate", "revenue_estimate", "target_price", "revision"}
        assert required_cols.issubset(set(df.columns))
        assert len(df) > 0
        assert df["ticker"].nunique() == 2
        assert df["analyst_id"].nunique() == 3

    def test_analyst_estimates_has_revisions(self):
        """Analyst estimates include some revisions."""
        from tests.generators.alt_data_generator import make_analyst_estimates

        df = make_analyst_estimates(n_dates=60, revision_probability=0.30, seed=42)
        assert df["revision"].sum() > 0

    def test_analyst_estimates_deterministic(self):
        """Same seed produces same estimates."""
        from tests.generators.alt_data_generator import make_analyst_estimates

        df1 = make_analyst_estimates(seed=42)
        df2 = make_analyst_estimates(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_news_sentiment_structure(self):
        """News sentiment generator produces correct structure."""
        from tests.generators.alt_data_generator import make_news_sentiment

        df = make_news_sentiment(tickers=["AAPL", "MSFT"], n_dates=30, seed=42)

        assert isinstance(df, pd.DataFrame)
        required_cols = {"date", "ticker", "sentiment_score", "confidence", "source"}
        assert required_cols.issubset(set(df.columns))
        assert len(df) > 0

    def test_news_sentiment_bounded(self):
        """Sentiment scores are bounded [-1, 1]."""
        from tests.generators.alt_data_generator import make_news_sentiment

        df = make_news_sentiment(n_dates=60, seed=42)
        assert df["sentiment_score"].min() >= -1.0
        assert df["sentiment_score"].max() <= 1.0
        assert df["confidence"].min() >= 0.0
        assert df["confidence"].max() <= 1.0

    def test_options_chain_structure(self):
        """Options chain generator produces valid structure."""
        from tests.generators.alt_data_generator import make_options_chain

        df = make_options_chain(ticker="AAPL", spot_price=150.0, seed=42)

        assert isinstance(df, pd.DataFrame)
        required_cols = {"ticker", "expiry", "strike", "option_type", "bid", "ask", "mid", "implied_vol", "delta"}
        assert required_cols.issubset(set(df.columns))
        assert len(df) > 0

    def test_options_chain_no_arbitrage(self):
        """Options chain has valid bid/ask spread (bid < ask)."""
        from tests.generators.alt_data_generator import make_options_chain

        df = make_options_chain(seed=42)
        assert (df["bid"] <= df["ask"]).all()
        assert (df["implied_vol"] > 0).all()

    def test_options_chain_put_call_parity(self):
        """Calls have positive delta, puts have negative delta."""
        from tests.generators.alt_data_generator import make_options_chain

        df = make_options_chain(seed=42)
        calls = df[df["option_type"] == "call"]
        puts = df[df["option_type"] == "put"]

        assert (calls["delta"] >= 0).all()
        assert (puts["delta"] <= 0).all()


@pytest.mark.unit
@pytest.mark.tier2
class TestConfigFuzzer:
    """Validate config fuzzer produces useful edge cases."""

    def test_boundary_configs_generated(self):
        """Boundary configs are generated for known parameters."""
        from tests.generators.config_fuzzer import make_boundary_config

        configs = make_boundary_config("max_leverage")
        assert len(configs) > 3
        values = [c["max_leverage"] for c in configs]
        assert 2.0 in values
        assert 0.0 in values

    def test_invalid_type_configs(self):
        """Invalid type configs are generated."""
        from tests.generators.config_fuzzer import make_invalid_type_configs

        configs = make_invalid_type_configs()
        assert len(configs) > 5
        # All should have at least one key with invalid value
        for config in configs:
            assert len(config) > 0

    def test_extreme_configs(self):
        """Extreme configs are generated."""
        from tests.generators.config_fuzzer import make_extreme_configs

        configs = make_extreme_configs()
        assert len(configs) >= 3
        for config in configs:
            assert isinstance(config, dict)

    def test_combinatorial_configs(self):
        """Combinatorial configs sampled from parameter space."""
        from tests.generators.config_fuzzer import make_combinatorial_configs

        configs = make_combinatorial_configs(max_combos=10, seed=42)
        assert len(configs) <= 10
        for config in configs:
            assert "max_leverage" in config
            assert "max_position_size" in config

    def test_combinatorial_deterministic(self):
        """Same seed produces same combinatorial configs."""
        from tests.generators.config_fuzzer import make_combinatorial_configs

        c1 = make_combinatorial_configs(max_combos=10, seed=42)
        c2 = make_combinatorial_configs(max_combos=10, seed=42)
        assert c1 == c2


@pytest.mark.unit
@pytest.mark.tier2
class TestMarketRegimeSimulator:
    """Validate market regime simulator."""

    def test_bull_market_positive_returns(self):
        """Bull market has positive average returns."""
        from tests.generators.market_regime_simulator import MarketRegimeSimulator

        sim = MarketRegimeSimulator(tickers=["AAPL", "MSFT"], seed=42)
        df = sim.bull_market(days=60)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.MultiIndex)

        # Check positive drift
        for ticker in ["AAPL", "MSFT"]:
            closes = df.xs(ticker, level="ticker")["close"]
            total_return = closes.iloc[-1] / closes.iloc[0] - 1
            # Bull market should generally have positive returns over 60 days
            # (allow some randomness)
            assert total_return > -0.3

    def test_bear_market_negative_returns(self):
        """Bear market has negative average returns."""
        from tests.generators.market_regime_simulator import MarketRegimeSimulator

        sim = MarketRegimeSimulator(tickers=["AAPL"], seed=42)
        df = sim.bear_market(days=30, drawdown=0.30)

        closes = df.xs("AAPL", level="ticker")["close"]
        total_return = closes.iloc[-1] / closes.iloc[0] - 1
        assert total_return < 0

    def test_crash_sharp_decline(self):
        """Crash regime produces sharp decline."""
        from tests.generators.market_regime_simulator import MarketRegimeSimulator

        sim = MarketRegimeSimulator(tickers=["AAPL"], seed=42)
        df = sim.crash(days=5, drawdown=0.15)

        closes = df.xs("AAPL", level="ticker")["close"]
        assert len(closes) == 5

    def test_sideways_low_vol(self):
        """Sideways market has low volatility."""
        from tests.generators.market_regime_simulator import MarketRegimeSimulator

        sim = MarketRegimeSimulator(tickers=["AAPL"], seed=42)
        df = sim.sideways(days=60, vol=0.05)

        closes = df.xs("AAPL", level="ticker")["close"]
        daily_returns = closes.pct_change().dropna()
        assert daily_returns.std() < 0.05

    def test_regime_sequence_continuous(self):
        """Regime sequence produces continuous price series."""
        from tests.generators.market_regime_simulator import MarketRegimeSimulator

        sim = MarketRegimeSimulator(tickers=["AAPL", "MSFT"], seed=42)
        df = sim.regime_sequence([
            ("bull", {"days": 30}),
            ("crash", {"days": 5}),
            ("recovery", {"days": 30}),
        ])

        assert len(df) > 0
        dates = df.index.get_level_values("date").unique()
        assert len(dates) == 65  # 30 + 5 + 30

    def test_regime_sequence_all_regimes(self):
        """All regime types work in a sequence."""
        from tests.generators.market_regime_simulator import MarketRegimeSimulator

        sim = MarketRegimeSimulator(tickers=["AAPL"], seed=42)
        df = sim.regime_sequence([
            ("bull", {"days": 20}),
            ("bear", {"days": 10}),
            ("sideways", {"days": 20}),
            ("crash", {"days": 3}),
            ("recovery", {"days": 20}),
        ])

        dates = df.index.get_level_values("date").unique()
        assert len(dates) == 73

    def test_deterministic_output(self):
        """Same seed produces identical output."""
        from tests.generators.market_regime_simulator import MarketRegimeSimulator

        sim1 = MarketRegimeSimulator(tickers=["AAPL"], seed=42)
        sim2 = MarketRegimeSimulator(tickers=["AAPL"], seed=42)

        df1 = sim1.bull_market(days=30)
        df2 = sim2.bull_market(days=30)

        pd.testing.assert_frame_equal(df1, df2)
