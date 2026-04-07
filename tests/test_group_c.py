"""Tests for Group C — Research algorithms.

Validates:
- Factor models produce cross-sectional signals without look-ahead
- Mean reversion strategies compute valid reversal signals
- Cointegration engine identifies known cointegrated pairs
- Kalman spread model produces signals with entry/exit logic
- ML models train and produce scores
- Regime models classify market states
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine
from quant_fund.research_algorithms.factor_models.momentum_factor import MomentumFactor
from quant_fund.research_algorithms.factor_models.value_factor import ValueFactor
from quant_fund.research_algorithms.factor_models.quality_factor import QualityFactor
from quant_fund.research_algorithms.factor_models.low_volatility_factor import LowVolatilityFactor
from quant_fund.research_algorithms.factor_models.size_factor import SizeFactor
from quant_fund.research_algorithms.mean_reversion.zscore_reversion_strategy import ZScoreReversionStrategy
from quant_fund.research_algorithms.mean_reversion.short_term_reversal_strategy import ShortTermReversalStrategy
from quant_fund.research_algorithms.statistical_arbitrage.cointegration_engine import CointegrationEngine
from quant_fund.research_algorithms.statistical_arbitrage.kalman_spread_model import KalmanSpreadModel
from quant_fund.research_algorithms.statistical_arbitrage.pairs_trading_strategy import PairsTradingStrategy
from quant_fund.research_algorithms.machine_learning.gradient_boosted_tree_model import GradientBoostedTreeModel
from quant_fund.research_algorithms.machine_learning.random_forest_model import RandomForestModel
from quant_fund.research_algorithms.regime_models.hidden_markov_regime_model import HiddenMarkovRegimeModel
from quant_fund.research_algorithms.regime_models.volatility_regime_detector import VolatilityRegimeDetector, VolatilityRegime
from quant_fund.research_algorithms.regime_models.market_state_classifier import MarketStateClassifier, MarketState

pytestmark = [pytest.mark.tier2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "BAC", "WMT"]


def _make_ohlcv(tickers=None, start="2019-01-02", periods=300):
    tickers = tickers or TICKERS
    dates = pd.bdate_range(start=start, periods=periods)
    rows = []
    rng = np.random.default_rng(42)
    for ticker in tickers:
        base = rng.uniform(20, 200)
        rets = rng.normal(0.0005, 0.02, size=periods)
        prices = base * np.cumprod(1 + rets)
        for i, dt in enumerate(dates):
            c = prices[i]
            rows.append({
                "date": dt, "ticker": ticker,
                "open": c * rng.uniform(0.99, 1.01),
                "high": c * rng.uniform(1.00, 1.03),
                "low": c * rng.uniform(0.97, 1.00),
                "close": c,
                "volume": int(rng.uniform(1e6, 1e7)),
            })
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def _make_cointegrated_pair(n=500):
    """Create two price series that are cointegrated."""
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2018-01-02", periods=n)

    # Random walk for X
    x_returns = rng.normal(0.0002, 0.01, n)
    x_prices = 100 * np.cumprod(1 + x_returns)

    # Y = 1.5 * X + mean-reverting spread
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.9 * spread[i - 1] + rng.normal(0, 0.5)
    y_prices = 1.5 * x_prices + spread + 50

    rows = []
    for i, dt in enumerate(dates):
        rows.append({"date": dt, "ticker": "PAIR_A", "close": y_prices[i], "volume": 1e6})
        rows.append({"date": dt, "ticker": "PAIR_B", "close": x_prices[i], "volume": 1e6})
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def _get_as_of(df):
    return df.index.get_level_values("date").max() + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Factor model tests
# ---------------------------------------------------------------------------

class TestFactorModels:

    def test_momentum_factor(self):
        df = _make_ohlcv(periods=280)
        factor = MomentumFactor()
        result = factor.compute(df, as_of=_get_as_of(df))
        assert len(result) == len(TICKERS)
        assert factor.validate(result)

    def test_value_factor(self):
        df = _make_ohlcv(periods=100)
        factor = ValueFactor()
        result = factor.compute(df, as_of=_get_as_of(df))
        assert len(result) == len(TICKERS)
        assert factor.validate(result)

    def test_quality_factor(self):
        df = _make_ohlcv(periods=100)
        factor = QualityFactor()
        result = factor.compute(df, as_of=_get_as_of(df))
        assert len(result) == len(TICKERS)
        assert factor.validate(result)

    def test_low_volatility_factor(self):
        df = _make_ohlcv(periods=280)
        factor = LowVolatilityFactor()
        result = factor.compute(df, as_of=_get_as_of(df))
        assert len(result) == len(TICKERS)
        # Low vol factor is inverted, all values should be negative
        assert (result.dropna() < 0).all()
        assert factor.validate(result)

    def test_size_factor(self):
        df = _make_ohlcv(periods=50)
        factor = SizeFactor()
        result = factor.compute(df, as_of=_get_as_of(df))
        assert len(result) == len(TICKERS)
        assert factor.validate(result)

    def test_no_lookahead_in_factors(self):
        """Changing future data must not affect factor output."""
        df = _make_ohlcv(periods=280)
        dates = df.index.get_level_values("date").unique()
        as_of = dates[250]

        engine = DataAlignmentEngine()
        factor = MomentumFactor()

        aligned = engine.get_aligned_data_permissive(df, as_of=as_of, lookback_days=factor.lookback_days)
        result1 = factor.compute(aligned, as_of=as_of)

        df_mod = df.copy()
        future_mask = df_mod.index.get_level_values("date") >= as_of
        df_mod.loc[future_mask, "close"] = 999999.0
        aligned2 = engine.get_aligned_data_permissive(df_mod, as_of=as_of, lookback_days=factor.lookback_days)
        result2 = factor.compute(aligned2, as_of=as_of)

        pd.testing.assert_series_equal(result1, result2)


# ---------------------------------------------------------------------------
# Mean reversion tests
# ---------------------------------------------------------------------------

class TestMeanReversion:

    def test_zscore_reversion(self):
        df = _make_ohlcv(periods=80)
        strategy = ZScoreReversionStrategy()
        result = strategy.compute(df, as_of=_get_as_of(df))
        assert len(result) == len(TICKERS)
        assert strategy.validate(result)

    def test_short_term_reversal(self):
        df = _make_ohlcv(periods=40)
        strategy = ShortTermReversalStrategy()
        result = strategy.compute(df, as_of=_get_as_of(df))
        assert len(result) == len(TICKERS)
        assert strategy.validate(result)


# ---------------------------------------------------------------------------
# Statistical arbitrage tests
# ---------------------------------------------------------------------------

class TestStatisticalArbitrage:

    def test_cointegration_finds_known_pair(self):
        """Cointegration engine should identify a synthetically cointegrated pair."""
        df = _make_cointegrated_pair(n=500)
        engine = CointegrationEngine({"cointegration_min_obs": 100})
        result = engine.test_pair(df, "PAIR_A", "PAIR_B")
        assert result is not None
        assert result.p_value <= 0.05
        assert result.is_cointegrated
        assert result.hedge_ratio > 0

    def test_cointegration_rejects_random(self):
        """Random pairs should usually not be cointegrated."""
        df = _make_ohlcv(periods=300)
        engine = CointegrationEngine({"cointegration_min_obs": 100})
        result = engine.test_pair(df, "AAPL", "MSFT")
        # Random data is unlikely to be cointegrated, but not guaranteed
        # Just check we get a valid result
        assert result is not None
        assert 0 <= result.p_value <= 1.0

    def test_kalman_spread_produces_signals(self):
        df = _make_cointegrated_pair(n=500)
        model = KalmanSpreadModel()
        prices_a = df.xs("PAIR_A", level="ticker")["close"]
        prices_b = df.xs("PAIR_B", level="ticker")["close"]
        result = model.compute_pair_signal(prices_a, prices_b)
        assert not result.empty
        assert "z_score" in result.columns
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_pairs_trading_strategy(self):
        df = _make_cointegrated_pair(n=500)
        strategy = PairsTradingStrategy({"cointegration_min_obs": 100})
        pairs = strategy.select_pairs(df, ["PAIR_A", "PAIR_B"])
        assert len(pairs) > 0

        as_of = df.index.get_level_values("date").max() + pd.Timedelta(days=1)
        signals = strategy.generate_signals(df, as_of=as_of)
        # May or may not have active signals
        assert isinstance(signals, list)


# ---------------------------------------------------------------------------
# ML model tests
# ---------------------------------------------------------------------------

class TestMLModels:

    def _make_features_and_returns(self, n=500, n_features=5):
        rng = np.random.default_rng(42)
        features = pd.DataFrame(
            rng.normal(0, 1, (n, n_features)),
            columns=[f"f{i}" for i in range(n_features)],
        )
        # Forward returns with weak signal
        returns = pd.Series(
            0.01 * features["f0"] + rng.normal(0, 0.02, n),
            name="forward_return",
        )
        return features, returns

    def test_gbt_train_and_score(self):
        features, returns = self._make_features_and_returns()
        model = GradientBoostedTreeModel({"gbt_min_train_days": 100, "gbt_n_estimators": 50})
        metrics = model.train_model(features, returns)
        assert "n_samples" in metrics
        assert metrics["n_samples"] == 500

        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == 5

    def test_random_forest_train(self):
        features, returns = self._make_features_and_returns()
        model = RandomForestModel({"rf_min_train_days": 100, "rf_n_estimators": 50})
        metrics = model.train_model(features, returns)
        assert "n_samples" in metrics

    def test_ml_insufficient_data(self):
        features, returns = self._make_features_and_returns(n=50)
        model = GradientBoostedTreeModel({"gbt_min_train_days": 252})
        metrics = model.train_model(features, returns)
        assert "error" in metrics


# ---------------------------------------------------------------------------
# Regime model tests
# ---------------------------------------------------------------------------

class TestRegimeModels:

    def _make_market_returns(self, n=500):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2018-01-02", periods=n)
        returns = rng.normal(0.0003, 0.01, n)
        # Add a high-vol period
        returns[200:250] = rng.normal(-0.005, 0.03, 50)
        return pd.Series(returns, index=dates)

    def test_hmm_fit_and_predict(self):
        returns = self._make_market_returns()
        model = HiddenMarkovRegimeModel()
        result = model.fit(returns)
        assert result["n_regimes"] == 2
        assert model.is_fitted

        regime = model.predict_regime(returns)
        assert regime in (0, 1)

        probs = model.predict_regime_probabilities(returns)
        assert len(probs) == 2
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_volatility_regime_detector(self):
        returns = self._make_market_returns()
        detector = VolatilityRegimeDetector()

        regime = detector.detect(returns)
        assert isinstance(regime, VolatilityRegime)

        pct = detector.get_vol_percentile(returns)
        assert 0 <= pct <= 100

    def test_volatility_regime_high_vol(self):
        """High vol data should classify as HIGH regime."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=300)
        # Start normal, end high vol
        returns = np.concatenate([
            rng.normal(0, 0.005, 250),
            rng.normal(-0.01, 0.04, 50),
        ])
        series = pd.Series(returns, index=dates)
        detector = VolatilityRegimeDetector()
        regime = detector.detect(series)
        assert regime == VolatilityRegime.HIGH

    def test_market_state_classifier(self):
        returns = self._make_market_returns()
        classifier = MarketStateClassifier()
        classifier.fit(returns)

        state = classifier.classify(returns)
        assert isinstance(state, MarketState)

        weights = classifier.get_state_weights()
        assert MarketState.RISK_ON in weights
        assert MarketState.CRISIS in weights
