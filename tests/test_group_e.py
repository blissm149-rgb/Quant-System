"""Tests for Group E — Alpha discovery, alpha monitoring, and representation learning.

Validates:
- Feature combinator generates candidate signals
- Symbolic regression and GA search complete within budget
- ML feature selector removes redundant features
- Signal ranking produces ranked list with IC-weighted scores
- Alpha performance tracker and IC monitor track metrics correctly
- Signal decay detector identifies decaying signals
- Strategy retirement manager handles lifecycle transitions
- Autoencoder encodes and reconstructs
- LSTM and Transformer produce latent vectors
- Cross-asset embedding captures similarity structure
- Embedding feature store respects point-in-time rules
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.alpha_discovery.feature_combinator import FeatureCombinator
from quant_fund.alpha_discovery.symbolic_regression_engine import SymbolicRegressionEngine
from quant_fund.alpha_discovery.genetic_algorithm_search import GeneticAlgorithmSearch
from quant_fund.alpha_discovery.ml_feature_selector import MLFeatureSelector
from quant_fund.alpha_discovery.signal_ranking_engine import SignalRankingEngine
from quant_fund.alpha_monitoring.alpha_performance_tracker import AlphaPerformanceTracker
from quant_fund.alpha_monitoring.information_coefficient_monitor import (

pytestmark = [pytest.mark.tier2]
    InformationCoefficientMonitor,
    AlertLevel,
)
from quant_fund.alpha_monitoring.signal_decay_detector import (
    SignalDecayDetector,
    DecayAction,
)
from quant_fund.alpha_monitoring.strategy_retirement_manager import (
    StrategyRetirementManager,
    StrategyStatus,
)
from quant_fund.representation_learning.autoencoder_model import AutoencoderModel
from quant_fund.representation_learning.temporal_model_lstm import TemporalModelLSTM
from quant_fund.representation_learning.temporal_model_transformer import (
    TemporalModelTransformer,
)
from quant_fund.representation_learning.cross_asset_embedding_model import (
    CrossAssetEmbeddingModel,
)
from quant_fund.representation_learning.embedding_feature_store import (
    EmbeddingFeatureStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "BAC", "WMT"]


def _make_feature_matrix(n_tickers=10, n_features=5, seed=42):
    """Create a feature matrix indexed by ticker."""
    rng = np.random.default_rng(seed)
    tickers = TICKERS[:n_tickers]
    data = rng.normal(0, 1, (n_tickers, n_features))
    cols = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(data, index=tickers, columns=cols)


def _make_time_series_features(n_tickers=10, n_dates=300, n_features=5, seed=42):
    """Create a time-series feature matrix with MultiIndex (date, ticker)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_dates)
    tickers = TICKERS[:n_tickers]
    rows = []
    for dt in dates:
        for ticker in tickers:
            row = {"date": dt, "ticker": ticker}
            for i in range(n_features):
                row[f"feature_{i}"] = rng.normal(0, 1)
            row["close"] = rng.uniform(50, 200)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def _make_returns(n_tickers=10, n_dates=300, seed=42):
    """Create a returns DataFrame (dates × tickers)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_dates)
    tickers = TICKERS[:n_tickers]
    data = rng.normal(0.0003, 0.02, (n_dates, n_tickers))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _make_signal_and_returns(n_dates=300, n_tickers=10, seed=42):
    """Create signal and forward return DataFrames with MultiIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_dates)
    tickers = TICKERS[:n_tickers]
    rows = []
    for dt in dates:
        for ticker in tickers:
            signal = rng.normal(0, 1)
            ret = 0.01 * signal + rng.normal(0, 0.02)
            rows.append({
                "date": dt, "ticker": ticker,
                "signal": signal, "return": ret,
            })
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df[["signal"]], df[["return"]]


# ---------------------------------------------------------------------------
# Alpha Discovery Tests
# ---------------------------------------------------------------------------

class TestFeatureCombinator:

    def test_generates_candidates(self):
        fm = _make_feature_matrix()
        combinator = FeatureCombinator({"n_random_combinations": 10})
        candidates = combinator.generate_candidates(fm)
        assert len(candidates) > 0
        for c in candidates:
            assert len(c.values) == len(fm)
            assert c.name
            assert c.expression

    def test_with_regime_labels(self):
        fm = _make_feature_matrix()
        regime = pd.Series(
            np.random.default_rng(42).choice([0, 1], size=len(fm)),
            index=fm.index,
        )
        combinator = FeatureCombinator({"n_random_combinations": 5})
        candidates = combinator.generate_candidates(fm, regime_labels=regime)
        interaction_candidates = [c for c in candidates if "interact" in c.name]
        assert len(interaction_candidates) == fm.shape[1]

    def test_insufficient_features(self):
        fm = _make_feature_matrix(n_features=1)
        combinator = FeatureCombinator()
        candidates = combinator.generate_candidates(fm)
        assert len(candidates) == 0


class TestSymbolicRegression:

    def test_search_produces_results(self):
        rng = np.random.default_rng(42)
        features = pd.DataFrame(rng.normal(0, 1, (200, 3)), columns=["a", "b", "c"])
        returns = pd.Series(0.02 * features["a"] + rng.normal(0, 0.01, 200))
        engine = SymbolicRegressionEngine({
            "sr_population_size": 20,
            "sr_generations": 3,
        })
        results = engine.search(features, returns, n_best=5)
        assert len(results) > 0
        assert results[0].fitness_ic != 0

    def test_insufficient_data(self):
        features = pd.DataFrame(np.zeros((10, 2)), columns=["a", "b"])
        returns = pd.Series(np.zeros(10))
        engine = SymbolicRegressionEngine()
        results = engine.search(features, returns)
        assert len(results) == 0


class TestGeneticAlgorithmSearch:

    def test_search_improves(self):
        def fitness_fn(params):
            # Simple quadratic: optimal at x=0.5, y=0.3
            return -(params["x"] - 0.5) ** 2 - (params["y"] - 0.3) ** 2

        ga = GeneticAlgorithmSearch({
            "ga_population_size": 20,
            "ga_generations": 15,
        })
        result = ga.search(
            param_ranges={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            fitness_fn=fitness_fn,
        )
        assert result.best_fitness > -0.1
        assert 0.0 <= result.best_params["x"] <= 1.0
        assert len(result.population_history) == 15


class TestMLFeatureSelector:

    def test_selects_predictive_features(self):
        rng = np.random.default_rng(42)
        n = 500
        predictive = rng.normal(0, 1, n)
        noise1 = rng.normal(0, 1, n)
        noise2 = rng.normal(0, 1, n)
        redundant = predictive + rng.normal(0, 0.01, n)  # almost identical

        features = pd.DataFrame({
            "predictive": predictive,
            "noise1": noise1,
            "noise2": noise2,
            "redundant": redundant,
        })
        target = pd.Series(0.1 * predictive + rng.normal(0, 0.01, n))

        selector = MLFeatureSelector({"fs_correlation_threshold": 0.85})
        result = selector.select(features, target)

        assert "predictive" in result.selected_features
        # redundant should be removed due to high correlation
        assert "redundant" in result.removed_features
        assert len(result.selected_features) < len(features.columns)


class TestSignalRankingEngine:

    def test_evaluate_signal(self):
        signals, returns = _make_signal_and_returns()
        engine = SignalRankingEngine({"min_eval_days": 50})
        score = engine.evaluate_signal(signals, returns)
        assert score.ic_mean != 0
        assert score.ic_tstat != 0

    def test_combine_signals(self):
        sig1 = pd.Series({"AAPL": 0.5, "MSFT": -0.3, "GOOG": 0.1})
        sig2 = pd.Series({"AAPL": -0.2, "MSFT": 0.4, "GOOG": 0.3})
        engine = SignalRankingEngine()
        combined = engine.combine({"sig1": sig1, "sig2": sig2})
        assert len(combined) == 3
        assert not combined.isna().any()

    def test_combine_with_weights(self):
        sig1 = pd.Series({"AAPL": 1.0, "MSFT": 0.0})
        sig2 = pd.Series({"AAPL": 0.0, "MSFT": 1.0})
        engine = SignalRankingEngine()
        combined = engine.combine(
            {"sig1": sig1, "sig2": sig2},
            weights={"sig1": 0.7, "sig2": 0.3},
        )
        assert combined["AAPL"] == pytest.approx(0.7, abs=0.01)
        assert combined["MSFT"] == pytest.approx(0.3, abs=0.01)


# ---------------------------------------------------------------------------
# Alpha Monitoring Tests
# ---------------------------------------------------------------------------

class TestAlphaPerformanceTracker:

    def test_tracks_ic(self):
        tracker = AlphaPerformanceTracker()
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=100)

        for dt in dates:
            signal = pd.Series(rng.normal(0, 1, 10), index=TICKERS)
            returns = pd.Series(
                0.01 * signal + rng.normal(0, 0.02, 10), index=TICKERS
            )
            tracker.update("test_signal", signal, returns, dt)

        perf = tracker.get_performance("test_signal")
        assert perf is not None
        assert perf.n_days == 100
        assert perf.realised_ic != 0
        assert perf.signal_name == "test_signal"

    def test_multiple_signals(self):
        tracker = AlphaPerformanceTracker()
        rng = np.random.default_rng(42)
        dt = pd.Timestamp("2020-01-02")
        signal = pd.Series(rng.normal(0, 1, 10), index=TICKERS)
        returns = pd.Series(rng.normal(0, 0.02, 10), index=TICKERS)

        tracker.update("sig_a", signal, returns, dt)
        tracker.update("sig_b", signal, returns, dt)
        assert len(tracker.get_signal_names()) == 2


class TestICMonitor:

    def test_detects_degradation(self):
        monitor = InformationCoefficientMonitor({
            "ic_short_window": 10,
            "ic_long_window": 30,
        })
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=60)
        all_alerts = []

        for i, dt in enumerate(dates):
            signal = pd.Series(rng.normal(0, 1, 10), index=TICKERS)
            # First 30 days: good signal; last 30 days: noise
            if i < 30:
                returns = pd.Series(
                    0.05 * signal + rng.normal(0, 0.01, 10), index=TICKERS
                )
            else:
                returns = pd.Series(rng.normal(0, 0.02, 10), index=TICKERS)
            alerts = monitor.update("test", signal, returns, dt)
            all_alerts.extend(alerts)

        assert len(all_alerts) > 0

    def test_rolling_ic_series(self):
        monitor = InformationCoefficientMonitor({"ic_short_window": 5})
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=20)
        for dt in dates:
            sig = pd.Series(rng.normal(0, 1, 10), index=TICKERS)
            ret = pd.Series(rng.normal(0, 0.02, 10), index=TICKERS)
            monitor.update("s1", sig, ret, dt)

        ic = monitor.get_current_ic("s1")
        assert ic is not None


class TestSignalDecayDetector:

    def test_detects_fast_decay(self):
        detector = SignalDecayDetector({
            "decay_min_history": 30,
            "decay_review_threshold": 20,
            "decay_retire_threshold": 10,
        })
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=100)

        # Simulate fast-decaying IC with AR(1) coefficient close to 1
        ic = 0.05
        for dt in dates:
            ic = 0.8 * ic + rng.normal(0, 0.01)
            detector.update("fast_decay", ic, dt)

        result = detector.detect("fast_decay")
        assert result is not None
        assert result.ic_half_life_days < 100  # Should detect some decay

    def test_insufficient_history(self):
        detector = SignalDecayDetector({"decay_min_history": 60})
        dates = pd.bdate_range("2020-01-02", periods=10)
        for dt in dates:
            detector.update("short", 0.05, dt)
        assert detector.detect("short") is None


class TestStrategyRetirementManager:

    def test_lifecycle_transitions(self):
        manager = StrategyRetirementManager({
            "review_ic_threshold": 0.01,
            "suspend_ic_threshold": 0.0,
            "recovery_ic_threshold": 0.03,
            "suspend_after_review_days": 5,
        })
        dt = pd.Timestamp("2020-01-02")
        manager.register_strategy("strat_a", dt)
        state = manager.get_state("strat_a")
        assert state.status == StrategyStatus.ACTIVE

        # Degrade IC -> UNDER_REVIEW
        state = manager.update("strat_a", 0.005, dt + pd.Timedelta(days=1))
        assert state.status == StrategyStatus.UNDER_REVIEW
        assert state.allocation_multiplier < 1.0

        # Recovery -> ACTIVE
        state = manager.update("strat_a", 0.05, dt + pd.Timedelta(days=2))
        assert state.status == StrategyStatus.ACTIVE
        assert state.allocation_multiplier == 1.0

    def test_suspend_after_prolonged_review(self):
        manager = StrategyRetirementManager({
            "review_ic_threshold": 0.01,
            "suspend_ic_threshold": 0.0,
            "suspend_after_review_days": 5,
        })
        dt = pd.Timestamp("2020-01-02")
        manager.register_strategy("strat_b", dt)

        # Enter review
        manager.update("strat_b", 0.005, dt + pd.Timedelta(days=1))
        # Keep negative IC for days
        state = manager.update("strat_b", -0.01, dt + pd.Timedelta(days=10))
        assert state.status == StrategyStatus.SUSPENDED
        assert state.allocation_multiplier == 0.0

    def test_manual_retirement(self):
        manager = StrategyRetirementManager()
        dt = pd.Timestamp("2020-01-02")
        manager.register_strategy("strat_c", dt)
        state = manager.retire_strategy("strat_c", dt + pd.Timedelta(days=1))
        assert state.status == StrategyStatus.RETIRED
        assert state.allocation_multiplier == 0.0


# ---------------------------------------------------------------------------
# Representation Learning Tests
# ---------------------------------------------------------------------------

class TestAutoencoder:

    def test_fit_and_encode(self):
        rng = np.random.default_rng(42)
        fm = pd.DataFrame(
            rng.normal(0, 1, (100, 10)),
            columns=[f"f{i}" for i in range(10)],
        )
        model = AutoencoderModel({
            "ae_latent_dim": 4,
            "ae_hidden_dim": 16,
            "ae_epochs": 20,
        })
        metrics = model.fit(fm)
        assert "final_loss" in metrics
        assert model.is_fitted

        encoded = model.encode(fm)
        assert encoded.shape == (100, 4)
        assert not encoded.isna().any().any()

    def test_reconstruction(self):
        rng = np.random.default_rng(42)
        fm = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=[f"f{i}" for i in range(5)])
        model = AutoencoderModel({"ae_latent_dim": 3, "ae_hidden_dim": 10, "ae_epochs": 50})
        model.fit(fm)
        error = model.reconstruction_error(fm)
        assert error < 5.0  # should be reasonable after training

    def test_encode_before_fit_raises(self):
        model = AutoencoderModel()
        fm = pd.DataFrame(np.zeros((5, 3)), columns=["a", "b", "c"])
        with pytest.raises(RuntimeError):
            model.encode(fm)


class TestTemporalModelLSTM:

    def test_fit_and_encode(self):
        rng = np.random.default_rng(42)
        sequences = {}
        dates = pd.bdate_range("2019-01-02", periods=80)
        for ticker in TICKERS[:5]:
            data = pd.DataFrame(
                rng.normal(0, 1, (80, 4)),
                index=dates,
                columns=[f"f{i}" for i in range(4)],
            )
            sequences[ticker] = data

        model = TemporalModelLSTM({
            "lstm_sequence_length": 20,
            "lstm_hidden_dim": 16,
            "lstm_output_dim": 8,
        })
        metrics = model.fit(sequences)
        assert "n_sequences" in metrics
        assert model.is_fitted

        # Build MultiIndex data for encode
        rows = []
        for ticker, df in sequences.items():
            for dt, row in df.iterrows():
                entry = row.to_dict()
                entry["date"] = dt
                entry["ticker"] = ticker
                rows.append(entry)
        multi_df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

        as_of = dates[-1] + pd.Timedelta(days=1)
        encoded = model.encode(multi_df, as_of=as_of)
        assert len(encoded) > 0
        assert encoded.shape[1] == 8


class TestTemporalModelTransformer:

    def test_fit_and_encode(self):
        rng = np.random.default_rng(42)
        sequences = {}
        dates = pd.bdate_range("2019-01-02", periods=80)
        for ticker in TICKERS[:5]:
            data = pd.DataFrame(
                rng.normal(0, 1, (80, 4)),
                index=dates,
                columns=[f"f{i}" for i in range(4)],
            )
            sequences[ticker] = data

        model = TemporalModelTransformer({
            "transformer_sequence_length": 20,
            "transformer_d_model": 16,
            "transformer_n_heads": 4,
            "transformer_output_dim": 8,
        })
        metrics = model.fit(sequences)
        assert model.is_fitted

        rows = []
        for ticker, df in sequences.items():
            for dt, row in df.iterrows():
                entry = row.to_dict()
                entry["date"] = dt
                entry["ticker"] = ticker
                rows.append(entry)
        multi_df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

        as_of = dates[-1] + pd.Timedelta(days=1)
        encoded = model.encode(multi_df, as_of=as_of)
        assert len(encoded) > 0
        assert encoded.shape[1] == 8


class TestCrossAssetEmbedding:

    def test_fit_and_similarity(self):
        returns = _make_returns(n_tickers=10, n_dates=200)
        model = CrossAssetEmbeddingModel({"cross_asset_embedding_dim": 5})
        metrics = model.fit(returns)
        assert "n_tickers" in metrics
        assert model.is_fitted

        embeddings = model.get_embeddings()
        assert len(embeddings) == 10

        sim = model.get_similarity_matrix()
        assert sim.shape == (10, 10)
        # Diagonal should be ~1.0
        for ticker in sim.index:
            assert sim.loc[ticker, ticker] == pytest.approx(1.0, abs=0.01)

    def test_find_similar(self):
        returns = _make_returns(n_tickers=10, n_dates=200)
        model = CrossAssetEmbeddingModel()
        model.fit(returns)
        similar = model.find_similar("AAPL", top_n=3)
        assert len(similar) == 3
        assert "AAPL" not in similar.index


class TestEmbeddingFeatureStore:

    def test_store_and_retrieve(self):
        store = EmbeddingFeatureStore()
        emb = pd.DataFrame(
            {"e0": [0.1, 0.2], "e1": [0.3, 0.4]},
            index=["AAPL", "MSFT"],
        )
        dt = pd.Timestamp("2020-06-15")
        store.store("autoencoder", dt, emb)

        result = store.retrieve("autoencoder", as_of=pd.Timestamp("2020-06-16"))
        assert result is not None
        assert len(result) == 2
        assert "e0" in result.columns

    def test_point_in_time_filtering(self):
        store = EmbeddingFeatureStore()
        emb = pd.DataFrame({"e0": [1.0]}, index=["AAPL"])

        store.store("model_a", pd.Timestamp("2020-01-10"), emb)
        store.store("model_a", pd.Timestamp("2020-01-20"), emb * 2)

        # Retrieve before first date
        assert store.retrieve("model_a", as_of=pd.Timestamp("2020-01-05")) is None

        # Retrieve between dates — should get first
        result = store.retrieve("model_a", as_of=pd.Timestamp("2020-01-15"))
        assert result is not None
        assert result.loc["AAPL", "e0"] == pytest.approx(1.0)

        # Retrieve after both — should get latest
        result = store.retrieve("model_a", as_of=pd.Timestamp("2020-01-25"))
        assert result is not None
        assert result.loc["AAPL", "e0"] == pytest.approx(2.0)

    def test_nonexistent_model(self):
        store = EmbeddingFeatureStore()
        assert store.retrieve("nonexistent", as_of=pd.Timestamp("2020-01-01")) is None
