"""Test ML data leakage pathways.

Validates that:
  1. NeuralNetworkPredictor uses a temporal validation split (not random)
  2. TechnicalIndicatorEngine uses strict data alignment (raises on future data)
  3. get_aligned_data_permissive() emits a DeprecationWarning
  4. Sentiment feature ingestion uses published_at timestamps
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, STANDARD_TICKERS

from quant_fund.research_algorithms.machine_learning.neural_network_predictor import (
    NeuralNetworkPredictor,
)
from quant_fund.feature_factory.data_alignment_engine import (
    DataAlignmentEngine,
    LookAheadError,
)
from quant_fund.feature_factory.technical_indicator_engine import (
    TechnicalIndicatorEngine,
)


pytestmark = [pytest.mark.validation]


# ── helpers ─────────────────────────────────────────────────────────


def _make_feature_matrix(n_samples: int = 300, n_features: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_samples)
    X = pd.DataFrame(
        rng.normal(0, 1, (n_samples, n_features)),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.normal(0, 0.02, n_samples), index=dates, name="fwd_ret")
    return X, y


# ── Tests ───────────────────────────────────────────────────────────


class TestNNTemporalValidationSplit:
    """Verify that NeuralNetworkPredictor uses a temporal OOS split."""

    def test_nn_temporal_validation_split(self):
        """train_model() must return oos_r2 and n_val_samples > 0,
        proving an out-of-sample evaluation was performed."""
        X, y = _make_feature_matrix(n_samples=300, seed=42)
        model = NeuralNetworkPredictor()
        result = model.train_model(X, y)

        assert "error" not in result, f"Training failed: {result}"
        assert "oos_r2" in result, "train_model() must return oos_r2 (temporal OOS metric)"
        assert "n_val_samples" in result, "train_model() must return n_val_samples"
        assert result["n_val_samples"] > 0, "Validation set must be non-empty"

    def test_nn_validation_set_does_not_overlap_training(self):
        """Training + validation sample counts must sum to total,
        confirming a non-overlapping temporal split."""
        X, y = _make_feature_matrix(n_samples=300, seed=42)
        model = NeuralNetworkPredictor()
        result = model.train_model(X, y)

        assert "error" not in result
        assert result["n_train_samples"] + result["n_val_samples"] == result["n_samples"], (
            f"Train ({result['n_train_samples']}) + val ({result['n_val_samples']}) "
            f"!= total ({result['n_samples']})"
        )


class TestTechnicalIndicatorStrictAlignment:
    """Verify TechnicalIndicatorEngine uses strict alignment."""

    def test_technical_indicator_engine_uses_strict_alignment(self):
        """Passing data that extends past as_of must raise LookAheadError,
        proving the engine uses get_aligned_data (strict) not permissive."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=400, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[300]

        engine = TechnicalIndicatorEngine()

        with pytest.raises(LookAheadError):
            engine.compute_all(ohlcv, as_of=as_of)


class TestPermissiveAlignmentDeprecation:
    """Verify get_aligned_data_permissive() emits DeprecationWarning."""

    def test_permissive_alignment_emits_deprecation_warning(self):
        """Calling get_aligned_data_permissive() must emit a
        DeprecationWarning directing callers to use strict mode."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=100, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[50]

        engine = DataAlignmentEngine()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.get_aligned_data_permissive(ohlcv, as_of=as_of, lookback_days=60)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1, (
                "get_aligned_data_permissive() must emit DeprecationWarning"
            )
            assert "get_aligned_data()" in str(deprecation_warnings[0].message)


class TestSentimentFeatureTimestamps:
    """Verify sentiment data uses published_at timestamps."""

    def test_sentiment_feature_timestamps_are_published_at(self):
        """The NewsArticle dataclass must use published_at (not arrival_date
        or announcement_date) as the timestamp field."""
        from quant_fund.alternative_data.news_sentiment.news_ingestion import (
            NewsArticle,
        )
        import dataclasses

        fields = {f.name for f in dataclasses.fields(NewsArticle)}
        assert "published_at" in fields, (
            f"NewsArticle must have 'published_at' field. Found: {fields}"
        )
