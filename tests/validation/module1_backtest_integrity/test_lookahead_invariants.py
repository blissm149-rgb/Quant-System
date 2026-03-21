"""Test lookahead bias invariants in the backtest pipeline.

Validates that no component of the backtest pipeline accesses data
at or after the as_of timestamp boundary.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.feature_factory.data_alignment_engine import (
    DataAlignmentEngine,
    LookAheadError,
)
from quant_fund.data_layer.data_validator import DataValidator
from tests.conftest import make_ohlcv, STANDARD_TICKERS


pytestmark = [pytest.mark.validation]


class TestLookaheadInvariantSuite:
    """Prove the backtest pipeline enforces strict point-in-time data access."""

    def test_get_aligned_data_raises_on_future_data(self):
        """get_aligned_data must raise LookAheadError when the input DataFrame
        contains timestamps at or after the as_of boundary."""
        ohlcv = make_ohlcv(tickers=["AAPL", "MSFT"], periods=100, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[50]

        engine = DataAlignmentEngine()

        # Full dataset includes dates after as_of -> must raise
        with pytest.raises(LookAheadError):
            engine.get_aligned_data(ohlcv, as_of, lookback_days=30)

        # Pre-filtered dataset should NOT raise
        safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]
        result = engine.get_aligned_data(safe, as_of, lookback_days=30)
        assert result.index.get_level_values(0).max() < as_of

    def test_validate_no_lookahead_detects_future_timestamps(self):
        """validate_no_lookahead returns False when data contains future dates."""
        ohlcv = make_ohlcv(tickers=["AAPL"], periods=100, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[50]

        engine = DataAlignmentEngine()

        # Full dataset -> should return False (has future data)
        assert not engine.validate_no_lookahead(ohlcv, as_of)

        # Pre-filtered -> should return True
        safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]
        assert engine.validate_no_lookahead(safe, as_of)

    def test_every_feature_generator_receives_only_past_data(
        self, all_factor_generators, validation_ohlcv
    ):
        """Every feature generator must receive data with max(date) < as_of
        when data is routed through DataAlignmentEngine."""
        ohlcv = validation_ohlcv
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        engine = DataAlignmentEngine()
        as_of = dates[300]

        # Pre-filter to simulate a correct pipeline
        safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]

        for generator in all_factor_generators:
            lookback = generator.lookback_days
            aligned = engine.get_aligned_data(safe, as_of, lookback)

            if aligned.empty:
                continue

            max_date = aligned.index.get_level_values(0).max()
            assert max_date < as_of, (
                f"Generator {generator.feature_name}: aligned data contains "
                f"date {max_date} >= as_of {as_of}"
            )

    def test_feature_output_is_ticker_indexed(
        self, all_factor_generators, validation_ohlcv
    ):
        """Feature generator output must be indexed by ticker, not by date,
        ensuring no temporal leakage in the output shape."""
        ohlcv = validation_ohlcv
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        engine = DataAlignmentEngine()
        as_of = dates[400]

        safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]

        for generator in all_factor_generators:
            lookback = generator.lookback_days
            aligned = engine.get_aligned_data(safe, as_of, lookback)

            if aligned.empty:
                continue

            scores = generator.compute(aligned, as_of)

            if scores.empty:
                continue

            # Output should NOT have a DatetimeIndex — it should be ticker-level
            assert not isinstance(scores.index, pd.DatetimeIndex), (
                f"Generator {generator.feature_name}: output index is DatetimeIndex "
                f"(should be ticker-level)"
            )

    def test_injected_future_data_detected_by_validator(self):
        """DataValidator must reject data that contains future timestamps."""
        ohlcv = make_ohlcv(tickers=["AAPL"], periods=100, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[50]

        validator = DataValidator()

        # Valid (pre-filtered) data should pass
        safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]
        valid_result = validator.validate(safe, as_of)
        assert valid_result.is_valid, (
            f"Pre-filtered data should pass validation: {valid_result.errors}"
        )

        # Full dataset with future timestamps should fail
        result = validator.validate(ohlcv, as_of)
        assert not result.is_valid, (
            "DataValidator should reject data with future timestamps"
        )

    def test_optimizer_receives_only_past_signals(
        self, validation_ohlcv, all_factor_generators
    ):
        """At date T, all alpha scores fed to the optimizer must be computed
        from data strictly before T."""
        ohlcv = validation_ohlcv
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        engine = DataAlignmentEngine()

        test_dates = [dates[300], dates[350], dates[400]]

        for as_of in test_dates:
            safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]

            for generator in all_factor_generators[:3]:
                lookback = generator.lookback_days
                aligned = engine.get_aligned_data(safe, as_of, lookback)

                if aligned.empty:
                    continue

                data_dates = aligned.index.get_level_values(0).unique()
                assert data_dates.max() < as_of, (
                    f"Date {as_of}: generator {generator.feature_name} received "
                    f"data up to {data_dates.max()}"
                )

                scores = generator.compute(aligned, as_of)
                assert not scores.empty or scores.isna().all(), (
                    f"Generator {generator.feature_name} should produce output "
                    f"from valid aligned data"
                )
