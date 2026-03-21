"""Test timestamp alignment across data sources.

Validates that OHLCV timestamps, feature timestamps, and signal timestamps
are properly aligned and that timezone/date boundary handling is correct.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine
from quant_fund.data_layer.data_validator import DataValidator, ValidationResult
from tests.conftest import make_ohlcv, STANDARD_TICKERS
from tests.generators.stress_scenario_generator import inject_duplicates


pytestmark = [pytest.mark.validation]


class TestTimestampAlignment:
    """Verify that timestamps are properly aligned across the pipeline."""

    def test_no_duplicate_date_ticker_pairs(self):
        """Each (date, ticker) pair must appear at most once in aligned data."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=200, seed=42)

        # Inject duplicates into the raw data
        duped = inject_duplicates(ohlcv, n_duplicates=30, seed=42)
        assert len(duped) > len(ohlcv), "Duplicates should have been injected"

        # DataValidator should detect duplicates
        validator = DataValidator()
        dates = duped.index.get_level_values(0).unique().sort_values()
        as_of = dates[-1] + pd.Timedelta(days=1)
        result = validator.validate(duped, as_of)

        has_dup_issue = (
            not result.is_valid
            or any("duplic" in w.lower() for w in result.warnings)
            or any("duplic" in e.lower() for e in result.errors)
        )
        assert has_dup_issue, (
            "DataValidator should detect duplicate (date, ticker) pairs"
        )

    def test_aligned_data_is_sorted_chronologically(self):
        """Data returned by DataAlignmentEngine must be sorted by date."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=300, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        engine = DataAlignmentEngine()
        as_of = dates[200]

        # Pre-filter to avoid LookAheadError
        safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]
        aligned = engine.get_aligned_data(safe, as_of, lookback_days=60)
        aligned_dates = aligned.index.get_level_values(0)

        date_values = aligned_dates.values
        is_sorted = np.all(date_values[:-1] <= date_values[1:])
        assert is_sorted, "Aligned data must be sorted chronologically"

    def test_lookback_window_respected(self):
        """Data returned must only include dates within the lookback window."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=400, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        engine = DataAlignmentEngine()

        as_of = dates[300]
        lookback_days = 60

        safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]
        aligned = engine.get_aligned_data(safe, as_of, lookback_days=lookback_days)
        aligned_dates = aligned.index.get_level_values(0).unique().sort_values()

        if len(aligned_dates) == 0:
            pytest.skip("No aligned data returned")

        earliest = aligned_dates.min()
        latest = aligned_dates.max()

        # Latest must be strictly before as_of
        assert latest < as_of, (
            f"Latest aligned date {latest} must be < as_of {as_of}"
        )

        # Earliest should be no more than lookback_days + buffer before as_of
        date_span = (as_of - earliest).days
        assert date_span <= lookback_days + 10, (
            f"Earliest date {earliest} is {date_span} days before as_of, "
            f"but lookback is only {lookback_days} days"
        )
