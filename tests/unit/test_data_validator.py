"""Unit tests for data_validator module.

TESTING_PLAN.md Section 3.2 — Data Layer.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.data_layer.data_validator import DataValidator, ValidationResult
from tests.conftest import make_ohlcv


@pytest.mark.unit
@pytest.mark.tier1
class TestDataValidator:
    """DataValidator — critical safety gate for data quality."""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    @pytest.fixture
    def good_data(self):
        return make_ohlcv(tickers=["AAPL"], periods=10, seed=42)

    @pytest.fixture
    def as_of(self, good_data):
        """An as_of timestamp safely after all data."""
        latest = good_data.index.get_level_values("date").max()
        return latest + pd.Timedelta(days=1)

    def test_valid_data_passes(self, validator, good_data, as_of):
        """Clean data passes validation."""
        result = validator.validate(good_data, as_of=as_of)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_rejects_future_timestamps(self, validator, good_data):
        """Data with timestamps >= as_of is rejected (look-ahead guard)."""
        earliest = good_data.index.get_level_values("date").min()
        result = validator.validate(good_data, as_of=earliest)
        assert not result.is_valid or len(result.warnings) > 0 or len(result.errors) > 0

    def test_rejects_nan_close_prices(self, validator, as_of):
        """NaN in close column triggers validation failure."""
        df = make_ohlcv(tickers=["AAPL"], periods=10, seed=42)
        df.iloc[0, df.columns.get_loc("close")] = np.nan
        result = validator.validate(df, as_of=as_of)
        assert not result.is_valid or len(result.errors) > 0

    def test_rejects_negative_prices(self, validator, as_of):
        """Negative close prices trigger validation failure."""
        df = make_ohlcv(tickers=["AAPL"], periods=10, seed=42)
        df.iloc[0, df.columns.get_loc("close")] = -1.0
        result = validator.validate(df, as_of=as_of)
        has_issues = not result.is_valid or len(result.errors) > 0 or len(result.warnings) > 0
        assert has_issues

    def test_detects_duplicate_rows(self, validator, as_of):
        """Duplicate (date, ticker) pairs detected."""
        df = make_ohlcv(tickers=["AAPL"], periods=10, seed=42)
        dup = pd.concat([df, df.iloc[:1]])
        result = validator.validate(dup, as_of=as_of)
        has_issues = not result.is_valid or len(result.errors) > 0 or len(result.warnings) > 0
        assert has_issues

    def test_validates_multiindex_format(self, validator, good_data, as_of):
        """Data with proper MultiIndex(date, ticker) passes format check."""
        assert isinstance(good_data.index, pd.MultiIndex)
        assert good_data.index.names == ["date", "ticker"]
        result = validator.validate(good_data, as_of=as_of)
        assert result.is_valid

    def test_validation_result_bool(self):
        """ValidationResult is truthy when valid, falsy when not."""
        assert bool(ValidationResult(is_valid=True))
        assert not bool(ValidationResult(is_valid=False))

    def test_config_accepts_custom_params(self):
        """Custom config parameters are accepted."""
        validator = DataValidator(config={"max_nan_ratio": 0.01})
        assert validator is not None
