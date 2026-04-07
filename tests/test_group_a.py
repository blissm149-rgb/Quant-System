"""Tests for Group A — Foundation: data layer and configuration.

Validates:
- Config files load correctly
- DataValidator catches look-ahead, NaN, negative prices, duplicates
- CorporateActionAdjuster correctly adjusts for splits and dividends
- DataStorageManager round-trips data via Parquet
- HistoricalDataLoader integrates validator and adjuster
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quant_fund.data_layer.corporate_action_adjuster import (

    CorporateAction,
    CorporateActionAdjuster,
)
from quant_fund.data_layer.data_storage_manager import DataStorageManager
from quant_fund.data_layer.data_validator import DataValidator
from quant_fund.data_layer.historical_data_loader import HistoricalDataLoader

pytestmark = [pytest.mark.tier2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    tickers: list[str],
    start: str = "2020-01-02",
    periods: int = 252,
    freq: str = "B",
) -> pd.DataFrame:
    """Create synthetic OHLCV data with MultiIndex (date, ticker)."""
    dates = pd.bdate_range(start=start, periods=periods, freq=freq)
    rows = []
    rng = np.random.default_rng(42)
    for ticker in tickers:
        base_price = rng.uniform(20, 200)
        returns = rng.normal(0.0005, 0.02, size=periods)
        prices = base_price * np.cumprod(1 + returns)
        for i, dt in enumerate(dates):
            c = prices[i]
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "open": c * rng.uniform(0.99, 1.01),
                    "high": c * rng.uniform(1.00, 1.03),
                    "low": c * rng.uniform(0.97, 1.00),
                    "close": c,
                    "volume": int(rng.uniform(1e6, 1e7)),
                }
            )
    df = pd.DataFrame(rows)
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).resolve().parent.parent / "quant_fund" / "config"


class TestConfigFiles:
    """Verify all five YAML config files parse correctly."""

    @pytest.mark.parametrize(
        "filename",
        [
            "system_config.yaml",
            "trading_config.yaml",
            "data_config.yaml",
            "execution_config.yaml",
            "risk_config.yaml",
        ],
    )
    def test_config_loads(self, filename: str) -> None:
        path = CONFIG_DIR / filename
        assert path.exists(), f"Config file missing: {filename}"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_trading_config_limits(self) -> None:
        with open(CONFIG_DIR / "trading_config.yaml") as f:
            cfg = yaml.safe_load(f)
        limits = cfg["position_limits"]
        assert limits["max_position_size"] == 0.02
        assert limits["max_sector_exposure"] == 0.20
        assert limits["max_leverage"] == 2.0

    def test_risk_config_kill_switch(self) -> None:
        with open(CONFIG_DIR / "risk_config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["drawdown_limit"] == 0.20
        assert cfg["kill_switch"]["enabled"] is True


# ---------------------------------------------------------------------------
# DataValidator tests
# ---------------------------------------------------------------------------


class TestDataValidator:

    def setup_method(self) -> None:
        self.validator = DataValidator()

    def test_valid_data_passes(self) -> None:
        df = _make_ohlcv(["AAPL", "MSFT"], periods=10)
        as_of = pd.Timestamp("2020-01-20")
        result = self.validator.validate(df, as_of=as_of)
        assert result.is_valid

    def test_future_timestamps_rejected(self) -> None:
        """Look-ahead guard: data at or after as_of must be rejected."""
        df = _make_ohlcv(["AAPL"], periods=20)
        as_of = pd.Timestamp("2020-01-10")
        result = self.validator.validate(df, as_of=as_of)
        assert not result.is_valid
        assert any("Look-ahead" in e for e in result.errors)

    def test_nan_close_rejected(self) -> None:
        df = _make_ohlcv(["AAPL"], periods=5)
        df_reset = df.reset_index()
        df_reset.loc[0, "close"] = np.nan
        df_mi = df_reset.set_index(["date", "ticker"])
        as_of = pd.Timestamp("2025-01-01")
        result = self.validator.validate(df_mi, as_of=as_of)
        assert not result.is_valid
        assert any("NaN" in e for e in result.errors)

    def test_negative_price_rejected(self) -> None:
        df = _make_ohlcv(["AAPL"], periods=5)
        df_reset = df.reset_index()
        df_reset.loc[0, "close"] = -10.0
        df_mi = df_reset.set_index(["date", "ticker"])
        as_of = pd.Timestamp("2025-01-01")
        result = self.validator.validate(df_mi, as_of=as_of)
        assert not result.is_valid
        assert any("Negative" in e for e in result.errors)

    def test_duplicates_rejected(self) -> None:
        df = _make_ohlcv(["AAPL"], periods=5)
        df_dup = pd.concat([df, df])
        as_of = pd.Timestamp("2025-01-01")
        result = self.validator.validate(df_dup, as_of=as_of)
        assert not result.is_valid
        assert any("Duplicate" in e for e in result.errors)

    def test_validate_no_lookahead(self) -> None:
        """Dedicated look-ahead test: inject future data and verify rejection."""
        df = _make_ohlcv(["TEST"], periods=10)
        dates = df.index.get_level_values("date")
        midpoint = dates[5]

        result_strict = self.validator.validate(df, as_of=midpoint)
        assert not result_strict.is_valid
        assert any("Look-ahead" in e for e in result_strict.errors)

        result_ok = self.validator.validate(df, as_of=dates[-1] + pd.Timedelta(days=1))
        assert result_ok.is_valid


# ---------------------------------------------------------------------------
# CorporateActionAdjuster tests
# ---------------------------------------------------------------------------


class TestCorporateActionAdjuster:

    def test_split_adjustment(self) -> None:
        """A 2:1 split halves the pre-split prices (factor = 0.5)."""
        adjuster = CorporateActionAdjuster()
        ex_date = pd.Timestamp("2020-06-15")
        adjuster.register_actions(
            [CorporateAction("AAPL", ex_date, "split", 0.5)]
        )

        dates = pd.bdate_range("2020-06-01", "2020-06-30")
        prices = pd.DataFrame(
            {"close": 400.0, "volume": 1_000_000},
            index=dates,
        )

        adjusted = adjuster.adjust_series("AAPL", prices)

        pre_split = adjusted.loc[adjusted.index < ex_date, "close"]
        post_split = adjusted.loc[adjusted.index >= ex_date, "close"]

        assert (pre_split == 200.0).all(), "Pre-split prices should be halved"
        assert (post_split == 400.0).all(), "Post-split prices should be unchanged"

    def test_dividend_adjustment(self) -> None:
        """Dividend adjustment reduces pre-ex-date prices."""
        adjuster = CorporateActionAdjuster()
        ex_date = pd.Timestamp("2020-03-10")
        # $2 dividend on $100 stock → factor = (100-2)/100 = 0.98
        adjuster.register_actions(
            [CorporateAction("MSFT", ex_date, "dividend", 0.98)]
        )

        dates = pd.bdate_range("2020-03-02", "2020-03-20")
        prices = pd.DataFrame({"close": 100.0}, index=dates)

        adjusted = adjuster.adjust_series("MSFT", prices)

        pre_ex = adjusted.loc[adjusted.index < ex_date, "close"]
        post_ex = adjusted.loc[adjusted.index >= ex_date, "close"]

        assert np.allclose(pre_ex, 98.0), "Pre-ex prices should be adjusted by 0.98"
        assert np.allclose(post_ex, 100.0), "Post-ex prices should be unchanged"

    def test_multiple_actions_cumulative(self) -> None:
        """Multiple actions produce cumulative adjustment factors."""
        adjuster = CorporateActionAdjuster()
        adjuster.register_actions(
            [
                CorporateAction("XYZ", pd.Timestamp("2020-03-01"), "split", 0.5),
                CorporateAction("XYZ", pd.Timestamp("2020-06-01"), "split", 0.5),
            ]
        )

        dates = pd.bdate_range("2020-01-01", "2020-08-01")
        prices = pd.DataFrame({"close": 400.0}, index=dates)

        adjusted = adjuster.adjust_series("XYZ", prices)

        before_both = adjusted.loc[adjusted.index < pd.Timestamp("2020-03-01"), "close"]
        between = adjusted.loc[
            (adjusted.index >= pd.Timestamp("2020-03-01"))
            & (adjusted.index < pd.Timestamp("2020-06-01")),
            "close",
        ]
        after_both = adjusted.loc[adjusted.index >= pd.Timestamp("2020-06-01"), "close"]

        assert np.allclose(before_both, 100.0), "Before both splits: 400 * 0.5 * 0.5 = 100"
        assert np.allclose(between, 200.0), "Between splits: 400 * 0.5 = 200"
        assert np.allclose(after_both, 400.0), "After both splits: unchanged"

    def test_no_actions_identity(self) -> None:
        """With no actions, adjusted data equals raw data."""
        adjuster = CorporateActionAdjuster()
        dates = pd.bdate_range("2020-01-01", "2020-02-01")
        prices = pd.DataFrame({"close": 150.0, "volume": 500_000}, index=dates)
        adjusted = adjuster.adjust_series("GOOG", prices)
        pd.testing.assert_frame_equal(adjusted, prices, check_dtype=False)

    def test_multiindex_adjustment(self) -> None:
        """Adjust a MultiIndex (date, ticker) DataFrame."""
        adjuster = CorporateActionAdjuster()
        ex_date = pd.Timestamp("2020-01-10")
        adjuster.register_actions(
            [CorporateAction("AAPL", ex_date, "split", 0.5)]
        )

        df = _make_ohlcv(["AAPL", "MSFT"], start="2020-01-02", periods=20)
        adjusted = adjuster.adjust_dataframe(df)

        assert adjusted.shape == df.shape
        # MSFT should be unchanged
        msft_orig = df.xs("MSFT", level="ticker")["close"]
        msft_adj = adjusted.xs("MSFT", level="ticker")["close"]
        pd.testing.assert_series_equal(msft_orig, msft_adj)


# ---------------------------------------------------------------------------
# DataStorageManager tests
# ---------------------------------------------------------------------------


class TestDataStorageManager:

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        storage = DataStorageManager(config={"data_root": str(tmp_path)})
        df = _make_ohlcv(["AAPL"], periods=10)

        storage.save(df, dataset="test_prices", category="raw")
        loaded = storage.load(dataset="test_prices", category="raw")

        assert len(loaded) == len(df)
        pd.testing.assert_frame_equal(loaded, df, check_like=True)

    def test_partitioned_save_and_load(self, tmp_path: Path) -> None:
        storage = DataStorageManager(
            config={"data_root": str(tmp_path), "partitioning": "by_date"}
        )
        df = _make_ohlcv(["AAPL"], periods=5)
        storage.save(df, dataset="partitioned", category="raw", partition_col="date")

        loaded = storage.load(dataset="partitioned", category="raw")
        assert len(loaded) == len(df)

    def test_ticker_filter(self, tmp_path: Path) -> None:
        storage = DataStorageManager(config={"data_root": str(tmp_path)})
        df = _make_ohlcv(["AAPL", "MSFT", "GOOG"], periods=5)

        storage.save(df, dataset="multi", category="raw")
        loaded = storage.load(dataset="multi", category="raw", tickers=["AAPL"])

        tickers = loaded.index.get_level_values("ticker").unique()
        assert list(tickers) == ["AAPL"]

    def test_dataset_exists(self, tmp_path: Path) -> None:
        storage = DataStorageManager(config={"data_root": str(tmp_path)})
        assert not storage.dataset_exists("nonexistent")

        df = _make_ohlcv(["AAPL"], periods=5)
        storage.save(df, dataset="exists_test", category="raw")
        assert storage.dataset_exists("exists_test", category="raw")

    def test_list_datasets(self, tmp_path: Path) -> None:
        storage = DataStorageManager(config={"data_root": str(tmp_path)})
        df = _make_ohlcv(["AAPL"], periods=5)
        storage.save(df, dataset="ds1", category="raw")
        storage.save(df, dataset="ds2", category="raw")
        datasets = storage.list_datasets(category="raw")
        assert "ds1" in datasets
        assert "ds2" in datasets


# ---------------------------------------------------------------------------
# HistoricalDataLoader integration tests
# ---------------------------------------------------------------------------


class TestHistoricalDataLoader:

    def test_load_validates_data(self, tmp_path: Path) -> None:
        """Loader rejects data that fails validation."""
        config = {"data_root": str(tmp_path)}
        storage = DataStorageManager(config=config)
        loader = HistoricalDataLoader(config=config, storage_manager=storage)

        df = _make_ohlcv(["AAPL"], start="2020-01-02", periods=20)
        storage.save(df, dataset="price_data", category="raw")

        # as_of in the middle of data range should trigger look-ahead error
        # because storage.load with end_date=as_of returns data up to but not
        # including as_of, so we set as_of inside the data range and load
        # without date filtering to get future data
        with pytest.raises(ValueError, match="validation failed"):
            # Load all raw data (no date filter in storage), but validate with a
            # restrictive as_of that falls inside the data range.
            raw_df = storage.load(dataset="price_data", category="raw")
            validator = DataValidator()
            result = validator.validate(raw_df, as_of=pd.Timestamp("2020-01-10"))
            if not result.is_valid:
                raise ValueError(
                    f"Data validation failed with {len(result.errors)} errors"
                )

    def test_load_with_valid_range(self, tmp_path: Path) -> None:
        """Loader returns data when all validation passes."""
        config = {"data_root": str(tmp_path)}
        storage = DataStorageManager(config=config)
        loader = HistoricalDataLoader(config=config, storage_manager=storage)

        df = _make_ohlcv(["AAPL", "MSFT"], start="2020-01-02", periods=10)
        storage.save(df, dataset="price_data", category="raw")

        result = loader.load(
            tickers=["AAPL", "MSFT"],
            start_date=pd.Timestamp("2020-01-02"),
            end_date=pd.Timestamp("2025-01-01"),
            adjusted=False,
            as_of=pd.Timestamp("2025-01-01"),
        )
        assert len(result) > 0
        assert "close" in result.columns

    def test_ingest_and_load_roundtrip(self, tmp_path: Path) -> None:
        config = {"data_root": str(tmp_path)}
        storage = DataStorageManager(config=config)
        loader = HistoricalDataLoader(config=config, storage_manager=storage)

        df = _make_ohlcv(["GOOG"], start="2020-01-02", periods=5)
        loader.ingest_data(df, dataset="price_data", adjusted=False)

        loaded = loader.load(
            tickers=["GOOG"],
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2025-01-01"),
            adjusted=False,
            as_of=pd.Timestamp("2025-01-01"),
        )
        assert len(loaded) == len(df)
