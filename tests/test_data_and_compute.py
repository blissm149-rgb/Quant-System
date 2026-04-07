"""Tests for data ingestion, model persistence, scheduling, and compute gaps.

Covers:
- BaseConnector and CSVConnector (data connectors)
- ModelStore (model persistence and versioning)
- IngestionScheduler (schedule parsing and tick execution)
- DataStorageManager (purge_old_partitions, get_storage_stats)
- Worker scaling (environment-aware defaults)
"""

import os
import tempfile
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from quant_fund.data_layer.connectors.base_connector import BaseConnector
from quant_fund.data_layer.connectors.csv_connector import CSVConnector
from quant_fund.data_layer.data_storage_manager import DataStorageManager
from quant_fund.infrastructure.ingestion_scheduler import (

pytestmark = [pytest.mark.tier2]
    IngestionScheduler,
    ScheduleEntry,
)
from quant_fund.infrastructure.model_store import ModelStore


# ── Helpers ────────────────────────────────────────────────────────


class DummyConnector(BaseConnector):
    """Minimal concrete connector for testing base class logic."""

    def fetch_historical(self, tickers, start_date, end_date, fields=None):
        rows = []
        for t in tickers:
            rows.append({
                "date": start_date,
                "ticker": t,
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 1_000_000,
            })
        df = pd.DataFrame(rows)
        return self._to_multiindex(df)

    def fetch_latest(self, tickers):
        return self.fetch_historical(
            tickers, pd.Timestamp.now().normalize(), pd.Timestamp.now().normalize()
        )

    def get_source_name(self):
        return "dummy"


def _make_sklearn_model():
    """Create a simple trained sklearn model for persistence tests."""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1.0, 2.0, 3.0])
    model.fit(X, y)
    return model


def _make_date_partitioned_data(storage: DataStorageManager, dataset: str, dates: List[str]):
    """Create date-partitioned Parquet files for testing."""
    for date_str in dates:
        partition_dir = storage._data_root / "raw" / dataset / f"date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [103.0], "volume": [1_000_000],
        })
        df.to_parquet(partition_dir / "data.parquet", engine="pyarrow")


# ═══════════════════════════════════════════════════════════════════
# BaseConnector tests
# ═══════════════════════════════════════════════════════════════════


class TestBaseConnector:
    """Tests for the abstract BaseConnector and its helpers."""

    def test_to_multiindex_from_columns(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "ticker": ["AAPL"],
            "close": [150.0],
        })
        result = BaseConnector._to_multiindex(df)
        assert isinstance(result.index, pd.MultiIndex)
        assert list(result.index.names) == ["date", "ticker"]

    def test_to_multiindex_already_set(self):
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-01"), "AAPL")], names=["date", "ticker"]
        )
        df = pd.DataFrame({"close": [150.0]}, index=idx)
        result = BaseConnector._to_multiindex(df)
        assert result is df  # unchanged

    def test_empty_ohlcv(self):
        df = BaseConnector._empty_ohlcv()
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_track_ingestion(self):
        conn = DummyConnector()
        meta = conn._track_ingestion(
            ["AAPL", "MSFT"], 100,
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01"),
        )
        assert meta["source"] == "dummy"
        assert meta["tickers_requested"] == 2
        assert meta["rows_fetched"] == 100

    def test_dummy_fetch_historical(self):
        conn = DummyConnector()
        df = conn.fetch_historical(
            ["AAPL", "MSFT"],
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-06-01"),
        )
        assert not df.empty
        assert "close" in df.columns
        assert isinstance(df.index, pd.MultiIndex)

    def test_retry_with_backoff_success(self):
        conn = DummyConnector(config={"rate_limit_delay_sec": 0.01, "max_retries": 3})
        call_count = 0

        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        result = conn._retry_with_backoff(flaky_fn)
        assert result == "ok"
        assert call_count == 3

    def test_retry_with_backoff_exhausted(self):
        conn = DummyConnector(config={"rate_limit_delay_sec": 0.01, "max_retries": 2})

        def always_fail():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            conn._retry_with_backoff(always_fail)


# ═══════════════════════════════════════════════════════════════════
# CSVConnector tests
# ═══════════════════════════════════════════════════════════════════


class TestCSVConnector:
    """Tests for CSVConnector."""

    @pytest.fixture
    def csv_dir(self, tmp_path):
        """Create a temp dir with per-ticker CSV files."""
        (tmp_path / "AAPL.csv").write_text(
            "date,open,high,low,close,volume\n"
            "2024-01-02,150,155,149,153,1000000\n"
            "2024-01-03,153,157,152,156,1100000\n"
            "2024-06-01,180,185,179,183,900000\n"
        )
        (tmp_path / "MSFT.csv").write_text(
            "date,open,high,low,close,volume\n"
            "2024-01-02,370,375,369,373,800000\n"
        )
        return tmp_path

    @pytest.fixture
    def combined_dir(self, tmp_path):
        """Create a temp dir with a combined all_data.csv."""
        (tmp_path / "all_data.csv").write_text(
            "date,ticker,open,high,low,close,volume\n"
            "2024-01-02,AAPL,150,155,149,153,1000000\n"
            "2024-01-02,MSFT,370,375,369,373,800000\n"
        )
        return tmp_path

    def test_per_ticker_files(self, csv_dir):
        conn = CSVConnector(config={"data_dir": str(csv_dir)})
        df = conn.fetch_historical(
            ["AAPL", "MSFT"],
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
        )
        assert not df.empty
        tickers = df.index.get_level_values("ticker").unique()
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_date_filtering(self, csv_dir):
        conn = CSVConnector(config={"data_dir": str(csv_dir)})
        df = conn.fetch_historical(
            ["AAPL"],
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
        )
        assert len(df) == 2  # Jan 2 and Jan 3 only, not Jun 1

    def test_combined_file(self, combined_dir):
        conn = CSVConnector(config={"data_dir": str(combined_dir)})
        df = conn.fetch_historical(
            ["AAPL"],
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2025-01-01"),
        )
        assert not df.empty
        assert len(df) == 1

    def test_missing_ticker(self, csv_dir):
        conn = CSVConnector(config={"data_dir": str(csv_dir)})
        df = conn.fetch_historical(
            ["NONEXISTENT"],
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2025-01-01"),
        )
        assert df.empty

    def test_fetch_latest(self, csv_dir):
        conn = CSVConnector(config={"data_dir": str(csv_dir)})
        df = conn.fetch_latest(["AAPL"])
        assert len(df) == 1
        # Should be the June entry (latest)
        assert df["close"].iloc[0] == 183.0

    def test_empty_dir(self, tmp_path):
        conn = CSVConnector(config={"data_dir": str(tmp_path)})
        df = conn.fetch_historical(
            ["AAPL"],
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2025-01-01"),
        )
        assert df.empty

    def test_source_name(self):
        conn = CSVConnector()
        assert conn.get_source_name() == "csv"


# ═══════════════════════════════════════════════════════════════════
# ModelStore tests
# ═══════════════════════════════════════════════════════════════════


class TestModelStore:
    """Tests for ModelStore."""

    @pytest.fixture
    def store(self, tmp_path):
        return ModelStore(config={"model_dir": str(tmp_path), "max_versions_per_model": 5})

    def test_save_and_load_sklearn(self, store):
        model = _make_sklearn_model()
        vid = store.save_sklearn_model(
            model,
            model_name="test_model",
            train_start_date="2023-01-01",
            train_end_date="2024-01-01",
            feature_names=["f1", "f2"],
            metrics={"ic": 0.05, "sharpe": 1.2},
        )
        loaded = store.load_sklearn_model("test_model", vid)
        # Verify predictions match
        X = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_almost_equal(model.predict(X), loaded.predict(X))

    def test_save_and_load_numpy_weights(self, store):
        weights = {
            "W1": np.random.randn(10, 5),
            "b1": np.random.randn(5),
            "W2": np.random.randn(5, 1),
        }
        vid = store.save_numpy_weights(
            weights,
            model_name="lstm_model",
            train_start_date="2023-01-01",
            train_end_date="2024-01-01",
            metrics={"mse": 0.01},
        )
        loaded = store.load_numpy_weights("lstm_model", vid)
        for key in weights:
            np.testing.assert_array_equal(weights[key], loaded[key])

    def test_list_versions(self, store):
        model = _make_sklearn_model()
        v1 = store.save_sklearn_model(
            model, "m", "2023-01-01", "2023-06-01", ["f1"],
        )
        v2 = store.save_sklearn_model(
            model, "m", "2023-06-01", "2024-01-01", ["f1"],
        )
        versions = store.list_versions("m")
        assert len(versions) == 2
        # Newest first
        assert versions[0]["version_id"] == v2

    def test_get_version(self, store):
        model = _make_sklearn_model()
        vid = store.save_sklearn_model(
            model, "m", "2023-01-01", "2024-01-01", ["f1"],
            metrics={"ic": 0.03},
        )
        version = store.get_version("m", vid)
        assert version["metrics"]["ic"] == 0.03
        assert version["feature_names"] == ["f1"]

    def test_get_version_not_found(self, store):
        with pytest.raises(FileNotFoundError):
            store.get_version("nonexistent", "abc")

    def test_promote_to_champion(self, store):
        model = _make_sklearn_model()
        vid = store.save_sklearn_model(
            model, "m", "2023-01-01", "2024-01-01", ["f1"],
        )
        store.promote_to_champion("m", vid)
        champion = store.get_champion("m")
        assert champion["version_id"] == vid

    def test_load_champion_by_default(self, store):
        model = _make_sklearn_model()
        v1 = store.save_sklearn_model(
            model, "m", "2023-01-01", "2023-06-01", ["f1"],
        )
        v2 = store.save_sklearn_model(
            model, "m", "2023-06-01", "2024-01-01", ["f1"],
        )
        store.promote_to_champion("m", v1)
        # Loading without version_id should return champion (v1)
        loaded = store.load_sklearn_model("m")
        X = np.array([[1, 2]])
        np.testing.assert_array_almost_equal(model.predict(X), loaded.predict(X))

    def test_compare_versions(self, store):
        model = _make_sklearn_model()
        v1 = store.save_sklearn_model(
            model, "m", "2023-01-01", "2023-06-01", ["f1"],
            metrics={"ic": 0.03, "sharpe": 1.0},
        )
        v2 = store.save_sklearn_model(
            model, "m", "2023-06-01", "2024-01-01", ["f1"],
            metrics={"ic": 0.05, "sharpe": 1.5},
        )
        comp = store.compare_versions("m", v1, v2)
        assert comp["metrics"]["ic"]["diff"] == pytest.approx(0.02)
        assert comp["winner"] == v2

    def test_challenger_beats_champion(self, store):
        model = _make_sklearn_model()
        v1 = store.save_sklearn_model(
            model, "m", "2023-01-01", "2023-06-01", ["f1"],
            metrics={"ic": 0.03},
        )
        store.promote_to_champion("m", v1)
        v2 = store.save_sklearn_model(
            model, "m", "2023-06-01", "2024-01-01", ["f1"],
            metrics={"ic": 0.05},
        )
        beats, details = store.check_challenger_beats_champion("m", v2, "ic")
        assert beats
        assert details["improvement"] == pytest.approx(0.02)

    def test_challenger_does_not_beat(self, store):
        model = _make_sklearn_model()
        v1 = store.save_sklearn_model(
            model, "m", "2023-01-01", "2023-06-01", ["f1"],
            metrics={"ic": 0.05},
        )
        store.promote_to_champion("m", v1)
        v2 = store.save_sklearn_model(
            model, "m", "2023-06-01", "2024-01-01", ["f1"],
            metrics={"ic": 0.03},
        )
        beats, details = store.check_challenger_beats_champion(
            "m", v2, "ic", min_improvement=0.01
        )
        assert not beats

    def test_no_champion_challenger_always_wins(self, store):
        model = _make_sklearn_model()
        vid = store.save_sklearn_model(
            model, "m", "2023-01-01", "2024-01-01", ["f1"],
            metrics={"ic": 0.01},
        )
        beats, details = store.check_challenger_beats_champion("m", vid, "ic")
        assert beats
        assert details["reason"] == "no_existing_champion"

    def test_retention_enforcement(self, store):
        model = _make_sklearn_model()
        # max_versions is 5, create 7
        for i in range(7):
            store.save_sklearn_model(
                model, "m", f"2023-0{i+1}-01", f"2023-0{i+2}-01", ["f1"],
            )
        versions = store.list_versions("m")
        assert len(versions) == 5

    def test_load_no_versions_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_sklearn_model("nonexistent")


# ═══════════════════════════════════════════════════════════════════
# IngestionScheduler tests
# ═══════════════════════════════════════════════════════════════════


class TestIngestionScheduler:
    """Tests for IngestionScheduler."""

    def test_parse_time_only(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"daily_close": "16:30"}
        })
        schedules = scheduler.get_schedules()
        assert len(schedules) == 1
        assert schedules[0]["start_time"] == "16:30:00"
        assert schedules[0]["interval_minutes"] is None

    def test_parse_range_interval(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"price_update": "09:30-16:00/5min"}
        })
        schedules = scheduler.get_schedules()
        assert len(schedules) == 1
        assert schedules[0]["interval_minutes"] == 5
        assert schedules[0]["start_time"] == "09:30:00"
        assert schedules[0]["end_time"] == "16:00:00"

    def test_parse_hourly_interval(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"hourly": "08:00-18:00/1h"}
        })
        schedules = scheduler.get_schedules()
        assert schedules[0]["interval_minutes"] == 60

    def test_parse_invalid_schedule(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"bad": "not-a-schedule"}
        })
        assert len(scheduler.get_schedules()) == 0

    def test_register_callback(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"task": "12:00"}
        })
        assert scheduler.register_callback("task", lambda: None)
        assert not scheduler.register_callback("nonexistent", lambda: None)

    def test_tick_fires_due_schedule(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"noon": "12:00"}
        })
        fired = []
        scheduler.register_callback("noon", lambda: fired.append(True))

        # Tick at exactly 12:00
        now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        results = scheduler.tick(now)
        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert len(fired) == 1

    def test_tick_does_not_fire_wrong_time(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"noon": "12:00"}
        })
        fired = []
        scheduler.register_callback("noon", lambda: fired.append(True))

        now = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
        results = scheduler.tick(now)
        assert len(results) == 0
        assert len(fired) == 0

    def test_tick_once_daily_no_repeat(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"noon": "12:00"}
        })
        scheduler.register_callback("noon", lambda: None)

        now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        scheduler.tick(now)
        # Second tick same day should not fire
        results = scheduler.tick(now)
        assert len(results) == 0

    def test_tick_interval_based(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"updates": "09:00-17:00/5min"}
        })
        fired = []
        scheduler.register_callback("updates", lambda: fired.append(True))

        # First tick at 09:00 — should fire
        t1 = datetime(2024, 6, 15, 9, 0, tzinfo=timezone.utc)
        scheduler.tick(t1)
        assert len(fired) == 1

        # 2 minutes later — too soon
        t2 = t1 + timedelta(minutes=2)
        scheduler.tick(t2)
        assert len(fired) == 1

        # 5 minutes later — should fire
        t3 = t1 + timedelta(minutes=5)
        scheduler.tick(t3)
        assert len(fired) == 2

    def test_tick_outside_window(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"updates": "09:00-17:00/5min"}
        })
        scheduler.register_callback("updates", lambda: None)

        # Before window
        now = datetime(2024, 6, 15, 8, 0, tzinfo=timezone.utc)
        assert len(scheduler.tick(now)) == 0

        # After window
        now = datetime(2024, 6, 15, 18, 0, tzinfo=timezone.utc)
        assert len(scheduler.tick(now)) == 0

    def test_callback_error_handled(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"failing": "12:00"}
        })
        scheduler.register_callback("failing", lambda: 1/0)

        now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        results = scheduler.tick(now)
        assert results[0]["status"] == "error"

    def test_no_callback_status(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"unbound": "12:00"}
        })
        now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        results = scheduler.tick(now)
        assert results[0]["status"] == "no_callback"

    def test_disable_enable_schedule(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"task": "12:00"}
        })
        scheduler.register_callback("task", lambda: None)
        scheduler.disable_schedule("task")

        now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        assert len(scheduler.tick(now)) == 0

        scheduler.enable_schedule("task")
        assert len(scheduler.tick(now)) == 1

    def test_add_schedule_programmatically(self):
        scheduler = IngestionScheduler()
        assert scheduler.add_schedule("custom", "14:00", callback=lambda: None)
        assert len(scheduler.get_schedules()) == 1

    def test_run_history(self):
        scheduler = IngestionScheduler(config={
            "schedules": {"task": "12:00"}
        })
        scheduler.register_callback("task", lambda: None)

        now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        scheduler.tick(now)
        history = scheduler.get_run_history()
        assert len(history) == 1
        assert history[0]["schedule"] == "task"


# ═══════════════════════════════════════════════════════════════════
# DataStorageManager purge/stats tests
# ═══════════════════════════════════════════════════════════════════


class TestDataStoragePurge:
    """Tests for data retention and storage stats."""

    @pytest.fixture
    def storage(self, tmp_path):
        return DataStorageManager(config={
            "data_root": str(tmp_path),
            "lookback_years": 2,
        })

    def test_purge_old_partitions(self, storage):
        dates = [
            "2020-01-01", "2020-06-01",  # old — should be purged
            "2024-01-01", "2024-06-01",  # recent — should be kept
        ]
        _make_date_partitioned_data(storage, "prices", dates)

        cutoff = pd.Timestamp("2023-01-01")
        removed, kept = storage.purge_old_partitions("prices", cutoff_date=cutoff)
        assert removed == 2
        assert kept == 2

    def test_purge_nothing_to_remove(self, storage):
        dates = ["2024-01-01", "2024-06-01"]
        _make_date_partitioned_data(storage, "prices", dates)

        cutoff = pd.Timestamp("2020-01-01")
        removed, kept = storage.purge_old_partitions("prices", cutoff_date=cutoff)
        assert removed == 0
        assert kept == 2

    def test_purge_default_cutoff(self, storage):
        """Uses lookback_years=2 to compute cutoff automatically."""
        old_date = "2010-01-01"
        recent_date = "2026-01-01"
        _make_date_partitioned_data(storage, "prices", [old_date, recent_date])

        removed, kept = storage.purge_old_partitions("prices")
        assert removed == 1  # 2010 should be purged
        assert kept == 1

    def test_purge_nonexistent_dataset(self, storage):
        removed, kept = storage.purge_old_partitions("nonexistent")
        assert removed == 0
        assert kept == 0

    def test_get_storage_stats(self, storage):
        dates = ["2024-01-01", "2024-06-01", "2024-12-01"]
        _make_date_partitioned_data(storage, "prices", dates)

        stats = storage.get_storage_stats("prices")
        assert stats["total_files"] == 3
        assert stats["total_bytes"] > 0
        assert stats["oldest_partition"] == "2024-01-01"
        assert stats["newest_partition"] == "2024-12-01"
        assert stats["partition_count"] == 3

    def test_get_storage_stats_empty(self, storage):
        stats = storage.get_storage_stats("nonexistent")
        assert stats["total_files"] == 0
        assert stats["total_bytes"] == 0


# ═══════════════════════════════════════════════════════════════════
# Worker scaling tests
# ═══════════════════════════════════════════════════════════════════


class TestWorkerScaling:
    """Tests that worker count adapts to environment."""

    def test_backtest_runner_default_uses_cpu_count(self):
        from quant_fund.research_cluster.distributed_backtest_runner import (
            DistributedBacktestRunner,
        )
        runner = DistributedBacktestRunner()
        cpu_count = os.cpu_count() or 2
        expected = max(1, cpu_count - 1)
        assert runner._max_workers == expected

    def test_backtest_runner_config_override(self):
        from quant_fund.research_cluster.distributed_backtest_runner import (
            DistributedBacktestRunner,
        )
        runner = DistributedBacktestRunner(config={"max_workers": 2})
        assert runner._max_workers == 2

    def test_signal_evaluator_default_uses_cpu_count(self):
        from quant_fund.research_cluster.parallel_signal_evaluator import (
            ParallelSignalEvaluator,
        )
        evaluator = ParallelSignalEvaluator()
        cpu_count = os.cpu_count() or 2
        expected = max(1, cpu_count - 1)
        assert evaluator._max_workers == expected

    def test_signal_evaluator_config_override(self):
        from quant_fund.research_cluster.parallel_signal_evaluator import (
            ParallelSignalEvaluator,
        )
        evaluator = ParallelSignalEvaluator(config={"max_workers": 8})
        assert evaluator._max_workers == 8
