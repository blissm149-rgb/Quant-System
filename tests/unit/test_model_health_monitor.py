"""Tests for the observe-only model health monitoring system."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from quant_fund.monitoring.model_health_monitor import (
    DegradationEvent,
    ModelHealthMonitor,
)


class TestModelHealthMonitor:
    """Tests for the observe-only health monitoring system."""

    def test_prediction_drift_detection(self):
        """Feed drifted predictions, verify DegradationEvent emitted."""
        monitor = ModelHealthMonitor(
            event_log_path="/tmp/test_health.jsonl",
            drift_z_threshold=2.0,
        )
        monitor.register_baseline("test_model", mean=0.0, std=0.01)

        # Feed predictions that drift far from baseline
        for _ in range(100):
            monitor.observe_prediction("test_model", 0.1)  # 10x std away

        loaded_model = MagicMock()
        loaded_model.loaded_at = datetime.now(timezone.utc)
        event = monitor.check_health("test_model", loaded_model)

        assert event is not None
        assert event.metric == "prediction_drift_z"
        assert event.model_name == "test_model"
        assert event.value > 2.0

    def test_no_false_positive_on_normal_predictions(self):
        """Normal distribution within baseline should emit no events."""
        monitor = ModelHealthMonitor(
            event_log_path="/tmp/test_health.jsonl",
            drift_z_threshold=2.5,
        )
        monitor.register_baseline("test_model", mean=0.0, std=0.02)

        rng = np.random.default_rng(42)
        for _ in range(200):
            monitor.observe_prediction("test_model", rng.normal(0.0, 0.02))

        loaded_model = MagicMock()
        loaded_model.loaded_at = datetime.now(timezone.utc)
        event = monitor.check_health("test_model", loaded_model)

        assert event is None

    def test_prediction_collapse_detection(self):
        """All-constant predictions should trigger critical event."""
        monitor = ModelHealthMonitor(
            event_log_path="/tmp/test_health.jsonl",
        )

        # Feed 200 identical predictions
        for _ in range(200):
            monitor.observe_prediction("test_model", 0.5)

        loaded_model = MagicMock()
        loaded_model.loaded_at = datetime.now(timezone.utc)
        event = monitor.check_health("test_model", loaded_model)

        assert event is not None
        assert event.metric == "prediction_collapse_std"
        assert event.severity == "critical"

    def test_staleness_detection(self):
        """Model older than staleness_hours should trigger warning."""
        monitor = ModelHealthMonitor(
            event_log_path="/tmp/test_health.jsonl",
            staleness_hours=24,
        )
        monitor.register_baseline("test_model", mean=0.0, std=0.01)

        # Feed some predictions to avoid other triggers
        rng = np.random.default_rng(42)
        for _ in range(60):
            monitor.observe_prediction("test_model", rng.normal(0.0, 0.01))

        # Simulate a model loaded 48 hours ago
        loaded_model = MagicMock()
        loaded_model.loaded_at = datetime.now(timezone.utc) - timedelta(hours=48)

        event = monitor.check_health("test_model", loaded_model)
        assert event is not None
        assert event.metric == "model_age_hours"
        assert event.severity == "warning"
        assert event.value > 24

    def test_event_logged_to_file(self, tmp_path):
        """Verify events are appended to the JSONL log file."""
        log_path = tmp_path / "health_events.jsonl"
        monitor = ModelHealthMonitor(
            event_log_path=str(log_path),
            staleness_hours=1,
        )
        monitor.register_baseline("test_model", mean=0.0, std=0.01)

        # Feed some predictions
        rng = np.random.default_rng(42)
        for _ in range(60):
            monitor.observe_prediction("test_model", rng.normal(0.0, 0.01))

        loaded_model = MagicMock()
        loaded_model.loaded_at = datetime.now(timezone.utc) - timedelta(hours=48)

        monitor.check_health("test_model", loaded_model)

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        event_data = json.loads(lines[0])
        assert "model_name" in event_data
        assert "metric" in event_data

    def test_buffer_size_bounded(self):
        """Verify deque maxlen prevents unbounded memory growth."""
        monitor = ModelHealthMonitor(
            event_log_path="/tmp/test_health.jsonl",
            prediction_window=100,
        )

        for _ in range(500):
            monitor.observe_prediction("test_model", 0.5)

        assert len(monitor._prediction_buffers["test_model"]) == 100

    def test_health_check_with_insufficient_data_returns_none(self):
        """< 50 predictions should not trigger drift check."""
        monitor = ModelHealthMonitor(
            event_log_path="/tmp/test_health.jsonl",
        )
        monitor.register_baseline("test_model", mean=0.0, std=0.01)

        # Only 10 predictions (below threshold of 50)
        for _ in range(10):
            monitor.observe_prediction("test_model", 0.5)

        loaded_model = MagicMock()
        loaded_model.loaded_at = datetime.now(timezone.utc)
        event = monitor.check_health("test_model", loaded_model)
        assert event is None

    def test_monitor_never_modifies_model(self):
        """Verify no side effects on the model object."""
        monitor = ModelHealthMonitor(
            event_log_path="/tmp/test_health.jsonl",
        )
        monitor.register_baseline("test_model", mean=0.0, std=0.01)

        for _ in range(60):
            monitor.observe_prediction("test_model", 0.0)

        loaded_model = MagicMock(spec=["loaded_at", "name", "version"])
        loaded_model.loaded_at = datetime.now(timezone.utc)

        monitor.check_health("test_model", loaded_model)

        # No training methods should have been called
        assert not hasattr(loaded_model, "train") or not loaded_model.train.called
        assert not hasattr(loaded_model, "fit") or not loaded_model.fit.called

    def test_observe_prediction_creates_buffer_on_first_call(self):
        """Observing a new model auto-creates its buffer."""
        monitor = ModelHealthMonitor(
            event_log_path="/tmp/test_health.jsonl",
        )
        monitor.observe_prediction("new_model", 0.5)
        assert "new_model" in monitor._prediction_buffers
        assert len(monitor._prediction_buffers["new_model"]) == 1

    def test_critical_severity_on_extreme_drift(self):
        """Drift > 1.5x threshold should be critical, not just warning."""
        monitor = ModelHealthMonitor(
            event_log_path="/tmp/test_health.jsonl",
            drift_z_threshold=2.0,
        )
        monitor.register_baseline("test_model", mean=0.0, std=0.01)

        # Feed very extreme drift (should exceed 1.5x threshold)
        for _ in range(100):
            monitor.observe_prediction("test_model", 1.0)

        loaded_model = MagicMock()
        loaded_model.loaded_at = datetime.now(timezone.utc)
        event = monitor.check_health("test_model", loaded_model)

        assert event is not None
        assert event.severity == "critical"
