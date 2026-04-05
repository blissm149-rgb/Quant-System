"""Observe-only model health monitoring for the live trading loop.

Tracks prediction drift, model staleness, and prediction collapse.
Logs degradation events but NEVER triggers retraining. The offline
pipeline reads these events to prioritize which models to retrain.
"""

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DegradationEvent:
    """A single model degradation event."""

    model_name: str
    metric: str
    value: float
    threshold: float
    timestamp: str
    severity: str  # "warning" | "critical"


class ModelHealthMonitor:
    """Observes model behavior during live trading.

    Logs degradation events but NEVER triggers retraining.
    The offline pipeline reads these events to prioritize
    which models to retrain next.
    """

    def __init__(
        self,
        event_log_path: str = "logs/health_events.jsonl",
        prediction_window: int = 500,
        drift_z_threshold: float = 2.5,
        staleness_hours: int = 48,
    ):
        self.event_log_path = Path(event_log_path)
        self.prediction_window = prediction_window
        self.drift_z_threshold = drift_z_threshold
        self.staleness_hours = staleness_hours
        self._prediction_buffers: Dict[str, deque] = {}
        self._baseline_stats: Dict[str, Dict] = {}
        self._events: List[DegradationEvent] = []

    def register_baseline(self, model_name: str, mean: float, std: float):
        """Set baseline prediction distribution from validation metrics."""
        self._baseline_stats[model_name] = {"mean": mean, "std": std}
        self._prediction_buffers[model_name] = deque(maxlen=self.prediction_window)

    def observe_prediction(self, model_name: str, prediction: float):
        """Record a single prediction. Called every iteration of the live loop."""
        if model_name not in self._prediction_buffers:
            self._prediction_buffers[model_name] = deque(maxlen=self.prediction_window)
        self._prediction_buffers[model_name].append(prediction)

    def check_health(
        self, model_name: str, loaded_model
    ) -> Optional[DegradationEvent]:
        """Run health checks for a model.

        Returns a DegradationEvent if any threshold is breached, else None.
        Call once per bar/tick cycle.
        """
        events = []

        drift_event = self._check_prediction_drift(model_name)
        if drift_event:
            events.append(drift_event)

        staleness_event = self._check_staleness(model_name, loaded_model)
        if staleness_event:
            events.append(staleness_event)

        collapse_event = self._check_prediction_collapse(model_name)
        if collapse_event:
            events.append(collapse_event)

        for event in events:
            self._events.append(event)
            self._flush_event(event)

        return events[0] if events else None

    def _check_prediction_drift(
        self, model_name: str
    ) -> Optional[DegradationEvent]:
        buf = self._prediction_buffers.get(model_name)
        baseline = self._baseline_stats.get(model_name)
        if not buf or len(buf) < 50 or not baseline:
            return None

        recent_mean = np.mean(list(buf))
        z_score = abs(recent_mean - baseline["mean"]) / max(baseline["std"], 1e-10)

        if z_score > self.drift_z_threshold:
            return DegradationEvent(
                model_name=model_name,
                metric="prediction_drift_z",
                value=round(float(z_score), 4),
                threshold=self.drift_z_threshold,
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=(
                    "critical"
                    if z_score > self.drift_z_threshold * 1.5
                    else "warning"
                ),
            )
        return None

    def _check_staleness(
        self, model_name: str, loaded_model
    ) -> Optional[DegradationEvent]:
        age_hours = (
            datetime.now(timezone.utc) - loaded_model.loaded_at
        ).total_seconds() / 3600.0

        if age_hours > self.staleness_hours:
            return DegradationEvent(
                model_name=model_name,
                metric="model_age_hours",
                value=round(age_hours, 1),
                threshold=float(self.staleness_hours),
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity="warning",
            )
        return None

    def _check_prediction_collapse(
        self, model_name: str
    ) -> Optional[DegradationEvent]:
        buf = self._prediction_buffers.get(model_name)
        if not buf or len(buf) < 100:
            return None

        std = float(np.std(list(buf)))
        if std < 1e-8:
            return DegradationEvent(
                model_name=model_name,
                metric="prediction_collapse_std",
                value=round(std, 12),
                threshold=1e-8,
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity="critical",
            )
        return None

    def _flush_event(self, event: DegradationEvent):
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.event_log_path, "a") as f:
            f.write(json.dumps(event.__dict__) + "\n")
        logger.warning(
            "HEALTH [%s] %s: %s=%s (threshold=%s)",
            event.severity.upper(),
            event.model_name,
            event.metric,
            event.value,
            event.threshold,
        )
