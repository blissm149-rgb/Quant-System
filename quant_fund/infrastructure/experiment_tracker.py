"""Experiment tracking for ML research reproducibility.

Logs parameters, metrics, and artifacts for each experiment run.
Supports comparison across experiments. Storage: JSON files in
output/experiments/.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Centralized experiment logging and comparison.

    Each experiment gets a unique ID and stores parameters, metrics,
    feature sets, and artifact paths.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._output_dir = cfg.get("experiment_dir", "output/experiments")
        self._experiments: Dict[str, dict] = {}
        self._current_experiment: Optional[str] = None

    def start_experiment(
        self,
        experiment_id: str,
        description: str = "",
    ) -> str:
        """Start a new experiment.

        Args:
            experiment_id: Unique identifier for the experiment.
            description: Human-readable description.

        Returns:
            The experiment_id.
        """
        self._experiments[experiment_id] = {
            "id": experiment_id,
            "description": description,
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "params": {},
            "metrics": {},
            "artifacts": [],
            "feature_set": [],
            "status": "running",
        }
        self._current_experiment = experiment_id
        return experiment_id

    def log_params(self, params: Dict[str, Any], experiment_id: Optional[str] = None) -> None:
        """Log parameters for an experiment."""
        eid = experiment_id or self._current_experiment
        if eid not in self._experiments:
            raise ValueError(f"Experiment {eid} not found")
        self._experiments[eid]["params"].update(params)

    def log_metrics(self, metrics: Dict[str, float], experiment_id: Optional[str] = None) -> None:
        """Log metrics for an experiment."""
        eid = experiment_id or self._current_experiment
        if eid not in self._experiments:
            raise ValueError(f"Experiment {eid} not found")
        self._experiments[eid]["metrics"].update(metrics)

    def log_artifact(self, artifact_path: str, experiment_id: Optional[str] = None) -> None:
        """Log an artifact path for an experiment."""
        eid = experiment_id or self._current_experiment
        if eid not in self._experiments:
            raise ValueError(f"Experiment {eid} not found")
        self._experiments[eid]["artifacts"].append(artifact_path)

    def log_feature_set(self, feature_names: List[str], experiment_id: Optional[str] = None) -> None:
        """Log the feature set used in an experiment."""
        eid = experiment_id or self._current_experiment
        if eid not in self._experiments:
            raise ValueError(f"Experiment {eid} not found")
        self._experiments[eid]["feature_set"] = list(feature_names)

    def end_experiment(self, experiment_id: Optional[str] = None) -> dict:
        """Mark an experiment as completed and return its record."""
        eid = experiment_id or self._current_experiment
        if eid not in self._experiments:
            raise ValueError(f"Experiment {eid} not found")
        self._experiments[eid]["ended_at"] = datetime.now().isoformat()
        self._experiments[eid]["status"] = "completed"
        if self._current_experiment == eid:
            self._current_experiment = None
        return self._experiments[eid]

    def get_experiment(self, experiment_id: str) -> dict:
        """Retrieve a single experiment record."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        return self._experiments[experiment_id]

    def compare_experiments(self, experiment_ids: List[str]) -> List[dict]:
        """Compare metrics across experiments.

        Returns:
            List of dicts with id, params summary, and all metrics.
        """
        results = []
        for eid in experiment_ids:
            if eid not in self._experiments:
                continue
            exp = self._experiments[eid]
            row = {"id": eid, "status": exp["status"]}
            row.update(exp["metrics"])
            row["n_features"] = len(exp["feature_set"])
            results.append(row)
        return results

    def save(self, experiment_id: Optional[str] = None) -> str:
        """Save experiment to JSON file.

        Returns:
            Path to the saved file.
        """
        eid = experiment_id or self._current_experiment
        if eid not in self._experiments:
            raise ValueError(f"Experiment {eid} not found")

        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, f"{eid}.json")
        with open(path, "w") as f:
            json.dump(self._experiments[eid], f, indent=2, default=str)
        return path

    @property
    def all_experiments(self) -> Dict[str, dict]:
        return dict(self._experiments)
