"""Experiment scheduler.

Manages a queue of research experiments (parameter sweeps, signal
searches). Assigns jobs to available workers, tracks status, and
stores results keyed by (experiment_id, parameters_hash).
"""

import hashlib
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from quant_fund.research_cluster.distributed_backtest_runner import (
    BacktestJob,
    BacktestResult,
    DistributedBacktestRunner,
)

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Experiment:
    """A research experiment to be scheduled."""

    experiment_id: str
    strategy_id: str
    description: str = ""
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    base_params: dict = field(default_factory=dict)
    priority: int = 0  # higher = more urgent
    status: ExperimentStatus = ExperimentStatus.QUEUED
    results: List[BacktestResult] = field(default_factory=list)
    submitted_at: Optional[pd.Timestamp] = None
    completed_at: Optional[pd.Timestamp] = None


class ExperimentScheduler:
    """Schedules and manages research experiments.

    Maintains a priority queue of experiments. Runs them via
    DistributedBacktestRunner. Stores results keyed by
    (experiment_id, parameters_hash) for deduplication.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._runner = DistributedBacktestRunner(cfg.get("runner", {}))
        self._queue: deque = deque()
        self._experiments: Dict[str, Experiment] = {}
        self._results_store: Dict[str, BacktestResult] = {}

    def submit_experiment(self, experiment: Experiment) -> str:
        """Submit an experiment to the queue.

        Returns
        -------
        str
            Experiment ID.
        """
        experiment.submitted_at = pd.Timestamp.now()
        experiment.status = ExperimentStatus.QUEUED
        self._experiments[experiment.experiment_id] = experiment
        self._queue.append(experiment.experiment_id)
        logger.info(
            "Experiment %s submitted (priority=%d, grid_size=%d)",
            experiment.experiment_id,
            experiment.priority,
            self._grid_size(experiment.param_grid),
        )
        return experiment.experiment_id

    def run_next(
        self,
        backtest_fn: Optional[Callable] = None,
    ) -> Optional[Experiment]:
        """Run the next experiment in the queue.

        Returns the completed experiment or None if queue is empty.
        """
        if not self._queue:
            return None

        # Sort by priority (higher first)
        sorted_ids = sorted(
            self._queue,
            key=lambda eid: self._experiments[eid].priority,
            reverse=True,
        )
        exp_id = sorted_ids[0]
        self._queue.remove(exp_id)

        experiment = self._experiments[exp_id]
        experiment.status = ExperimentStatus.RUNNING

        try:
            results = self._runner.run_parameter_sweep(
                strategy_id=experiment.strategy_id,
                param_grid=experiment.param_grid,
                backtest_fn=backtest_fn,
                base_params=experiment.base_params,
            )

            experiment.results = results
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = pd.Timestamp.now()

            # Store results with dedup key
            for r in results:
                key = self._result_key(exp_id, r.parameters)
                self._results_store[key] = r

            logger.info(
                "Experiment %s completed: %d results, best Sharpe=%.3f",
                exp_id,
                len(results),
                results[0].sharpe_ratio if results else 0.0,
            )

        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            logger.error("Experiment %s failed: %s", exp_id, e)

        return experiment

    def run_all(
        self, backtest_fn: Optional[Callable] = None
    ) -> List[Experiment]:
        """Run all queued experiments."""
        completed = []
        while self._queue:
            exp = self.run_next(backtest_fn)
            if exp is not None:
                completed.append(exp)
        return completed

    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a queued experiment."""
        if experiment_id not in self._experiments:
            return False
        exp = self._experiments[experiment_id]
        if exp.status == ExperimentStatus.QUEUED:
            exp.status = ExperimentStatus.CANCELLED
            if experiment_id in self._queue:
                self._queue.remove(experiment_id)
            return True
        return False

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)

    def get_queue_status(self) -> pd.DataFrame:
        """Get status of all experiments."""
        rows = []
        for exp in self._experiments.values():
            rows.append({
                "experiment_id": exp.experiment_id,
                "strategy_id": exp.strategy_id,
                "status": exp.status.value,
                "priority": exp.priority,
                "grid_size": self._grid_size(exp.param_grid),
                "num_results": len(exp.results),
                "submitted_at": exp.submitted_at,
                "completed_at": exp.completed_at,
            })
        return pd.DataFrame(rows)

    def get_best_result(
        self, experiment_id: str
    ) -> Optional[BacktestResult]:
        """Get the best result (by Sharpe) for an experiment."""
        exp = self._experiments.get(experiment_id)
        if exp is None or not exp.results:
            return None
        return max(exp.results, key=lambda r: r.sharpe_ratio)

    def lookup_result(
        self, experiment_id: str, parameters: dict
    ) -> Optional[BacktestResult]:
        """Look up a cached result by experiment ID and parameters."""
        key = self._result_key(experiment_id, parameters)
        return self._results_store.get(key)

    @staticmethod
    def _result_key(experiment_id: str, parameters: dict) -> str:
        """Generate a dedup key from experiment ID and parameters hash."""
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
        return f"{experiment_id}:{param_hash}"

    @staticmethod
    def _grid_size(param_grid: Dict[str, List[Any]]) -> int:
        """Compute total number of grid points."""
        if not param_grid:
            return 0
        size = 1
        for vals in param_grid.values():
            size *= len(vals)
        return size
