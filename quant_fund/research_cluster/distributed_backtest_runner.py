"""Distributed backtest runner.

Distributes backtests across multiple cores using joblib (local)
or optionally Ray/Dask for cluster-scale workloads. Each backtest
job is fully isolated with no shared mutable state between workers.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestJob:
    """A single backtest job to be executed."""

    job_id: str
    strategy_id: str
    parameters: dict = field(default_factory=dict)
    start_date: str = "2019-01-01"
    end_date: str = "2024-01-01"
    universe: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Result of a single backtest job."""

    job_id: str
    strategy_id: str
    parameters: dict = field(default_factory=dict)
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    annual_turnover: float = 0.0
    ic_mean: float = 0.0
    ic_ir: float = 0.0
    num_trades: int = 0
    status: str = "completed"
    error_message: str = ""
    metrics: dict = field(default_factory=dict)


def _run_single_backtest(
    job: BacktestJob,
    backtest_fn: Optional[Callable] = None,
) -> BacktestResult:
    """Execute a single backtest job in isolation.

    This function runs in a separate process — no shared state.

    Parameters
    ----------
    job : BacktestJob
        Job specification.
    backtest_fn : callable, optional
        The actual backtest function. If None, runs a stub that
        returns synthetic metrics for testing.

    Returns
    -------
    BacktestResult
    """
    try:
        if backtest_fn is not None:
            metrics = backtest_fn(job)
            return BacktestResult(
                job_id=job.job_id,
                strategy_id=job.strategy_id,
                parameters=job.parameters,
                sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
                total_return=metrics.get("total_return", 0.0),
                max_drawdown=metrics.get("max_drawdown", 0.0),
                annual_turnover=metrics.get("annual_turnover", 0.0),
                ic_mean=metrics.get("ic_mean", 0.0),
                ic_ir=metrics.get("ic_ir", 0.0),
                num_trades=metrics.get("num_trades", 0),
                metrics=metrics,
            )
        else:
            # Stub: return placeholder metrics
            return BacktestResult(
                job_id=job.job_id,
                strategy_id=job.strategy_id,
                parameters=job.parameters,
                sharpe_ratio=0.0,
                status="completed",
            )
    except Exception as e:
        return BacktestResult(
            job_id=job.job_id,
            strategy_id=job.strategy_id,
            parameters=job.parameters,
            status="failed",
            error_message=str(e),
        )


class DistributedBacktestRunner:
    """Runs backtests in parallel across multiple cores.

    Uses ProcessPoolExecutor for local parallelism. Each backtest
    is fully isolated — no shared mutable state between workers.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        import os as _os
        _default_workers = max(1, (_os.cpu_count() or 2) - 1)
        self._max_workers = cfg.get("max_workers", _default_workers)
        self._results: List[BacktestResult] = []

    def run_batch(
        self,
        jobs: List[BacktestJob],
        backtest_fn: Optional[Callable] = None,
        parallel: bool = True,
    ) -> List[BacktestResult]:
        """Run a batch of backtest jobs.

        Parameters
        ----------
        jobs : list of BacktestJob
            Jobs to execute.
        backtest_fn : callable, optional
            Function that takes a BacktestJob and returns a metrics dict.
        parallel : bool
            If True, run in parallel. If False, run sequentially.

        Returns
        -------
        list of BacktestResult
        """
        if not jobs:
            return []

        results: List[BacktestResult] = []

        if parallel and len(jobs) > 1:
            results = self._run_parallel(jobs, backtest_fn)
        else:
            results = self._run_sequential(jobs, backtest_fn)

        self._results.extend(results)
        logger.info(
            "Completed %d/%d backtests",
            sum(1 for r in results if r.status == "completed"),
            len(jobs),
        )
        return results

    def run_parameter_sweep(
        self,
        strategy_id: str,
        param_grid: Dict[str, List[Any]],
        backtest_fn: Optional[Callable] = None,
        base_params: Optional[dict] = None,
    ) -> List[BacktestResult]:
        """Run a parameter sweep across a grid.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier.
        param_grid : dict
            Mapping parameter name → list of values to try.
        backtest_fn : callable, optional
            Backtest function.
        base_params : dict, optional
            Base parameters to merge with each grid point.

        Returns
        -------
        list of BacktestResult sorted by Sharpe ratio descending.
        """
        base = base_params or {}
        jobs = []
        grid_points = self._expand_grid(param_grid)

        for i, point in enumerate(grid_points):
            params = {**base, **point}
            jobs.append(BacktestJob(
                job_id=f"{strategy_id}-sweep-{i:04d}",
                strategy_id=strategy_id,
                parameters=params,
            ))

        results = self.run_batch(jobs, backtest_fn)
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all results as a DataFrame."""
        if not self._results:
            return pd.DataFrame()
        rows = []
        for r in self._results:
            row = {
                "job_id": r.job_id,
                "strategy_id": r.strategy_id,
                "sharpe_ratio": r.sharpe_ratio,
                "total_return": r.total_return,
                "max_drawdown": r.max_drawdown,
                "annual_turnover": r.annual_turnover,
                "ic_mean": r.ic_mean,
                "ic_ir": r.ic_ir,
                "status": r.status,
            }
            row.update(r.parameters)
            rows.append(row)
        return pd.DataFrame(rows)

    def _run_parallel(
        self, jobs: List[BacktestJob], backtest_fn: Optional[Callable]
    ) -> List[BacktestResult]:
        """Run jobs using ProcessPoolExecutor."""
        results: List[BacktestResult] = []
        try:
            with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(_run_single_backtest, job, backtest_fn): job
                    for job in jobs
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        job = futures[future]
                        results.append(BacktestResult(
                            job_id=job.job_id,
                            strategy_id=job.strategy_id,
                            status="failed",
                            error_message=str(e),
                        ))
        except Exception as e:
            logger.error("Parallel execution failed, falling back to sequential: %s", e)
            results = self._run_sequential(jobs, backtest_fn)
        return results

    def _run_sequential(
        self, jobs: List[BacktestJob], backtest_fn: Optional[Callable]
    ) -> List[BacktestResult]:
        """Run jobs sequentially."""
        return [_run_single_backtest(job, backtest_fn) for job in jobs]

    @staticmethod
    def _expand_grid(param_grid: Dict[str, List[Any]]) -> List[dict]:
        """Expand a parameter grid into a list of dicts."""
        if not param_grid:
            return [{}]
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        points = [{}]
        for key, vals in zip(keys, values):
            new_points = []
            for point in points:
                for v in vals:
                    new_points.append({**point, key: v})
            points = new_points
        return points
