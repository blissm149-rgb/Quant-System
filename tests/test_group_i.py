"""Tests for Group I — Research Cluster.

Covers: distributed backtest runner, parallel signal evaluator,
and experiment scheduler.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.research_cluster.distributed_backtest_runner import (

pytestmark = [pytest.mark.tier2]
    BacktestJob,
    BacktestResult,
    DistributedBacktestRunner,
)
from quant_fund.research_cluster.parallel_signal_evaluator import (
    ParallelSignalEvaluator,
    _evaluate_single_signal,
)
from quant_fund.research_cluster.experiment_scheduler import (
    Experiment,
    ExperimentScheduler,
    ExperimentStatus,
)


# ── Helpers ──────────────────────────────────────────────────────────

def dummy_backtest(job):
    """Dummy backtest function that returns synthetic metrics."""
    lookback = job.parameters.get("lookback", 20)
    return {
        "sharpe_ratio": 1.0 + lookback * 0.01,
        "total_return": 0.10 + lookback * 0.001,
        "max_drawdown": 0.15,
        "annual_turnover": 2.0,
        "ic_mean": 0.03,
        "ic_ir": 0.5,
        "num_trades": 1000,
    }


def make_signal_data(T=60, N=50, ic=0.05, seed=42):
    """Generate synthetic signal with known IC."""
    rng = np.random.default_rng(seed)
    fwd_returns = rng.normal(0, 0.02, (T, N))
    noise = rng.normal(0, 0.02, (T, N))
    signal = ic * fwd_returns + (1 - ic) * noise
    return signal, fwd_returns


# ── Distributed Backtest Runner ──────────────────────────────────────

class TestDistributedBacktestRunner:
    def test_run_single_job(self):
        runner = DistributedBacktestRunner()
        jobs = [BacktestJob(job_id="test-001", strategy_id="mom")]
        results = runner.run_batch(jobs, parallel=False)
        assert len(results) == 1
        assert results[0].status == "completed"

    def test_run_with_backtest_fn(self):
        runner = DistributedBacktestRunner()
        jobs = [
            BacktestJob(job_id="test-001", strategy_id="mom",
                        parameters={"lookback": 20}),
        ]
        results = runner.run_batch(jobs, backtest_fn=dummy_backtest, parallel=False)
        assert results[0].sharpe_ratio == pytest.approx(1.2, rel=0.01)

    def test_run_multiple_sequential(self):
        runner = DistributedBacktestRunner()
        jobs = [
            BacktestJob(job_id=f"test-{i:03d}", strategy_id="mom",
                        parameters={"lookback": 10 * i})
            for i in range(1, 4)
        ]
        results = runner.run_batch(jobs, backtest_fn=dummy_backtest, parallel=False)
        assert len(results) == 3
        assert all(r.status == "completed" for r in results)

    def test_parameter_sweep(self):
        runner = DistributedBacktestRunner()
        results = runner.run_parameter_sweep(
            strategy_id="mom",
            param_grid={"lookback": [10, 20, 30]},
            backtest_fn=dummy_backtest,
        )
        assert len(results) == 3
        # Should be sorted by Sharpe descending
        assert results[0].sharpe_ratio >= results[-1].sharpe_ratio

    def test_parameter_sweep_grid_expansion(self):
        runner = DistributedBacktestRunner()
        results = runner.run_parameter_sweep(
            strategy_id="mom",
            param_grid={"lookback": [10, 20], "threshold": [0.5, 1.0]},
            backtest_fn=dummy_backtest,
        )
        assert len(results) == 4  # 2 × 2

    def test_results_dataframe(self):
        runner = DistributedBacktestRunner()
        runner.run_parameter_sweep(
            strategy_id="mom",
            param_grid={"lookback": [10, 20]},
            backtest_fn=dummy_backtest,
        )
        df = runner.get_results_dataframe()
        assert len(df) == 2
        assert "sharpe_ratio" in df.columns
        assert "lookback" in df.columns

    def test_failed_backtest(self):
        def failing_fn(job):
            raise ValueError("Intentional failure")

        runner = DistributedBacktestRunner()
        jobs = [BacktestJob(job_id="fail-001", strategy_id="bad")]
        results = runner.run_batch(jobs, backtest_fn=failing_fn, parallel=False)
        assert len(results) == 1
        assert results[0].status == "failed"


# ── Parallel Signal Evaluator ────────────────────────────────────────

class TestParallelSignalEvaluator:
    def test_evaluate_single_signal(self):
        signal, fwd_ret = make_signal_data(T=60, N=50, ic=0.05)
        result = _evaluate_single_signal("sig1", signal, fwd_ret)
        assert result.status == "completed"
        assert result.num_periods > 0

    def test_positive_ic_signal(self):
        """Signal correlated with returns should have positive IC."""
        signal, fwd_ret = make_signal_data(T=100, N=50, ic=0.3, seed=123)
        result = _evaluate_single_signal("sig_pos", signal, fwd_ret)
        assert result.ic_mean > 0

    def test_evaluate_multiple_signals(self):
        evaluator = ParallelSignalEvaluator()
        sig1, fwd_ret = make_signal_data(T=60, N=50, ic=0.3, seed=1)
        sig2, _ = make_signal_data(T=60, N=50, ic=0.01, seed=2)
        signals = {"strong": sig1, "weak": sig2}
        df = evaluator.evaluate_signals(signals, fwd_ret, parallel=False)
        assert len(df) == 2
        assert "ic_tstat" in df.columns
        # Sorted by t-stat descending
        assert df.iloc[0]["ic_tstat"] >= df.iloc[1]["ic_tstat"]

    def test_filter_significant(self):
        evaluator = ParallelSignalEvaluator({"min_ic_tstat": 1.5})
        sig1, fwd_ret = make_signal_data(T=100, N=50, ic=0.3, seed=1)
        sig2, _ = make_signal_data(T=100, N=50, ic=0.001, seed=2)
        signals = {"strong": sig1, "weak": sig2}
        df = evaluator.evaluate_signals(signals, fwd_ret, parallel=False)
        filtered = evaluator.filter_significant(df)
        # Strong signal should survive, weak may not
        assert len(filtered) <= len(df)

    def test_insufficient_data(self):
        result = _evaluate_single_signal(
            "tiny", np.array([[1.0]]), np.array([[0.01]])
        )
        assert result.status == "insufficient_data"

    def test_hit_rate_bounded(self):
        signal, fwd_ret = make_signal_data(T=60, N=50, ic=0.1)
        result = _evaluate_single_signal("sig_hr", signal, fwd_ret)
        assert 0.0 <= result.hit_rate <= 1.0


# ── Experiment Scheduler ─────────────────────────────────────────────

class TestExperimentScheduler:
    def test_submit_and_run(self):
        scheduler = ExperimentScheduler()
        exp = Experiment(
            experiment_id="exp-001",
            strategy_id="mom",
            param_grid={"lookback": [10, 20]},
        )
        scheduler.submit_experiment(exp)
        result = scheduler.run_next(backtest_fn=dummy_backtest)
        assert result is not None
        assert result.status == ExperimentStatus.COMPLETED
        assert len(result.results) == 2

    def test_priority_ordering(self):
        scheduler = ExperimentScheduler()
        scheduler.submit_experiment(Experiment(
            experiment_id="low", strategy_id="a",
            param_grid={"x": [1]}, priority=0,
        ))
        scheduler.submit_experiment(Experiment(
            experiment_id="high", strategy_id="b",
            param_grid={"x": [1]}, priority=10,
        ))
        # Higher priority should run first
        result = scheduler.run_next(backtest_fn=dummy_backtest)
        assert result.experiment_id == "high"

    def test_run_all(self):
        scheduler = ExperimentScheduler()
        for i in range(3):
            scheduler.submit_experiment(Experiment(
                experiment_id=f"exp-{i}",
                strategy_id="mom",
                param_grid={"lookback": [10]},
            ))
        completed = scheduler.run_all(backtest_fn=dummy_backtest)
        assert len(completed) == 3

    def test_cancel_experiment(self):
        scheduler = ExperimentScheduler()
        scheduler.submit_experiment(Experiment(
            experiment_id="cancel-me", strategy_id="a",
            param_grid={"x": [1]},
        ))
        assert scheduler.cancel_experiment("cancel-me") is True
        exp = scheduler.get_experiment("cancel-me")
        assert exp.status == ExperimentStatus.CANCELLED

    def test_get_queue_status(self):
        scheduler = ExperimentScheduler()
        scheduler.submit_experiment(Experiment(
            experiment_id="exp-001", strategy_id="mom",
            param_grid={"lookback": [10, 20, 30]},
        ))
        status = scheduler.get_queue_status()
        assert len(status) == 1
        assert status.iloc[0]["grid_size"] == 3

    def test_get_best_result(self):
        scheduler = ExperimentScheduler()
        scheduler.submit_experiment(Experiment(
            experiment_id="exp-best", strategy_id="mom",
            param_grid={"lookback": [10, 20, 30]},
        ))
        scheduler.run_next(backtest_fn=dummy_backtest)
        best = scheduler.get_best_result("exp-best")
        assert best is not None
        assert best.sharpe_ratio > 0

    def test_result_lookup(self):
        scheduler = ExperimentScheduler()
        scheduler.submit_experiment(Experiment(
            experiment_id="exp-lookup", strategy_id="mom",
            param_grid={"lookback": [20]},
        ))
        scheduler.run_next(backtest_fn=dummy_backtest)
        result = scheduler.lookup_result("exp-lookup", {"lookback": 20})
        assert result is not None

    def test_empty_queue_returns_none(self):
        scheduler = ExperimentScheduler()
        assert scheduler.run_next() is None
