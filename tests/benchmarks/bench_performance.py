"""Performance benchmarks: feature computation, portfolio optimization,
order generation, event bus throughput.

These tests measure execution time and fail if performance degrades
beyond acceptable thresholds.  Designed to run in the nightly CI tier.
"""

import time
import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, make_returns, STANDARD_TICKERS, STANDARD_SECTORS

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ---------------------------------------------------------------------------
# 8.1  Feature Computation Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.tier4
class TestFeatureComputationPerformance:

    def test_10_tickers_252_days_under_5s(self):
        """Feature computation for 10 tickers × 252 days < 5s."""
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:10], periods=252, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max() + pd.Timedelta(days=1)
        engine = TechnicalIndicatorEngine()

        result, elapsed = _time_fn(engine.compute_all, ohlcv, as_of)

        assert elapsed < 5.0, f"Feature computation took {elapsed:.2f}s (limit: 5s)"
        assert result is not None
        assert len(result) > 0

    def test_5_tickers_500_days_under_5s(self):
        """Feature computation for 5 tickers × 500 days < 5s."""
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=500, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max() + pd.Timedelta(days=1)
        engine = TechnicalIndicatorEngine()

        result, elapsed = _time_fn(engine.compute_all, ohlcv, as_of)

        assert elapsed < 5.0, f"Feature computation took {elapsed:.2f}s (limit: 5s)"
        assert result is not None


# ---------------------------------------------------------------------------
# 8.2  Portfolio Optimization Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.tier4
class TestPortfolioOptimizationPerformance:

    def _make_inputs(self, n_tickers, n_factors=5, seed=SEED):
        rng = np.random.default_rng(seed)
        tickers = [f"T{i:04d}" for i in range(n_tickers)]
        alpha = pd.Series(rng.normal(0, 0.01, n_tickers), index=tickers)
        factors = [f"F{i}" for i in range(n_factors)]
        exposures = pd.DataFrame(
            rng.normal(0, 1, (n_tickers, n_factors)),
            index=tickers, columns=factors,
        )
        factor_cov = pd.DataFrame(
            np.eye(n_factors) * 0.01,
            index=factors, columns=factors,
        )
        return alpha, factor_cov, exposures

    def test_10_tickers_under_2s(self):
        """Portfolio optimization for 10 tickers < 2s."""
        from quant_fund.portfolio.portfolio_construction.portfolio_optimizer import PortfolioOptimizer
        from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintEngine

        alpha, factor_cov, exposures = self._make_inputs(10)
        constraints = ConstraintEngine().build_constraints()
        opt = PortfolioOptimizer()

        result, elapsed = _time_fn(opt.optimize, alpha, factor_cov, exposures, constraints)

        assert elapsed < 2.0, f"Optimization took {elapsed:.2f}s (limit: 2s)"
        assert len(result) == 10

    def test_100_tickers_under_5s(self):
        """Portfolio optimization for 100 tickers < 5s."""
        from quant_fund.portfolio.portfolio_construction.portfolio_optimizer import PortfolioOptimizer
        from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintEngine

        alpha, factor_cov, exposures = self._make_inputs(100)
        constraints = ConstraintEngine().build_constraints()
        opt = PortfolioOptimizer()

        result, elapsed = _time_fn(opt.optimize, alpha, factor_cov, exposures, constraints)

        assert elapsed < 5.0, f"Optimization took {elapsed:.2f}s (limit: 5s)"


# ---------------------------------------------------------------------------
# 8.3  Order Generation Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.tier4
class TestOrderGenerationPerformance:

    def test_100_trades_under_1s(self):
        """Generating 100 orders < 1s."""
        from quant_fund.execution.order_management.order_generator import OrderGenerator

        rng = np.random.default_rng(SEED)
        n = 100
        tickers = [f"T{i:04d}" for i in range(n)]
        target = pd.Series(rng.normal(0, 0.01, n), index=tickers)
        current = pd.Series(0.0, index=tickers)
        prices = pd.Series(rng.uniform(10, 500, n), index=tickers)

        gen = OrderGenerator()
        orders, elapsed = _time_fn(
            gen.generate_orders, target, current, prices, nav=1_000_000.0
        )

        assert elapsed < 1.0, f"Order generation took {elapsed:.2f}s (limit: 1s)"

    def test_500_trades_under_2s(self):
        """Generating orders for 500-ticker rebalance < 2s."""
        from quant_fund.execution.order_management.order_generator import OrderGenerator

        rng = np.random.default_rng(SEED)
        n = 500
        tickers = [f"T{i:04d}" for i in range(n)]
        target = pd.Series(rng.normal(0, 0.002, n), index=tickers)
        current = pd.Series(rng.normal(0, 0.001, n), index=tickers)
        prices = pd.Series(rng.uniform(10, 500, n), index=tickers)

        gen = OrderGenerator()
        orders, elapsed = _time_fn(
            gen.generate_orders, target, current, prices, nav=10_000_000.0
        )

        assert elapsed < 2.0, f"Order generation took {elapsed:.2f}s (limit: 2s)"


# ---------------------------------------------------------------------------
# 8.4  EventBus Throughput Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.tier4
class TestEventBusThroughput:

    def test_10k_events_throughput(self):
        """EventBus delivers 10,000 events in synchronous mode."""
        from quant_fund.infrastructure.event_bus import EventBus, Event, EventType

        bus = EventBus(synchronous=True)
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe("bench", handler, event_types={EventType.MARKET_DATA})

        events = [
            Event(event_type=EventType.MARKET_DATA, payload={"seq": i})
            for i in range(10_000)
        ]

        start = time.perf_counter()
        for e in events:
            bus.publish(e)
        elapsed = time.perf_counter() - start

        throughput = len(received) / max(elapsed, 1e-9)
        assert len(received) == 10_000, f"Only {len(received)}/10000 events delivered"
        # Expect > 1K events/sec minimum in synchronous mode
        assert throughput > 1_000, f"Throughput {throughput:.0f} events/s (need > 1K)"

    def test_1k_events_with_multiple_subscribers(self):
        """1K events × 5 subscribers all deliver correctly."""
        from quant_fund.infrastructure.event_bus import EventBus, Event, EventType

        bus = EventBus(synchronous=True)
        counts = {f"sub_{i}": 0 for i in range(5)}

        for i in range(5):
            sub_id = f"sub_{i}"
            def handler(event, sid=sub_id):
                counts[sid] += 1
            bus.subscribe(sub_id, handler, event_types={EventType.SIGNAL_GENERATED})

        for j in range(1_000):
            bus.publish(Event(event_type=EventType.SIGNAL_GENERATED, payload={"seq": j}))

        for sub_id, count in counts.items():
            assert count == 1_000, f"{sub_id} received {count}/1000"


# ---------------------------------------------------------------------------
# 8.5  Research Cycle Benchmark
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.tier4
class TestResearchCyclePerformance:

    def test_5_tickers_1_year_under_10s(self):
        """Full research cycle for 5 tickers × 252 days < 10s."""
        from quant_fund.main.research_runner import ResearchRunner
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=300, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max() + pd.Timedelta(days=1)

        runner = ResearchRunner({"universe": STANDARD_TICKERS[:5], "lookback_days": 252})
        runner.inject_components(
            feature_generators=TechnicalIndicatorEngine().generators,
            feature_normalizer=FeatureNormalizer(),
        )

        result, elapsed = _time_fn(runner.run_cycle, as_of, ohlcv)

        assert elapsed < 10.0, f"Research cycle took {elapsed:.2f}s (limit: 10s)"
