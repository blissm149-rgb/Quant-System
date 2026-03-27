"""Unit tests for TradingEngine research → optimization pipeline.

Verifies _run_research_and_optimize(), market hours gate, and
factor estimator injection work correctly within the TradingEngine.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.infrastructure.event_bus import EventType
from quant_fund.main.trading_engine import TradingEngine
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from tests.conftest import STANDARD_MARKET_DATA


@dataclass
class _FakeResearchResult:
    as_of: pd.Timestamp = None
    alpha_scores: Optional[pd.Series] = None
    status: str = "completed"
    validation_flags: List[str] = field(default_factory=list)


class _FakeResearchRunner:
    """Minimal mock that returns configurable alpha scores."""

    def __init__(self, alpha_scores=None):
        self._alpha_scores = alpha_scores
        self.call_count = 0

    def run_cycle(self, as_of=None, market_data=None):
        self.call_count += 1
        return _FakeResearchResult(
            as_of=as_of or pd.Timestamp.now(),
            alpha_scores=self._alpha_scores,
        )


def _make_engine(broker, **extras):
    e = TradingEngine(config={"synchronous": True})
    e.inject_components(
        broker=broker,
        order_generator=OrderGenerator(),
        order_router=OrderRouter(broker),
        kill_switch=KillSwitch(config={"drawdown_limit": 0.20}),
        **extras,
    )
    return e


@pytest.mark.unit
@pytest.mark.tier2
class TestTradingEngineResearchPipeline:
    """Research → factor model → optimization → target weights."""

    @pytest.fixture
    def broker(self):
        b = SimulationBroker(config={"initial_cash": 1_000_000})
        b.set_market_data(STANDARD_MARKET_DATA)
        return b

    def test_research_runner_injection(self, broker):
        """ResearchRunner is injected and stored."""
        runner = _FakeResearchRunner()
        engine = _make_engine(broker, research_runner=runner)
        assert engine._research_runner is runner

    def test_run_research_produces_target_weights(self, broker):
        """Full pipeline: research → alpha → normalized weights."""
        alpha = pd.Series({"AAPL": 0.5, "MSFT": -0.3, "GOOG": 0.2})
        runner = _FakeResearchRunner(alpha_scores=alpha)
        engine = _make_engine(broker, research_runner=runner)

        engine._run_research_and_optimize()

        assert engine._target_weights is not None
        assert len(engine._target_weights) == 3
        assert runner.call_count == 1

    def test_run_research_with_optimizer(self, broker):
        """When portfolio_optimizer is injected, optimize() is called."""
        alpha = pd.Series({"AAPL": 0.5, "MSFT": -0.3})
        runner = _FakeResearchRunner(alpha_scores=alpha)

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = pd.Series(
            {"AAPL": 0.03, "MSFT": -0.02}
        )

        engine = _make_engine(
            broker,
            research_runner=runner,
            portfolio_optimizer=mock_optimizer,
        )
        engine._run_research_and_optimize()

        mock_optimizer.optimize.assert_called_once()
        assert engine._target_weights is not None
        assert engine._target_weights["AAPL"] == pytest.approx(0.03)

    def test_run_research_no_alpha_scores_no_weights(self, broker):
        """If research produces no alpha scores, weights are not set."""
        runner = _FakeResearchRunner(alpha_scores=None)
        engine = _make_engine(broker, research_runner=runner)
        engine._target_weights = None

        engine._run_research_and_optimize()

        assert engine._target_weights is None

    def test_run_research_empty_alpha_scores_no_weights(self, broker):
        """If research produces empty alpha scores, weights are not set."""
        runner = _FakeResearchRunner(alpha_scores=pd.Series(dtype=float))
        engine = _make_engine(broker, research_runner=runner)
        engine._target_weights = None

        engine._run_research_and_optimize()

        assert engine._target_weights is None

    def test_no_research_runner_is_noop(self, broker):
        """Without research_runner, _run_research_and_optimize is a no-op."""
        engine = _make_engine(broker)
        engine._target_weights = pd.Series({"AAPL": 0.05})

        engine._run_research_and_optimize()

        # Existing weights are untouched
        assert engine._target_weights["AAPL"] == pytest.approx(0.05)

    def test_research_exception_does_not_crash(self, broker):
        """Exception in research cycle is logged, not propagated."""
        runner = MagicMock()
        runner.run_cycle.side_effect = RuntimeError("data feed down")
        engine = _make_engine(broker, research_runner=runner)
        engine._target_weights = None

        engine._run_research_and_optimize()  # should not raise

        assert engine._target_weights is None

    def test_alpha_scores_stored(self, broker):
        """Latest alpha scores are cached for signal monitoring."""
        alpha = pd.Series({"AAPL": 1.0, "MSFT": 0.5})
        runner = _FakeResearchRunner(alpha_scores=alpha)
        engine = _make_engine(broker, research_runner=runner)

        engine._run_research_and_optimize()

        assert engine._latest_alpha_scores is not None
        assert "AAPL" in engine._latest_alpha_scores.index

    def test_signal_event_published(self, broker):
        """Research producing alpha publishes SIGNAL_GENERATED event."""
        alpha = pd.Series({"AAPL": 0.5})
        runner = _FakeResearchRunner(alpha_scores=alpha)
        engine = _make_engine(broker, research_runner=runner)

        events_received = []
        engine.event_bus.subscribe(
            "test_signal",
            lambda e: events_received.append(e),
            {EventType.SIGNAL_GENERATED},
        )

        engine._run_research_and_optimize()

        assert len(events_received) >= 1

    def test_factor_estimator_injection(self, broker):
        """Factor exposure and covariance estimators are injected."""
        mock_fe = MagicMock()
        mock_fc = MagicMock()
        engine = _make_engine(
            broker,
            factor_exposure_estimator=mock_fe,
            factor_covariance_estimator=mock_fc,
        )
        assert engine._factor_exposure_estimator is mock_fe
        assert engine._factor_covariance_estimator is mock_fc

    def test_factor_estimators_called_when_data_available(self, broker):
        """Factor estimators are called when returns data is provided."""
        alpha = pd.Series({"AAPL": 0.5, "MSFT": -0.3})
        runner = _FakeResearchRunner(alpha_scores=alpha)

        mock_fe = MagicMock()
        mock_fe.estimate.return_value = pd.DataFrame(
            {"market": [1.0, 1.1]}, index=["AAPL", "MSFT"]
        )
        mock_fc = MagicMock()
        mock_fc.estimate.return_value = pd.DataFrame(
            {"market": [0.04]}, index=["market"]
        )

        stock_returns = pd.DataFrame(
            {"AAPL": [0.01, -0.02], "MSFT": [0.02, 0.01]}
        )
        factor_returns = pd.DataFrame({"market": [0.01, -0.01]})

        engine = _make_engine(
            broker,
            research_runner=runner,
            factor_exposure_estimator=mock_fe,
            factor_covariance_estimator=mock_fc,
            stock_returns=stock_returns,
            factor_returns=factor_returns,
        )
        engine._run_research_and_optimize()

        mock_fe.estimate.assert_called_once()
        mock_fc.estimate.assert_called_once()


@pytest.mark.unit
@pytest.mark.tier2
class TestTradingEngineMarketHoursGate:
    """Market hours enforcement in the main loop."""

    @pytest.fixture
    def broker(self):
        b = SimulationBroker(config={"initial_cash": 1_000_000})
        b.set_market_data(STANDARD_MARKET_DATA)
        return b

    def test_market_hours_enforcer_injection(self, broker):
        """MarketHoursEnforcer is injected and stored."""
        mock_enforcer = MagicMock()
        engine = _make_engine(broker, market_hours_enforcer=mock_enforcer)
        assert engine._market_hours_enforcer is mock_enforcer

    def test_no_market_hours_enforcer_allows_convergence(self, broker):
        """Without enforcer, convergence always runs."""
        engine = _make_engine(broker)
        weights = pd.Series({"AAPL": 0.05})
        engine.update_target_weights(weights)

        engine._convergence_tick()

        positions = broker.get_positions()
        assert len(positions) > 0

    def test_research_interval_configurable(self):
        """research_interval_s is read from config."""
        engine = TradingEngine(config={
            "synchronous": True,
            "research_interval_s": 1800.0,
        })
        assert engine._research_interval_s == 1800.0
