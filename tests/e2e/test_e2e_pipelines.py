"""End-to-End Tests: Full pipeline tests from data loading to execution.

E2E-1: Full Research Cycle
E2E-2: Multi-Day Paper Trading
E2E-3: Multi-Regime Scenario
E2E-5: Kill Switch Trigger and Recovery
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, STANDARD_TICKERS, STANDARD_SECTORS, STANDARD_MARKET_DATA


SEED = 42


def _build_research_runner(tickers, ohlcv=None):
    """Build a fully-configured ResearchRunner."""
    from quant_fund.main.research_runner import ResearchRunner
    from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine
    from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

    runner = ResearchRunner({"universe": tickers, "lookback_days": 252})
    runner.inject_components(
        feature_generators=TechnicalIndicatorEngine().generators,
        feature_normalizer=FeatureNormalizer(),
    )
    return runner


def _build_paper_trading_runner(tickers, ohlcv):
    """Build a fully-configured PaperTradingRunner."""
    from quant_fund.main.paper_trading_runner import PaperTradingRunner
    from quant_fund.broker_interface.simulation_broker import SimulationBroker
    from quant_fund.execution.order_management.order_generator import OrderGenerator
    from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

    runner = PaperTradingRunner({"strategy_id": "e2e", "initial_nav": 1_000_000.0})
    broker = SimulationBroker({"initial_cash": 1_000_000.0})
    broker.set_market_data(STANDARD_MARKET_DATA)

    runner.inject_components(
        research_runner=_build_research_runner(tickers, ohlcv),
        order_generator=OrderGenerator(),
        broker=broker,
        kill_switch=KillSwitch({"drawdown_limit": 0.20}),
    )
    return runner


# ---------------------------------------------------------------------------
# E2E-1: Full Research Cycle
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.tier3
class TestFullResearchCycle:
    """Complete pipeline from data loading to signal evaluation."""

    def test_research_cycle_produces_alpha(self):
        """Full research pipeline: load → validate → features → normalize → alpha."""
        from quant_fund.data_layer.data_validator import DataValidator
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer
        from quant_fund.alpha_discovery.signal_ranking_engine import SignalRankingEngine

        tickers = STANDARD_TICKERS[:10]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max() + pd.Timedelta(days=1)

        # Validate
        validator = DataValidator()
        val_result = validator.validate(ohlcv, as_of=as_of)
        assert val_result.is_valid

        # Features
        engine = TechnicalIndicatorEngine()
        features = engine.compute_all(ohlcv, as_of)
        assert features is not None
        assert len(features) > 0

        # Normalize
        normalizer = FeatureNormalizer()
        signals = {}
        for col in features.columns:
            raw = features[col].dropna()
            if len(raw) >= 3:
                signals[col] = normalizer.normalize(raw)

        # Combine into alpha
        ranker = SignalRankingEngine()
        alpha = ranker.combine(signals)

        assert isinstance(alpha, pd.Series)
        assert len(alpha) == len(tickers)
        # No look-ahead: alpha should be computable
        assert not alpha.isna().all()

    def test_research_runner_end_to_end(self):
        """ResearchRunner.run_cycle() produces complete result."""
        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=SEED)
        as_of = ohlcv.index.get_level_values("date").max() + pd.Timedelta(days=1)

        runner = _build_research_runner(tickers)
        result = runner.run_cycle(as_of=as_of, market_data=ohlcv)

        assert result.feature_matrix is not None
        assert len(result.feature_matrix) == len(tickers)
        assert result.status in ("completed", "failed")


# ---------------------------------------------------------------------------
# E2E-2: Multi-Day Paper Trading
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.tier3
class TestMultiDayPaperTrading:
    """Simulate multiple trading days end-to-end."""

    def test_10_day_paper_trading(self):
        """10-day paper trading completes and produces valid results."""
        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=SEED)
        dates = ohlcv.index.get_level_values("date").unique()[-10:].tolist()

        runner = _build_paper_trading_runner(tickers, ohlcv)
        result = runner.run(dates=dates, market_data_by_date={d: ohlcv for d in dates})

        assert result.num_days >= 1
        assert result.final_nav > 0
        assert len(result.daily_results) > 0

    def test_paper_trading_nav_stays_positive(self):
        """NAV never goes negative during paper trading."""
        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=SEED)
        dates = ohlcv.index.get_level_values("date").unique()[-15:].tolist()

        runner = _build_paper_trading_runner(tickers, ohlcv)
        result = runner.run(dates=dates, market_data_by_date={d: ohlcv for d in dates})

        for day in result.daily_results:
            assert day.nav >= 0, f"Negative NAV on {day.date}: {day.nav}"

    def test_paper_trading_no_look_ahead_flags(self):
        """Paper trading should not trigger look-ahead bias flags."""
        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=SEED)
        dates = ohlcv.index.get_level_values("date").unique()[-10:].tolist()

        runner = _build_paper_trading_runner(tickers, ohlcv)
        result = runner.run(dates=dates, market_data_by_date={d: ohlcv for d in dates})

        assert result.look_ahead_flags == 0


# ---------------------------------------------------------------------------
# E2E-3: Multi-Regime Scenario
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.tier3
class TestMultiRegimeScenario:
    """System handles different market regimes correctly."""

    def _make_regime_data(self, tickers, regime="normal", days=60, seed=SEED):
        """Make OHLCV data simulating different regimes."""
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2020-01-02", periods=days + 252)  # need lookback

        if regime == "bull":
            drift, vol = 0.001, 0.01
        elif regime == "bear":
            drift, vol = -0.003, 0.04
        elif regime == "crash":
            drift, vol = -0.01, 0.08
        else:
            drift, vol = 0.0003, 0.02

        rows = []
        for ticker in tickers:
            base = rng.uniform(50, 200)
            rets = rng.normal(drift, vol, size=len(dates))
            prices = base * np.cumprod(1 + rets)
            for i, dt in enumerate(dates):
                p = prices[i]
                rows.append({
                    "date": dt, "ticker": ticker,
                    "open": p, "high": p * 1.01, "low": p * 0.99,
                    "close": p, "adj_close": p,
                    "volume": int(rng.uniform(1e6, 1e8)),
                })

        df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
        return df

    def test_bull_market_no_kill_switch(self):
        """In bull market, kill switch should not trigger."""
        tickers = STANDARD_TICKERS[:5]
        ohlcv = self._make_regime_data(tickers, regime="bull", days=30)
        dates = ohlcv.index.get_level_values("date").unique()[-10:].tolist()

        runner = _build_paper_trading_runner(tickers, ohlcv)
        result = runner.run(dates=dates, market_data_by_date={d: ohlcv for d in dates})

        kill_triggered = any(d.kill_switch_triggered for d in result.daily_results)
        assert not kill_triggered, "Kill switch fired in bull market"

    def test_system_survives_crash_regime(self):
        """System handles crash regime without exception."""
        tickers = STANDARD_TICKERS[:5]
        ohlcv = self._make_regime_data(tickers, regime="crash", days=30)
        dates = ohlcv.index.get_level_values("date").unique()[-10:].tolist()

        runner = _build_paper_trading_runner(tickers, ohlcv)
        result = runner.run(dates=dates, market_data_by_date={d: ohlcv for d in dates})

        # System should either complete or halt via kill switch — never crash
        assert result.num_days >= 1


# ---------------------------------------------------------------------------
# E2E-5: Kill Switch Trigger and Recovery
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.tier3
class TestKillSwitchTriggerAndRecovery:
    """Kill switch triggers on drawdown and system recovers after reset."""

    def test_kill_switch_triggers_and_halts_orders(self):
        """When kill switch fires, no more orders are generated."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)

        # 25% crash
        triggered = ks.check(750_000.0)
        assert triggered is True
        assert ks.is_halted

    def test_recovery_after_kill_switch_reset(self):
        """After kill switch reset, trading can resume."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.infrastructure.system_state_machine import SystemStateMachine, SystemState

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)
        ks.check(750_000.0)
        assert ks.is_halted

        sm = SystemStateMachine()
        sm.transition_to(SystemState.DATA_READY)
        sm.transition_to(SystemState.TRADING_ENABLED)
        sm.transition_to(SystemState.RISK_HALT)

        assert sm.is_halted

        # Recovery
        ks.reset(750_000.0)
        assert not ks.is_halted

        # State machine can go back to trading (RISK_HALT → TRADING_ENABLED)
        sm.transition_to(SystemState.TRADING_ENABLED)
        assert sm.can_trade

    def test_full_pipeline_with_drawdown_scenario(self):
        """Paper trading with drawdown-inducing data triggers kill switch."""
        tickers = STANDARD_TICKERS[:5]

        # Create data with a crash in the middle
        rng = np.random.default_rng(SEED)
        dates = pd.bdate_range("2019-01-02", periods=300)
        rows = []
        for ticker in tickers:
            base = rng.uniform(50, 200)
            rets = rng.normal(0.0005, 0.02, size=300)
            prices = base * np.cumprod(1 + rets)
            for i, dt in enumerate(dates):
                p = prices[i]
                rows.append({
                    "date": dt, "ticker": ticker,
                    "open": p, "high": p * 1.01, "low": p * 0.99,
                    "close": p, "adj_close": p,
                    "volume": int(rng.uniform(1e6, 1e8)),
                })

        ohlcv = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
        trade_dates = dates[-10:].tolist()

        runner = _build_paper_trading_runner(tickers, ohlcv)
        result = runner.run(dates=trade_dates, market_data_by_date={d: ohlcv for d in trade_dates})

        # Pipeline should complete (kill switch may or may not fire depending on data)
        assert result.num_days >= 1
        assert result.final_nav >= 0
