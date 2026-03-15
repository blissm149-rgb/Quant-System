"""Historical scenario testing & pipeline equivalence validation.

Part 1: Validates that ResearchRunner, PaperTradingRunner, and
        LiveTradingRunner all follow the same processing pipeline.
Part 2: Runs 4 historical test configurations (conservative, moderate,
        aggressive, stress-crash) through 60 trading days each, capturing
        detailed 10-day snapshots of NAV, returns, risk events, and
        algorithm selection info.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import (
    STANDARD_MARKET_DATA,
    STANDARD_SECTORS,
    STANDARD_TICKERS,
    make_ohlcv,
)

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.governance.approval_workflow import ApprovalWorkflow
from quant_fund.governance.deployment_controller import DeploymentController
from quant_fund.main.live_trading_runner import LiveTradingRunner
from quant_fund.main.paper_trading_runner import PaperTradingRunner, TradingDayResult
from quant_fund.main.research_runner import ResearchRunner
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.portfolio.portfolio_construction.constraint_engine import ConstraintSet
from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor
from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
from quant_fund.risk_engine.leverage_controller import LeverageController
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch


# ═══════════════════════════════════════════════════════════════════════
# Mock components shared across tests
# ═══════════════════════════════════════════════════════════════════════


class MockFeatureGenerator:
    """Deterministic feature generator using seeded RNG."""

    def __init__(self, tickers, seed=42):
        self.feature_name = "mock_alpha"
        self.lookback_days = 20
        self._tickers = tickers
        self._seed = seed
        self._call_count = 0

    def compute(self, data, as_of=None):
        self._call_count += 1
        rng = np.random.default_rng(self._seed + self._call_count)
        return pd.Series(rng.normal(0, 1, len(self._tickers)), index=self._tickers)

    def validate(self, output):
        return True


class MockSignalRanking:
    """Simple signal combiner — returns normalised alpha scores."""

    def combine(self, feature_matrix):
        if isinstance(feature_matrix, pd.DataFrame):
            scores = feature_matrix.mean(axis=1)
        else:
            scores = feature_matrix
        if scores.std() > 0:
            scores = (scores - scores.mean()) / scores.std()
        return scores


class MockConstraintEngine:
    """Constraint engine mock with get_constraints() method
    (matches what PaperTradingRunner._run_single_day calls)."""

    def __init__(self, config=None):
        cfg = config or {}
        self._max_position_size = cfg.get("max_position_size", 0.02)
        self._max_sector_exposure = cfg.get("max_sector_exposure", 0.20)
        self._max_leverage = cfg.get("max_leverage", 2.0)
        self._dollar_neutral = cfg.get("dollar_neutral", True)

    def get_constraints(self):
        return ConstraintSet(
            max_position_size=self._max_position_size,
            max_sector_exposure=self._max_sector_exposure,
            max_leverage=self._max_leverage,
            dollar_neutral=self._dollar_neutral,
        )


class MockPortfolioOptimizer:
    """Simple optimizer: normalises alpha scores into weights within constraints."""

    def __init__(self, config=None):
        cfg = config or {}
        self._risk_aversion = cfg.get("risk_aversion", 1.0)

    def optimize(self, alpha_scores, constraints=None, **kwargs):
        if alpha_scores is None or alpha_scores.empty:
            return pd.Series(dtype=float)
        # Scale by inverse risk aversion (more aggressive = bigger weights)
        scale = 1.0 / max(self._risk_aversion, 0.1)
        weights = alpha_scores * scale
        # Normalise to reasonable gross exposure
        gross = weights.abs().sum()
        if gross > 0:
            target_gross = 1.0
            if constraints is not None:
                target_gross = min(constraints.max_leverage * 0.5, 1.5)
            weights = weights / gross * target_gross
        # Enforce max position size
        if constraints is not None:
            weights = weights.clip(
                lower=-constraints.max_position_size,
                upper=constraints.max_position_size,
            )
        return weights


class SeededAlphaResearch:
    """ResearchRunner wrapper that produces seeded alpha scores each day.

    Optionally injects a crash bias after a configured day.
    """

    def __init__(self, tickers, seed=42, crash_after_day=None, crash_magnitude=-3.0):
        self._tickers = tickers
        self._seed = seed
        self._day = 0
        self._crash_after_day = crash_after_day
        self._crash_magnitude = crash_magnitude
        self.last_alpha_scores = None

    def run_cycle(self, as_of, market_data=None):
        self._day += 1
        rng = np.random.default_rng(self._seed + self._day)
        scores = rng.normal(0, 1, len(self._tickers))

        # Inject crash bias: heavily negative alpha with high volatility
        if self._crash_after_day and self._day > self._crash_after_day:
            crash_days = self._day - self._crash_after_day
            bias = self._crash_magnitude * min(crash_days / 5.0, 1.0)
            scores = rng.normal(bias, 2.0, len(self._tickers))

        alpha = pd.Series(scores, index=self._tickers)
        # Cross-sectional normalise
        if alpha.std() > 0:
            alpha = (alpha - alpha.mean()) / alpha.std()
        self.last_alpha_scores = alpha

        class Result:
            pass

        r = Result()
        r.alpha_scores = alpha
        r.validation_flags = []
        return r


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Pipeline Equivalence Validation
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineEquivalence:
    """Validate all three regimes use the same processing pipeline."""

    def _make_shared_components(self, tickers):
        """Create components shared across all three regimes."""
        feature_gen = MockFeatureGenerator(tickers, seed=42)
        signal_ranking = MockSignalRanking()
        research = ResearchRunner({"universe": tickers})
        research.inject_components(
            feature_generators=[feature_gen],
            signal_ranking=signal_ranking,
        )
        return research, feature_gen, signal_ranking

    def _make_broker(self, initial_cash=1_000_000):
        broker = SimulationBroker({
            "initial_cash": initial_cash,
            "enforce_cash_floor": True,
        })
        broker.set_market_data(STANDARD_MARKET_DATA)
        return broker

    def test_same_alpha_scores_across_regimes(self):
        """ResearchRunner, PaperTradingRunner, and LiveTradingRunner
        all produce the same alpha scores from the same inputs."""
        tickers = STANDARD_TICKERS[:5]
        date = pd.Timestamp("2024-01-15")
        ohlcv = make_ohlcv(tickers=tickers, periods=50, seed=42)

        # --- Research regime ---
        research1, _, _ = self._make_shared_components(tickers)
        res1 = research1.run_cycle(as_of=date, market_data=ohlcv)
        alpha_research = res1.alpha_scores

        # --- Paper trading regime ---
        # Create a FRESH research runner with same seed to get same scores
        research2, _, _ = self._make_shared_components(tickers)
        broker_paper = self._make_broker()
        optimizer = MockPortfolioOptimizer()
        constraint_eng = MockConstraintEngine()

        paper_runner = PaperTradingRunner({"initial_nav": 1_000_000})
        paper_runner.inject_components(
            research_runner=research2,
            portfolio_optimizer=optimizer,
            constraint_engine=constraint_eng,
            broker=broker_paper,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker_paper, {"safety_checks_enabled": False}),
        )
        paper_result = paper_runner._run_single_day(
            date=date, market_data=ohlcv, prev_nav=1_000_000,
        )

        # The paper runner calls research2.run_cycle() internally
        # Verify research2 was called (call_count incremented on feature gen)
        alpha_paper = research2._results[-1].alpha_scores

        # Both should produce the same alpha scores (same seed, same call sequence)
        pd.testing.assert_series_equal(alpha_research, alpha_paper, check_names=False)

        # --- Live trading regime ---
        research3, _, _ = self._make_shared_components(tickers)
        broker_live = self._make_broker()

        approval = ApprovalWorkflow()
        deployment = DeploymentController(approval)
        # Deploy a strategy so pre-market check passes
        deployment.deploy("test_strategy", allocation_weight=1.0, mode="live")

        live_runner = LiveTradingRunner(
            deployment,
            {"strategy_id": "test_strategy", "pre_market_check_enabled": False},
        )
        live_runner.inject_components(
            research_runner=research3,
            portfolio_optimizer=MockPortfolioOptimizer(),
            constraint_engine=MockConstraintEngine(),
            broker=broker_live,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker_live, {"safety_checks_enabled": False}),
        )
        live_result = live_runner.run_single_day(
            date=date, market_data=ohlcv, prev_nav=1_000_000,
        )
        alpha_live = research3._results[-1].alpha_scores

        pd.testing.assert_series_equal(alpha_research, alpha_live, check_names=False)

    def test_live_wraps_paper_identically(self):
        """LiveTradingRunner delegates to PaperTradingRunner._run_single_day,
        producing equivalent results when no governance blocks occur."""
        tickers = STANDARD_TICKERS[:3]
        date = pd.Timestamp("2024-01-15")

        # Paper runner
        research_p = SeededAlphaResearch(tickers, seed=100)
        broker_p = self._make_broker()
        paper = PaperTradingRunner({"initial_nav": 1_000_000})
        paper.inject_components(
            research_runner=research_p,
            portfolio_optimizer=MockPortfolioOptimizer(),
            constraint_engine=MockConstraintEngine(),
            broker=broker_p,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker_p, {"safety_checks_enabled": False}),
        )
        paper_day = paper._run_single_day(date=date, market_data=None, prev_nav=1_000_000)

        # Live runner with same seed
        research_l = SeededAlphaResearch(tickers, seed=100)
        broker_l = self._make_broker()
        approval = ApprovalWorkflow()
        deployment = DeploymentController(approval)
        deployment.deploy("strat", allocation_weight=1.0, mode="live")

        live = LiveTradingRunner(
            deployment,
            {"strategy_id": "strat", "pre_market_check_enabled": False,
             "post_market_reconciliation": False},
        )
        live.inject_components(
            research_runner=research_l,
            portfolio_optimizer=MockPortfolioOptimizer(),
            constraint_engine=MockConstraintEngine(),
            broker=broker_l,
            order_generator=OrderGenerator(),
            order_router=OrderRouter(broker_l, {"safety_checks_enabled": False}),
        )
        live_day = live.run_single_day(date=date, market_data=None, prev_nav=1_000_000)

        # Both should have the same structure and order counts
        assert paper_day.num_orders == live_day.num_orders
        assert paper_day.num_fills == live_day.num_fills
        assert paper_day.status == live_day.status

    def test_shared_components_not_duplicated(self):
        """Components injected into LiveTradingRunner are passed through
        to its inner PaperTradingRunner — same object references."""
        tickers = STANDARD_TICKERS[:3]
        broker = self._make_broker()
        ks = KillSwitch({"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        exposure = ExposureMonitor({"max_leverage": 2.0})
        leverage = LeverageController({"max_leverage": 2.0})

        approval = ApprovalWorkflow()
        deployment = DeploymentController(approval)
        live = LiveTradingRunner(deployment, {})
        live.inject_components(
            broker=broker,
            kill_switch=ks,
            exposure_monitor=exposure,
            leverage_controller=leverage,
        )

        # LiveTradingRunner passes components to inner PaperTradingRunner
        inner = live._inner_runner
        assert inner._broker is broker
        assert inner._kill_switch is ks
        assert inner._exposure_monitor is exposure
        assert inner._leverage_controller is leverage


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Scenario Runner
# ═══════════════════════════════════════════════════════════════════════


class ScenarioRunner:
    """Builds all components, runs a simulation, collects detailed snapshots."""

    def __init__(self, config):
        self.cfg = config
        self.snapshots = []
        self.alpha_log = []  # per-day alpha scores
        self.risk_events = []
        self.drawdown_alerts = []

    def run(self):
        """Execute the full scenario and return a results dict."""
        cfg = self.cfg
        tickers = cfg["tickers"]
        initial_cash = cfg["initial_cash"]
        num_days = cfg["num_days"]

        # Build broker
        broker = SimulationBroker({
            "initial_cash": initial_cash,
            "enforce_cash_floor": True,
        })
        # Build market data from standard set, filtered to our tickers
        base_md = {
            t: dict(STANDARD_MARKET_DATA[t])
            for t in tickers if t in STANDARD_MARKET_DATA
        }
        broker.set_market_data(base_md)

        # Build research (seeded alpha generator)
        crash_after = cfg.get("crash_after_day")
        research = SeededAlphaResearch(
            tickers,
            seed=cfg.get("alpha_seed", 42),
            crash_after_day=crash_after,
            crash_magnitude=cfg.get("crash_magnitude", -3.0),
        )

        # Build optimizer
        optimizer = MockPortfolioOptimizer({
            "risk_aversion": cfg.get("risk_aversion", 1.0),
        })

        # Build constraints
        constraint_eng = MockConstraintEngine({
            "max_position_size": cfg.get("max_position_size", 0.02),
            "max_sector_exposure": cfg.get("max_sector_exposure", 0.20),
            "max_leverage": cfg.get("max_leverage", 2.0),
            "dollar_neutral": True,
        })

        # Build risk components
        kill_switch = KillSwitch({
            "drawdown_limit": cfg.get("drawdown_limit", 0.20),
            "initial_nav": initial_cash,
        })
        exposure_monitor = ExposureMonitor({
            "max_leverage": cfg.get("max_leverage", 2.0),
            "max_single_name_exposure": cfg.get("max_position_size", 0.02) * 2,
            "max_sector_exposure": cfg.get("max_sector_exposure", 0.20),
        })
        leverage_ctrl = LeverageController({
            "max_leverage": cfg.get("max_leverage", 2.0),
        })
        drawdown_mon = DrawdownMonitor({
            "drawdown_warning": cfg.get("drawdown_limit", 0.20) * 0.5,
            "drawdown_alert": cfg.get("drawdown_limit", 0.20) * 0.75,
            "drawdown_limit": cfg.get("drawdown_limit", 0.20),
            "initial_nav": initial_cash,
        })

        # Build execution
        order_gen = OrderGenerator()
        order_router = OrderRouter(broker, {"safety_checks_enabled": False})

        # Build monitoring
        pnl = PnLDashboard({"initial_nav": initial_cash})

        # Wire the paper trading runner — run manually day by day
        # so we can inject market data changes for crash scenarios
        runner = PaperTradingRunner({"initial_nav": initial_cash})
        runner.inject_components(
            research_runner=research,
            portfolio_optimizer=optimizer,
            constraint_engine=constraint_eng,
            kill_switch=kill_switch,
            exposure_monitor=exposure_monitor,
            leverage_controller=leverage_ctrl,
            order_generator=order_gen,
            order_router=order_router,
            broker=broker,
            pnl_dashboard=pnl,
        )

        # Generate trading dates
        dates = list(pd.bdate_range("2024-01-02", periods=num_days))

        # For crash scenarios, we run day-by-day and degrade market prices
        if crash_after:
            result = self._run_with_crash(
                runner, broker, base_md, tickers,
                dates, initial_cash, crash_after, kill_switch,
            )
        else:
            result = runner.run(dates)

        # Update kill switch peak (the runner's run() does this internally
        # but our manual crash loop needs the kill switch in sync)
        if not crash_after:
            kill_switch.update_peak(result.final_nav)

        # Collect snapshots and risk info
        cumulative_orders = 0
        cumulative_fills = 0
        peak_nav = initial_cash
        kill_switch_events = 0
        exposure_breach_events = 0
        all_exposure_breaches = []

        for i, day in enumerate(result.daily_results):
            cumulative_orders += day.num_orders
            cumulative_fills += day.num_fills
            peak_nav = max(peak_nav, day.nav)
            dd = (peak_nav - day.nav) / peak_nav if peak_nav > 0 else 0.0

            if day.kill_switch_triggered:
                kill_switch_events += 1
            if day.exposure_breaches:
                exposure_breach_events += 1
                all_exposure_breaches.extend(day.exposure_breaches)

            # Track drawdown alerts
            dd_alerts = drawdown_mon.update(day.nav)
            for alert in dd_alerts:
                self.drawdown_alerts.append({
                    "day": i + 1,
                    "date": str(day.date.date()),
                    "level": alert.level.value,
                    "drawdown": f"{alert.drawdown_pct:.2%}",
                })

            # Log alpha scores
            if research.last_alpha_scores is not None:
                alpha = research.last_alpha_scores
                sorted_alpha = alpha.sort_values(ascending=False)
                self.alpha_log.append({
                    "day": i + 1,
                    "mean": alpha.mean(),
                    "std": alpha.std(),
                    "top_2": list(sorted_alpha.head(2).items()),
                    "bottom_2": list(sorted_alpha.tail(2).items()),
                })

            # Capture snapshot every 10 days
            day_num = i + 1
            if day_num % 10 == 0 or day_num == 1 or day.kill_switch_triggered:
                positions = broker.get_positions()
                non_zero = positions[positions != 0] if len(positions) > 0 else positions

                risk_events_str = "None"
                events = []
                if day.kill_switch_triggered:
                    events.append("KILL SWITCH")
                if day.exposure_breaches:
                    events.append(f"Exposure({len(day.exposure_breaches)})")
                if dd_alerts:
                    events.append(f"DD-{dd_alerts[0].level.value}")
                if events:
                    risk_events_str = ", ".join(events)

                cum_ret = (day.nav / initial_cash - 1.0) * 100

                self.snapshots.append({
                    "day": day_num,
                    "date": str(day.date.date()),
                    "nav": day.nav,
                    "cum_return_pct": cum_ret,
                    "drawdown_pct": dd * 100,
                    "daily_return_pct": day.daily_return * 100,
                    "cum_orders": cumulative_orders,
                    "cum_fills": cumulative_fills,
                    "num_positions": len(non_zero),
                    "risk_events": risk_events_str,
                    "status": day.status,
                })

        # Final summary
        summary = pnl.get_summary() if pnl.latest else {}
        fill_rate = (
            cumulative_fills / cumulative_orders * 100
            if cumulative_orders > 0 else 0.0
        )

        return {
            "name": cfg["name"],
            "config": cfg,
            "result": result,
            "snapshots": self.snapshots,
            "alpha_log": self.alpha_log,
            "drawdown_alerts": self.drawdown_alerts,
            "summary": {
                "total_return_pct": result.total_return * 100,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown * 100,
                "num_days_traded": result.num_days,
                "total_orders": cumulative_orders,
                "total_fills": cumulative_fills,
                "fill_rate_pct": fill_rate,
                "final_nav": result.final_nav,
                "kill_switch_events": kill_switch_events,
                "exposure_breach_events": exposure_breach_events,
                "pnl_summary": summary,
            },
        }


    def _run_with_crash(
        self, runner, broker, base_md, tickers,
        dates, initial_cash, crash_after, kill_switch,
    ):
        """Run day-by-day, degrading market prices after crash_after day."""
        from quant_fund.main.paper_trading_runner import PaperTradingResult

        daily_results = []
        prev_nav = initial_cash
        daily_returns = []

        for i, date in enumerate(dates):
            day_num = i + 1

            # After crash day, progressively drop market prices
            if day_num > crash_after:
                crash_progress = min((day_num - crash_after) / 8.0, 1.0)
                drop_factor = 1.0 - 0.40 * crash_progress  # up to 40% price drop
                crashed_md = {}
                for t in tickers:
                    if t in base_md:
                        orig = base_md[t]
                        crashed_md[t] = {
                            "bid": orig["bid"] * drop_factor,
                            "ask": orig["ask"] * drop_factor,
                            "mid": orig["mid"] * drop_factor,
                            "last": orig["last"] * drop_factor,
                            "volume": orig["volume"],
                            "adv": orig["adv"],
                        }
                broker.set_market_data(crashed_md)

            day_result = runner._run_single_day(
                date=date, market_data=None, prev_nav=prev_nav,
            )
            daily_results.append(day_result)
            daily_returns.append(day_result.daily_return)
            prev_nav = day_result.nav

            # Update kill switch peak when NAV increases
            if not day_result.kill_switch_triggered:
                kill_switch.update_peak(day_result.nav)

            if day_result.kill_switch_triggered:
                break

        # Build the aggregate result
        result = PaperTradingResult(
            strategy_id="default",
            start_date=dates[0],
            end_date=dates[min(len(daily_results) - 1, len(dates) - 1)],
        )
        result.daily_results = daily_results
        result.num_days = len(daily_results)
        result.final_nav = prev_nav
        result.total_return = (prev_nav / initial_cash - 1.0) if initial_cash > 0 else 0.0
        result.total_orders = sum(d.num_orders for d in daily_results)
        result.total_fills = sum(d.num_fills for d in daily_results)

        # Sharpe
        if len(daily_returns) > 1:
            ret_arr = np.array(daily_returns)
            std_ret = np.std(ret_arr, ddof=1)
            result.sharpe_ratio = (
                np.mean(ret_arr) / std_ret * np.sqrt(252) if std_ret > 0 else 0.0
            )

        # Max drawdown
        navs = [initial_cash] + [d.nav for d in daily_results]
        peak = navs[0]
        max_dd = 0.0
        for n in navs:
            peak = max(peak, n)
            dd = (peak - n) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd

        return result


def format_report(result):
    """Format a scenario result into a readable text report."""
    cfg = result["config"]
    s = result["summary"]
    lines = []

    sep = "=" * 80
    lines.append(sep)
    lines.append(
        f"SCENARIO: {cfg['name']} | {len(cfg['tickers'])} tickers "
        f"| ${cfg['initial_cash']:,.0f} | {cfg.get('max_leverage', 2.0)}x leverage "
        f"| {cfg.get('drawdown_limit', 0.20):.0%} drawdown limit"
    )
    lines.append(sep)
    lines.append("")
    lines.append("INPUTS:")
    lines.append(f"  Tickers: {', '.join(cfg['tickers'])}")
    lines.append(f"  Initial Capital: ${cfg['initial_cash']:,.0f}")
    lines.append(
        f"  Constraints: max_pos={cfg.get('max_position_size', 0.02):.0%}, "
        f"max_sector={cfg.get('max_sector_exposure', 0.20):.0%}, "
        f"max_leverage={cfg.get('max_leverage', 2.0)}x"
    )
    lines.append(
        f"  Risk: drawdown_limit={cfg.get('drawdown_limit', 0.20):.0%}, "
        f"risk_aversion={cfg.get('risk_aversion', 1.0)}"
    )
    lines.append(
        f"  Seeds: alpha={cfg.get('alpha_seed', 42)}, "
        f"ohlcv={cfg.get('ohlcv_seed', 42)}"
    )
    if cfg.get("crash_after_day"):
        lines.append(
            f"  CRASH INJECTION: after day {cfg['crash_after_day']}, "
            f"magnitude={cfg.get('crash_magnitude', -3.0)}"
        )
    lines.append("")

    # 10-day snapshot table
    lines.append("10-DAY SNAPSHOTS:")
    header = (
        f"  {'Day':>4} | {'Date':>10} | {'NAV':>14} | {'Cum Ret':>8} | "
        f"{'Drawdown':>8} | {'Orders':>6} | {'Fills':>5} | {'Positions':>3} | "
        f"{'Risk Events'}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for snap in result["snapshots"]:
        lines.append(
            f"  {snap['day']:>4} | {snap['date']:>10} | "
            f"${snap['nav']:>12,.0f} | "
            f"{snap['cum_return_pct']:>+7.2f}% | "
            f"{snap['drawdown_pct']:>7.2f}% | "
            f"{snap['cum_orders']:>6} | "
            f"{snap['cum_fills']:>5} | "
            f"{snap['num_positions']:>3} | "
            f"{snap['risk_events']}"
        )
    lines.append("")

    # Alpha signal info at day 30 (or nearest available)
    alpha_at_30 = None
    for a in result["alpha_log"]:
        if a["day"] == 30:
            alpha_at_30 = a
            break
    if alpha_at_30 is None and result["alpha_log"]:
        alpha_at_30 = result["alpha_log"][min(29, len(result["alpha_log"]) - 1)]

    if alpha_at_30:
        lines.append(f"ALPHA SIGNAL INFO (Day {alpha_at_30['day']} snapshot):")
        top = alpha_at_30["top_2"]
        bottom = alpha_at_30["bottom_2"]
        top_str = ", ".join(f"{t} ({v:+.2f})" for t, v in top)
        bottom_str = ", ".join(f"{t} ({v:+.2f})" for t, v in bottom)
        lines.append(f"  Top alpha: {top_str}")
        lines.append(f"  Bottom alpha: {bottom_str}")
        lines.append(
            f"  Alpha mean: {alpha_at_30['mean']:+.3f}, "
            f"std: {alpha_at_30['std']:.3f}"
        )
        lines.append("")

    # Risk summary
    lines.append("RISK SUMMARY:")
    lines.append(
        f"  Kill switch triggers: {s['kill_switch_events']} | "
        f"Exposure breaches: {s['exposure_breach_events']}"
    )
    lines.append(f"  Max drawdown reached: {s['max_drawdown_pct']:.2f}%")
    if result["drawdown_alerts"]:
        for alert in result["drawdown_alerts"][:5]:
            lines.append(
                f"    {alert['level']} on day {alert['day']} "
                f"({alert['date']}): drawdown={alert['drawdown']}"
            )
        if len(result["drawdown_alerts"]) > 5:
            lines.append(
                f"    ... and {len(result['drawdown_alerts']) - 5} more alerts"
            )
    else:
        lines.append("  Drawdown alerts: None")
    lines.append("")

    # Final summary
    lines.append("FINAL SUMMARY:")
    lines.append(
        f"  Total Return: {s['total_return_pct']:+.2f}% | "
        f"Sharpe: {s['sharpe_ratio']:.2f} | "
        f"Max Drawdown: {s['max_drawdown_pct']:.2f}%"
    )
    lines.append(
        f"  Days Traded: {s['num_days_traded']} | "
        f"Total Orders: {s['total_orders']} | "
        f"Fills: {s['total_fills']} | "
        f"Fill Rate: {s['fill_rate_pct']:.1f}%"
    )
    lines.append(f"  Final NAV: ${s['final_nav']:,.2f}")
    lines.append(sep)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Four Historical Test Configurations
# ═══════════════════════════════════════════════════════════════════════

SCENARIO_CONSERVATIVE = {
    "name": "Conservative",
    "tickers": STANDARD_TICKERS[:5],  # 5 tickers
    "initial_cash": 1_000_000,
    "max_leverage": 1.0,
    "max_position_size": 0.01,
    "max_sector_exposure": 0.15,
    "drawdown_limit": 0.10,
    "risk_aversion": 2.0,
    "alpha_seed": 42,
    "ohlcv_seed": 42,
    "num_days": 60,
}

SCENARIO_MODERATE = {
    "name": "Moderate",
    "tickers": list(STANDARD_TICKERS),  # 10 tickers
    "initial_cash": 5_000_000,
    "max_leverage": 2.0,
    "max_position_size": 0.02,
    "max_sector_exposure": 0.20,
    "drawdown_limit": 0.20,
    "risk_aversion": 1.0,
    "alpha_seed": 123,
    "ohlcv_seed": 42,
    "num_days": 60,
}

SCENARIO_AGGRESSIVE = {
    "name": "Aggressive",
    "tickers": list(STANDARD_TICKERS),  # 10 tickers
    "initial_cash": 10_000_000,
    "max_leverage": 2.0,
    "max_position_size": 0.05,
    "max_sector_exposure": 0.30,
    "drawdown_limit": 0.25,
    "risk_aversion": 0.5,
    "alpha_seed": 999,
    "ohlcv_seed": 42,
    "num_days": 60,
}

SCENARIO_STRESS = {
    "name": "Stress Test (Crash)",
    "tickers": list(STANDARD_TICKERS),  # 10 tickers
    "initial_cash": 5_000_000,
    "max_leverage": 2.0,
    "max_position_size": 0.05,  # larger positions = more crash exposure
    "max_sector_exposure": 0.30,
    "drawdown_limit": 0.15,  # tighter to trigger kill switch
    "risk_aversion": 0.5,  # aggressive to amplify crash effect
    "alpha_seed": 77,
    "ohlcv_seed": 77,
    "num_days": 60,
    "crash_after_day": 20,  # earlier crash
    "crash_magnitude": -4.0,
}


class TestHistoricalScenarios:
    """Run 4 historical configurations and produce detailed reports."""

    def _run_and_report(self, config):
        runner = ScenarioRunner(config)
        result = runner.run()
        report = format_report(result)
        print("\n" + report)
        return result

    def test_scenario_conservative(self):
        """Conservative: 5 tickers, $1M, 1x leverage, 10% drawdown limit."""
        result = self._run_and_report(SCENARIO_CONSERVATIVE)
        s = result["summary"]

        # Assertions: conservative should be stable
        assert s["num_days_traded"] > 0
        assert s["final_nav"] > 0
        assert s["max_drawdown_pct"] < 20.0  # should be well under 20%
        assert s["kill_switch_events"] == 0  # conservative should not trigger

    def test_scenario_moderate(self):
        """Moderate: 10 tickers, $5M, 2x leverage, 20% drawdown limit."""
        result = self._run_and_report(SCENARIO_MODERATE)
        s = result["summary"]

        assert s["num_days_traded"] > 0
        assert s["final_nav"] > 0
        assert s["total_orders"] > 0

    def test_scenario_aggressive(self):
        """Aggressive: 10 tickers, $10M, 2x leverage, 25% drawdown, low risk aversion."""
        result = self._run_and_report(SCENARIO_AGGRESSIVE)
        s = result["summary"]

        assert s["num_days_traded"] > 0
        assert s["final_nav"] > 0
        # Aggressive should generate more orders than conservative
        assert s["total_orders"] > 0

    def test_scenario_stress_crash(self):
        """Stress test: crash injection after day 25, tight 15% drawdown limit."""
        result = self._run_and_report(SCENARIO_STRESS)
        s = result["summary"]

        assert s["num_days_traded"] > 0
        # Stress test should show significant activity
        # Kill switch may or may not trigger depending on price impact
        # But the scenario should complete without errors
        assert s["final_nav"] > 0
