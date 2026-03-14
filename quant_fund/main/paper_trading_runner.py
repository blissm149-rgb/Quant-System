"""Paper trading runner — simulated trading loop.

Runs the full pipeline in paper mode using the simulation broker:
1. Research cycle (compute features → alpha scores)
2. Portfolio construction (optimise → constraints)
3. Risk checks (kill switch, exposure monitor)
4. Execution (order generator → order router → sim broker)
5. Monitoring (PnL dashboard, alpha monitoring)

This is the primary validation tool before live deployment.
Must run for minimum 6 months (126 trading days) before a
strategy can be promoted to live via the approval workflow.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradingDayResult:
    """Result of a single paper trading day."""

    date: pd.Timestamp
    nav: float
    daily_return: float
    num_orders: int
    num_fills: int
    kill_switch_triggered: bool = False
    exposure_breaches: List[str] = field(default_factory=list)
    look_ahead_flags: List[str] = field(default_factory=list)
    status: str = "completed"
    error_message: str = ""


@dataclass
class PaperTradingResult:
    """Aggregate result of a paper trading run."""

    strategy_id: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    num_days: int = 0
    final_nav: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_orders: int = 0
    total_fills: int = 0
    look_ahead_flags: int = 0
    daily_results: List[TradingDayResult] = field(default_factory=list)


class PaperTradingRunner:
    """Runs the full trading pipeline in paper/simulation mode.

    Uses the simulation broker and processes one day at a time.
    All components are injected — the runner only orchestrates.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._strategy_id = cfg.get("strategy_id", "default")
        self._initial_nav = cfg.get("initial_nav", 1_000_000.0)

        # Injected components
        self._research_runner = None
        self._portfolio_optimizer = None
        self._constraint_engine = None
        self._risk_model = None
        self._kill_switch = None
        self._exposure_monitor = None
        self._leverage_controller = None
        self._order_generator = None
        self._order_router = None
        self._broker = None
        self._pnl_dashboard = None
        self._alerting = None

    def inject_components(self, **components) -> None:
        """Inject pipeline components by name.

        Accepted keys: research_runner, portfolio_optimizer,
        constraint_engine, risk_model, kill_switch, exposure_monitor,
        leverage_controller, order_generator, order_router, broker,
        pnl_dashboard, alerting.
        """
        for name, component in components.items():
            attr = f"_{name}"
            if hasattr(self, attr):
                setattr(self, attr, component)
            else:
                logger.warning("Unknown component: %s", name)

    def run(
        self,
        dates: List[pd.Timestamp],
        market_data_by_date: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None,
    ) -> PaperTradingResult:
        """Run paper trading over a list of trading dates.

        Parameters
        ----------
        dates : list of pd.Timestamp
            Trading dates.
        market_data_by_date : dict, optional
            Mapping date → market data DataFrame for that day.

        Returns
        -------
        PaperTradingResult
        """
        result = PaperTradingResult(
            strategy_id=self._strategy_id,
            start_date=dates[0] if dates else pd.Timestamp.now(),
            end_date=dates[-1] if dates else pd.Timestamp.now(),
        )

        prev_nav = self._initial_nav
        daily_returns = []

        for date in dates:
            day_result = self._run_single_day(
                date=date,
                market_data=(
                    market_data_by_date.get(date)
                    if market_data_by_date else None
                ),
                prev_nav=prev_nav,
            )
            result.daily_results.append(day_result)
            result.total_orders += day_result.num_orders
            result.total_fills += day_result.num_fills
            result.look_ahead_flags += len(day_result.look_ahead_flags)
            daily_returns.append(day_result.daily_return)

            prev_nav = day_result.nav

            if day_result.kill_switch_triggered:
                logger.warning("Kill switch triggered on %s, stopping", date)
                break

        result.num_days = len(result.daily_results)
        result.final_nav = prev_nav
        result.total_return = (
            (prev_nav / self._initial_nav - 1.0)
            if self._initial_nav > 0 else 0.0
        )

        # Compute Sharpe
        if daily_returns and len(daily_returns) > 1:
            ret_arr = np.array(daily_returns)
            mean_ret = np.mean(ret_arr)
            std_ret = np.std(ret_arr, ddof=1)
            result.sharpe_ratio = (
                mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0.0
            )

        # Max drawdown
        if result.daily_results:
            navs = [self._initial_nav] + [d.nav for d in result.daily_results]
            peak = navs[0]
            max_dd = 0.0
            for n in navs:
                peak = max(peak, n)
                dd = (peak - n) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)
            result.max_drawdown = max_dd

        return result

    def _run_single_day(
        self,
        date: pd.Timestamp,
        market_data: Optional[pd.DataFrame],
        prev_nav: float,
    ) -> TradingDayResult:
        """Run a single trading day."""
        day = TradingDayResult(
            date=date, nav=prev_nav, daily_return=0.0,
            num_orders=0, num_fills=0,
        )

        try:
            # 1. Get current NAV
            nav = prev_nav
            if self._broker is not None:
                nav = self._broker.get_account_value()

            # 2. Kill switch check
            if self._kill_switch is not None:
                if self._kill_switch.check(nav):
                    day.kill_switch_triggered = True
                    day.nav = nav
                    day.status = "kill_switch"
                    return day

            # 3. Research: compute alpha scores
            alpha_scores = None
            if self._research_runner is not None:
                research_result = self._research_runner.run_cycle(
                    as_of=date, market_data=market_data,
                )
                alpha_scores = research_result.alpha_scores
                day.look_ahead_flags = research_result.validation_flags

            # 4. Portfolio optimisation
            target_weights = None
            if self._portfolio_optimizer is not None and alpha_scores is not None:
                constraints = None
                if self._constraint_engine is not None:
                    constraints = self._constraint_engine.get_constraints()
                target_weights = self._portfolio_optimizer.optimize(
                    alpha_scores=alpha_scores,
                    constraints=constraints,
                )

            # 5. Risk checks on target weights
            if target_weights is not None:
                if self._exposure_monitor is not None:
                    breaches = self._exposure_monitor.check(target_weights)
                    if breaches:
                        day.exposure_breaches = [str(b) for b in breaches]

                if self._leverage_controller is not None:
                    target_weights = self._leverage_controller.enforce(
                        target_weights
                    )

            # 6. Generate and execute orders
            if target_weights is not None and self._order_generator is not None:
                current_positions = pd.Series(dtype=float)
                prices = pd.Series(dtype=float)
                if self._broker is not None:
                    current_positions = self._broker.get_positions()
                    md = self._broker.get_market_data(
                        list(target_weights.index)
                    )
                    if not md.empty and "mid" in md.columns:
                        prices = md["mid"]

                orders = self._order_generator.generate_orders(
                    target_weights=target_weights,
                    current_positions=current_positions,
                    prices=prices,
                    nav=nav,
                    strategy_id=self._strategy_id,
                )
                day.num_orders = len(orders)

                if self._order_router is not None and orders:
                    acks = self._order_router.route_orders(orders)
                    day.num_fills = sum(
                        1 for a in acks if a.status.value in ("filled", "partial_fill")
                    )

            # 7. Update NAV
            if self._broker is not None:
                nav = self._broker.get_account_value()

            day.nav = nav
            day.daily_return = (nav / prev_nav - 1.0) if prev_nav > 0 else 0.0

            # 8. Update monitoring
            if self._pnl_dashboard is not None:
                self._pnl_dashboard.update(nav, timestamp=date)

        except Exception as e:
            day.status = "failed"
            day.error_message = str(e)
            logger.error("Trading day %s failed: %s", date, e)

        return day
