"""Paper trading runner — simulated trading loop.

.. deprecated::
    This module is superseded by ``TradingEngine`` in
    ``quant_fund/main/trading_engine.py`` with ``main_run.py`` as the CLI
    entry point. Use ``python main_run.py`` for paper trading instead.
    This module will be removed in a future release.

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
import warnings

warnings.warn(
    "PaperTradingRunner is deprecated. Use TradingEngine via main_run.py instead.",
    DeprecationWarning,
    stacklevel=2,
)
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
        self._factor_exposure_estimator = None
        self._factor_covariance_estimator = None
        self._factor_returns = None  # pd.DataFrame (dates x factors)
        self._stock_returns = None  # pd.DataFrame (dates x tickers)
        self._sector_map = None  # Dict[str, str] ticker -> sector
        self._risk_cascade = None
        self._execution_quality_monitor = None

    def inject_components(self, **components) -> None:
        """Inject pipeline components by name.

        Accepted keys: research_runner, portfolio_optimizer,
        constraint_engine, risk_model, kill_switch, exposure_monitor,
        leverage_controller, order_generator, order_router, broker,
        pnl_dashboard, alerting, factor_exposure_estimator,
        factor_covariance_estimator, factor_returns, stock_returns,
        sector_map.
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
            # 0. Update broker with today's market data (price evolution)
            if self._broker is not None and market_data is not None:
                updated_md = {}
                if hasattr(market_data, "iterrows"):
                    for ticker, row in market_data.iterrows():
                        mid = row.get("mid") if "mid" in row.index else row.get("close", 0)
                        if mid is None or mid == 0:
                            continue
                        spread = mid * 0.001  # 10bps default spread
                        updated_md[ticker] = {
                            "bid": mid - spread / 2,
                            "ask": mid + spread / 2,
                            "mid": mid,
                            "last": mid,
                            "volume": row.get("volume", 1_000_000),
                            "adv": row.get("adv", row.get("volume", 1_000_000)),
                        }
                    if updated_md:
                        self._broker.set_market_data(updated_md)

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

            # 4. Factor risk model estimation
            factor_exposures = None
            factor_covariance = None
            if (
                self._factor_exposure_estimator is not None
                and self._factor_covariance_estimator is not None
                and self._factor_returns is not None
                and self._stock_returns is not None
            ):
                factor_exposures = self._factor_exposure_estimator.estimate(
                    returns=self._stock_returns,
                    factor_returns=self._factor_returns,
                    as_of=date,
                    sector_map=self._sector_map,
                )
                factor_covariance = self._factor_covariance_estimator.estimate(
                    factor_returns=self._factor_returns,
                    as_of=date,
                )

            # 5. Portfolio optimisation
            target_weights = None
            if self._portfolio_optimizer is not None and alpha_scores is not None:
                constraints = None
                if self._constraint_engine is not None:
                    constraints = self._constraint_engine.build_constraints(
                        sector_map=self._sector_map,
                    )

                current_positions = pd.Series(dtype=float)
                if self._broker is not None:
                    current_positions = self._broker.get_positions()

                target_weights = self._portfolio_optimizer.optimize(
                    alpha_scores=alpha_scores,
                    factor_covariance=factor_covariance,
                    factor_exposures=factor_exposures,
                    constraints=constraints,
                    current_positions=current_positions,
                )

            # 6. Risk checks on target weights
            if target_weights is not None:
                if self._risk_cascade is not None:
                    cascade_result = self._risk_cascade.run_cascade(
                        nav, target_weights, self._sector_map, factor_exposures,
                    )
                    if cascade_result.orders_blocked:
                        if cascade_result.kill_switch_triggered:
                            day.kill_switch_triggered = True
                            day.status = "kill_switch"
                        else:
                            day.exposure_breaches = [
                                str(b) for b in cascade_result.exposure_breaches
                            ]
                            day.status = "exposure_breach"
                        day.nav = nav
                        target_weights = None
                    else:
                        target_weights = cascade_result.adjusted_weights
                else:
                    # Fallback: inline risk checks
                    if self._leverage_controller is not None:
                        target_weights = self._leverage_controller.enforce(
                            target_weights
                        )

                    if self._exposure_monitor is not None:
                        breaches = self._exposure_monitor.check(
                            target_weights,
                            sector_map=self._sector_map,
                            factor_exposures=factor_exposures,
                        )
                        if breaches:
                            day.exposure_breaches = [str(b) for b in breaches]
                            day.status = "exposure_breach"
                            day.nav = nav
                            logger.warning(
                                "Exposure breach on %s — orders blocked: %s",
                                date, day.exposure_breaches,
                            )
                            target_weights = None  # block order generation

            # 7. Generate and execute orders
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

                    # Record execution quality
                    if self._execution_quality_monitor is not None:
                        order_by_id = {o.order_id: o for o in orders}
                        for ack in acks:
                            if ack.status.value in ("filled", "partial_fill"):
                                order = order_by_id.get(ack.order_id)
                                if order is not None:
                                    decision_price = prices.get(order.ticker, 0.0)
                                    fill_price = decision_price
                                    recent_fills = getattr(self._broker, "_fills", [])
                                    for f in reversed(recent_fills):
                                        if f.order_id == ack.order_id:
                                            fill_price = f.fill_price
                                            break
                                    self._execution_quality_monitor.record_execution(
                                        order_id=ack.order_id,
                                        ticker=order.ticker,
                                        side=order.side.value,
                                        target_qty=order.quantity,
                                        filled_qty=order.quantity,
                                        decision_price=decision_price,
                                        fill_price=fill_price,
                                    )

            # 8. Update NAV
            if self._broker is not None:
                nav = self._broker.get_account_value()

            day.nav = nav
            day.daily_return = (nav / prev_nav - 1.0) if prev_nav > 0 else 0.0

            # 9. Update kill switch peak NAV
            if self._kill_switch is not None:
                self._kill_switch.update_peak(nav)

            # 10. Update monitoring
            if self._pnl_dashboard is not None:
                self._pnl_dashboard.update(nav, timestamp=date)

        except Exception as e:
            day.status = "failed"
            day.error_message = str(e)
            logger.error("Trading day %s failed: %s", date, e)

        return day
