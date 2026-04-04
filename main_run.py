#!/usr/bin/env python3
"""main_run.py — Click-to-run paper trading with live prices and dashboard.

Launch this script in VSCode (or any terminal) and leave it running.
It will:
  1. Fetch live prices from Yahoo Finance every 60 seconds
  2. Feed them into a SimulationBroker for paper-traded execution
  3. Run the full TradingEngine loop: research → risk → orders → fills
  4. Print a live dashboard to the console every tick

Usage:
    python main_run.py                    # defaults (10 tickers, $1M)
    python main_run.py --tickers AAPL MSFT GOOG --cash 500000
    python main_run.py --research-interval 1800  # research every 30 min

To stop: Ctrl+C (graceful shutdown with state snapshot).
"""

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ──────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quant_fund.alpha_discovery.signal_ranking_engine import SignalRankingEngine
from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.data_layer.connectors.yfinance_connector import YFinanceConnector
from quant_fund.data_layer.live_data_stream_adapter import (
    FeedProvider,
    LiveDataStreamAdapter,
)
from quant_fund.execution.order_management.market_hours_enforcer import (
    MarketHoursEnforcer,
)
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.main.trading_engine import TradingEngine
from quant_fund.monitoring.alerting_system import AlertingSystem
from quant_fund.monitoring.execution_quality_monitor import ExecutionQualityMonitor
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.monitoring.system_health_monitor import SystemHealthMonitor
from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer
from quant_fund.feature_factory.technical_indicator_engine import (
    TechnicalIndicatorEngine,
)
from quant_fund.main.research_runner import ResearchRunner
from quant_fund.portfolio.capacity_model.market_impact_model import MarketImpactModel
from quant_fund.portfolio.portfolio_construction.constraint_engine import (
    ConstraintEngine,
)
from quant_fund.portfolio.portfolio_construction.portfolio_optimizer import (
    PortfolioOptimizer,
)
from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor
from quant_fund.risk_engine.leverage_controller import LeverageController
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from quant_fund.risk_engine.risk_cascade_coordinator import RiskCascadeCoordinator

logger = logging.getLogger("main_run")

# ── Default universe ────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "BAC", "WMT",
]

DEFAULT_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
    "AMZN": "Consumer Discretionary", "META": "Technology",
    "TSLA": "Consumer Discretionary", "NVDA": "Technology",
    "JPM": "Financials", "BAC": "Financials", "WMT": "Consumer Staples",
}


# ── YFinance FeedProvider adapter ───────────────────────────────────

class YFinanceFeedProvider(FeedProvider):
    """Wraps YFinanceConnector to implement the FeedProvider interface.

    Yahoo Finance data has a ~15-minute delay which is acceptable
    for paper trading and strategy development.
    """

    def __init__(self, config: Optional[dict] = None):
        self._connector = YFinanceConnector(config=config)
        self._connected = False
        self._consecutive_failures = 0
        self._failure_warning_threshold = 3
        self._failure_critical_threshold = 10

    def connect(self) -> None:
        self._connected = True
        logger.info("YFinanceFeedProvider connected")

    def disconnect(self) -> None:
        self._connected = False
        logger.info("YFinanceFeedProvider disconnected")

    def fetch_latest_bars(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch latest bars from Yahoo Finance."""
        if not tickers:
            return pd.DataFrame()
        try:
            df = self._connector.fetch_latest(tickers)
            if df.empty:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_critical_threshold:
                    logger.critical(
                        "YFinance returned empty data %d consecutive times — "
                        "possible rate limit or outage",
                        self._consecutive_failures,
                    )
                elif self._consecutive_failures >= self._failure_warning_threshold:
                    logger.warning(
                        "YFinance returned empty data %d consecutive times",
                        self._consecutive_failures,
                    )
            else:
                if self._consecutive_failures > 0:
                    logger.info(
                        "YFinance recovered after %d empty responses",
                        self._consecutive_failures,
                    )
                self._consecutive_failures = 0
            return df
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_critical_threshold:
                logger.critical(
                    "YFinance fetch failed %d consecutive times — "
                    "possible rate limit or network outage",
                    self._consecutive_failures,
                )
            else:
                logger.warning("YFinance fetch_latest_bars failed", exc_info=True)
            return pd.DataFrame()

    def fetch_snapshot(
        self, tickers: List[str], fields: List[str]
    ) -> pd.DataFrame:
        """Fetch snapshot — delegates to fetch_latest_bars."""
        return self.fetch_latest_bars(tickers)


def bars_to_market_data(bars: pd.DataFrame) -> Dict[str, dict]:
    """Convert a (date, ticker) MultiIndex OHLCV DataFrame to broker market data format.

    Returns dict: ticker -> {bid, ask, mid, last, volume, adv}.
    """
    result = {}
    if bars.empty:
        return result

    # Handle MultiIndex (date, ticker)
    if isinstance(bars.index, pd.MultiIndex):
        # Get the latest date's data
        dates = bars.index.get_level_values(0)
        latest_date = dates.max()
        bars = bars.loc[latest_date]

    for ticker in bars.index:
        row = bars.loc[ticker]
        close = float(row.get("close", row.get("adj_close", 0)))
        if close <= 0:
            continue
        volume = float(row.get("volume", 1_000_000))
        spread = close * 0.001  # 10 bps spread estimate
        result[ticker] = {
            "bid": close - spread / 2,
            "ask": close + spread / 2,
            "mid": close,
            "last": close,
            "volume": volume,
            "adv": volume,  # single-day estimate; good enough for paper trading
        }
    return result


# ── Console dashboard ───────────────────────────────────────────────

def print_dashboard(
    engine: TradingEngine,
    broker: SimulationBroker,
    dashboard: PnLDashboard,
    health: SystemHealthMonitor,
    tick_count: int,
):
    """Print a live status dashboard to the console."""
    os.system("clear" if os.name != "nt" else "cls")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nav = broker.get_account_value()
    positions = broker.get_positions()

    print("=" * 72)
    print(f"  QUANTFUND V8 — PAPER TRADING DASHBOARD")
    print(f"  {now}    tick #{tick_count}")
    print("=" * 72)

    # Engine state
    print(f"\n  Engine State:    {engine.state.value}")
    print(f"  NAV:             ${nav:,.2f}")

    # PnL summary
    summary = dashboard.get_summary()
    if summary:
        total_ret = summary.get("total_return", 0)
        sharpe = summary.get("sharpe_ratio", 0)
        max_dd = summary.get("max_drawdown", 0)
        ann_vol = summary.get("annualised_vol", 0)
        print(f"  Total Return:    {total_ret:+.4%}")
        print(f"  Sharpe Ratio:    {sharpe:.2f}")
        print(f"  Max Drawdown:    {max_dd:.4%}")
        print(f"  Annualized Vol:  {ann_vol:.4%}")

    # Positions
    print(f"\n  {'POSITIONS':=<52}")
    if positions.empty:
        print("  (no positions)")
    else:
        print(f"  {'Ticker':<10} {'Shares':>10} {'Value':>14}")
        print(f"  {'-'*10} {'-'*10} {'-'*14}")
        md = broker.get_market_data(list(positions.index))
        for ticker in sorted(positions.index):
            qty = positions[ticker]
            if qty == 0:
                continue
            price = md.loc[ticker, "mid"] if ticker in md.index else 0
            value = qty * price
            print(f"  {ticker:<10} {qty:>10,.0f} ${value:>13,.2f}")
        print(f"  {'':10} {'':>10} {'─'*14}")
        total_pos = sum(
            positions[t] * (md.loc[t, "mid"] if t in md.index else 0)
            for t in positions.index
        )
        print(f"  {'Total':<10} {'':>10} ${total_pos:>13,.2f}")

    # Execution quality
    eq_summary = engine.get_execution_quality_summary()
    if eq_summary and eq_summary.num_orders > 0:
        print(f"\n  {'EXECUTION QUALITY':=<52}")
        print(f"  Orders:          {eq_summary.num_orders}")
        print(f"  Avg Fill Rate:   {eq_summary.avg_fill_rate:.2%}")
        print(f"  Avg IS (bps):    {eq_summary.avg_implementation_shortfall_bps:.1f}")
        print(f"  Adverse Flags:   {eq_summary.total_adverse_selection_flags}")

    # Health
    if health.latest is not None:
        snap = health.latest
        print(f"\n  {'SYSTEM HEALTH':=<52}")
        print(f"  Status:          {snap.overall_status}")
        print(f"  Uptime:          {snap.uptime_s / 60:.1f} min")
        print(f"  Data Feed Lag:   {snap.data_feed_lag_s:.1f}s")

    # Target weights
    if engine._target_weights is not None and not engine._target_weights.empty:
        print(f"\n  {'TARGET WEIGHTS':=<52}")
        for ticker, w in engine._target_weights.sort_values(ascending=False).items():
            if abs(w) > 0.001:
                print(f"  {ticker:<10} {w:>+8.4f}")

    print(f"\n{'=' * 72}")
    print("  Press Ctrl+C to stop.")


# ── Main entry point ────────────────────────────────────────────────

def build_engine(
    tickers: List[str],
    initial_cash: float = 1_000_000.0,
    research_interval_s: float = 3600.0,
    price_poll_interval_s: float = 60.0,
    db_path: str = "quantfund_paper.db",
):
    """Construct a fully wired TradingEngine for paper trading.

    Returns (engine, broker, dashboard, health, live_adapter).
    """
    # Broker
    broker = SimulationBroker(config={
        "initial_cash": initial_cash,
        "half_spread_bps": 5.0,
        "market_impact_bps": 2.0,
    })

    # Live data feed
    feed_provider = YFinanceFeedProvider()
    live_adapter = LiveDataStreamAdapter(
        config={"polling_interval_sec": price_poll_interval_s, "buffer_size": 1000},
        feed_provider=feed_provider,
    )

    # Monitoring
    dashboard = PnLDashboard(config={"initial_nav": initial_cash})
    health = SystemHealthMonitor(config={"max_data_lag_s": price_poll_interval_s * 5})
    alerting = AlertingSystem()
    eq_monitor = ExecutionQualityMonitor()

    # Risk
    kill_switch = KillSwitch(config={
        "drawdown_limit": 0.15,
        "initial_nav": initial_cash,
    })
    drawdown_monitor = DrawdownMonitor(config={
        "initial_nav": initial_cash,
        "drawdown_warning": 0.05,
        "drawdown_alert": 0.10,
        "drawdown_limit": 0.15,
    })
    leverage_controller = LeverageController(config={"max_leverage": 1.5})
    capacity_model = MarketImpactModel()

    risk_cascade = RiskCascadeCoordinator(
        kill_switch=kill_switch,
        drawdown_monitor=drawdown_monitor,
        leverage_controller=leverage_controller,
    )

    # Market hours
    market_hours = MarketHoursEnforcer()

    # Research pipeline — generates real alpha-driven weights
    tech_engine = TechnicalIndicatorEngine(config={
        "technical_indicators": {
            "return_windows": [5, 20, 60],
            "volatility_windows": [20],
            "rsi_window": 14,
            "bollinger_window": 20,
            "relative_volume_window": 20,
        }
    })
    research_runner = ResearchRunner(config={
        "universe": tickers,
        "lookback_days": 252,
    })
    research_runner.inject_components(
        feature_generators=tech_engine.generators,
        feature_normalizer=FeatureNormalizer(),
        signal_ranking=SignalRankingEngine(config={
            "min_ic": 0.0,       # relax IC filter for live trading
            "min_ic_tstat": 0.0,  # relax t-stat filter
            "min_eval_days": 0,   # no minimum eval period
        }),
    )

    # Portfolio optimizer and constraints
    portfolio_optimizer = PortfolioOptimizer(config={"risk_aversion": 1.0})
    constraint_engine = ConstraintEngine(config={
        "position_limits": {
            "max_position_size": 0.10,     # 10% max per stock (paper trading)
            "max_sector_exposure": 0.40,   # 40% per sector (relaxed for 10 tickers)
            "max_leverage": 1.0,           # long-only for paper trading
        }
    })

    # Engine
    engine = TradingEngine(config={
        "tick_interval_s": 1.0,
        "convergence_interval_s": price_poll_interval_s,
        "research_interval_s": research_interval_s,
        "snapshot_interval_s": 300.0,
        "reconciliation_interval_s": 600.0,
        "health_check_interval_s": 60.0,
        "state_db_path": db_path,
    })

    # Inject all components
    engine.inject_components(
        broker=broker,
        order_generator=OrderGenerator(),
        order_router=OrderRouter(broker),
        kill_switch=kill_switch,
        drawdown_monitor=drawdown_monitor,
        leverage_controller=leverage_controller,
        pnl_dashboard=dashboard,
        alerting=alerting,
        health_monitor=health,
        live_data_adapter=live_adapter,
        risk_cascade=risk_cascade,
        execution_quality_monitor=eq_monitor,
        capacity_model=capacity_model,
        market_hours_enforcer=market_hours,
        sector_map=DEFAULT_SECTORS,
        research_runner=research_runner,
        portfolio_optimizer=portfolio_optimizer,
        constraint_engine=constraint_engine,
    )

    # Subscribe to tickers
    live_adapter.subscribe(tickers)

    # Register a bar callback to update broker market data
    def on_new_bars(bars: pd.DataFrame):
        md = bars_to_market_data(bars)
        if md:
            broker.set_market_data(md)
            nav = broker.get_account_value()
            dashboard.update(nav)
            health.record_data_timestamp("yfinance", pd.Timestamp.now())
            logger.debug("Market data updated: %d tickers", len(md))
        else:
            logger.warning("Received empty market data — prices may be stale")

    live_adapter.on_bar(on_new_bars)

    return engine, broker, dashboard, health, live_adapter


def run_paper_trading(
    tickers: List[str],
    initial_cash: float = 1_000_000.0,
    research_interval_s: float = 3600.0,
    price_poll_interval_s: float = 60.0,
    db_path: str = "quantfund_paper.db",
    skip_market_hours: bool = False,
):
    """Run the paper trading engine with a live console dashboard.

    This function blocks until Ctrl+C is pressed.
    """
    engine, broker, dashboard, health, live_adapter = build_engine(
        tickers=tickers,
        initial_cash=initial_cash,
        research_interval_s=research_interval_s,
        price_poll_interval_s=price_poll_interval_s,
        db_path=db_path,
    )

    if skip_market_hours:
        engine._market_hours_enforcer = None
        logger.info("Market hours enforcement DISABLED (--skip-market-hours)")

    # Seed initial market data and historical data for research
    logger.info("Fetching initial market data for %d tickers...", len(tickers))
    connector = YFinanceConnector()
    try:
        bars = connector.fetch_latest(tickers)
        md = bars_to_market_data(bars)
        if md:
            broker.set_market_data(md)
            dashboard.update(broker.get_account_value())
            logger.info(
                "Initial market data loaded: %d tickers, NAV=$%,.2f",
                len(md), broker.get_account_value(),
            )
        else:
            logger.warning(
                "Could not fetch initial market data. "
                "The engine will start with no prices and update on next poll."
            )
    except Exception:
        logger.warning("Initial market data fetch failed", exc_info=True)

    # Fetch 1 year of historical OHLCV for feature computation
    logger.info("Fetching 1 year of historical data for research pipeline...")
    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365)
        historical = connector.fetch_historical(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
        if not historical.empty:
            engine.inject_components(historical_data=historical)
            logger.info(
                "Historical data loaded: %d rows for research features",
                len(historical),
            )
        else:
            logger.warning("Historical data fetch returned empty — research will use fallback weights")
    except Exception:
        logger.warning("Historical data fetch failed — research will use fallback weights", exc_info=True)

    # Seed equal-weight targets as a starting point. The research pipeline
    # will overwrite these with alpha-driven weights on the first cycle.
    if engine._target_weights is None:
        n = len(tickers)
        if n > 0:
            equal_weight = 0.8 / n  # 80% invested, 20% cash buffer
            weights = pd.Series({t: equal_weight for t in tickers})
            engine.update_target_weights(weights)
            logger.info(
                "Seeded equal-weight targets (%.2f%% per ticker). "
                "Research pipeline will update on first cycle.",
                equal_weight * 100,
            )

    # Install signal handlers in the main thread (they fail from background threads)
    import threading

    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — requesting graceful shutdown", sig_name)
        engine.request_shutdown()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Run the engine in a background thread so we can print the dashboard
    tick_count = [0]
    engine_thread = threading.Thread(target=engine.run, daemon=True)
    engine_thread.start()
    logger.info("TradingEngine started in background thread")

    # Dashboard loop — signal handler above triggers engine.request_shutdown()
    try:
        while engine_thread.is_alive():
            tick_count[0] += 1
            try:
                print_dashboard(engine, broker, dashboard, health, tick_count[0])
            except Exception:
                logger.warning("Dashboard render error", exc_info=True)
            time.sleep(max(price_poll_interval_s, 10.0))
    except KeyboardInterrupt:
        logger.info("Shutdown requested via Ctrl+C")
    finally:
        engine.request_shutdown()
        engine_thread.join(timeout=30)
        logger.info("Paper trading session ended.")

    # Print final summary
    print("\n" + "=" * 72)
    print("  SESSION SUMMARY")
    print("=" * 72)
    nav = broker.get_account_value()
    summary = dashboard.get_summary()
    print(f"  Final NAV:       ${nav:,.2f}")
    if summary:
        print(f"  Total Return:    {summary.get('total_return', 0):+.4%}")
        print(f"  Max Drawdown:    {summary.get('max_drawdown', 0):.4%}")
    positions = broker.get_positions()
    print(f"  Open Positions:  {sum(1 for v in positions if v != 0)}")
    eq = engine.get_execution_quality_summary()
    if eq:
        print(f"  Total Orders:    {eq.num_orders}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="QuantFund V8 — Paper Trading with Live Prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="Ticker symbols to trade (default: 10 large-cap US equities)",
    )
    parser.add_argument(
        "--cash", type=float, default=1_000_000.0,
        help="Initial cash for paper trading (default: $1,000,000)",
    )
    parser.add_argument(
        "--research-interval", type=float, default=3600.0,
        help="Seconds between research cycles (default: 3600 = 1 hour)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=60.0,
        help="Seconds between price updates from Yahoo Finance (default: 60)",
    )
    parser.add_argument(
        "--db-path", type=str, default="quantfund_paper.db",
        help="SQLite database path for state persistence (default: quantfund_paper.db)",
    )
    parser.add_argument(
        "--skip-market-hours", action="store_true",
        help="Disable market hours enforcement (trade anytime)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging — console + rotating file
    log_level = getattr(logging, args.log_level)
    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_fmt,
        datefmt=log_datefmt,
    )

    # Add rotating file handler (100 MB per file, keep 10 = ~1 GB total)
    log_file = os.path.splitext(args.db_path)[0] + ".log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=100 * 1024 * 1024, backupCount=10,
    )
    file_handler.setFormatter(logging.Formatter(log_fmt, datefmt=log_datefmt))
    file_handler.setLevel(log_level)
    logging.getLogger().addHandler(file_handler)

    print("=" * 72)
    print("  QUANTFUND V8 — PAPER TRADING")
    print("=" * 72)
    print(f"  Tickers:         {', '.join(args.tickers)}")
    print(f"  Initial Cash:    ${args.cash:,.2f}")
    print(f"  Research Every:  {args.research_interval:.0f}s")
    print(f"  Price Poll:      {args.poll_interval:.0f}s")
    print(f"  Market Hours:    {'DISABLED' if args.skip_market_hours else 'ENFORCED'}")
    print(f"  State DB:        {args.db_path}")
    print("=" * 72)
    print()

    run_paper_trading(
        tickers=args.tickers,
        initial_cash=args.cash,
        research_interval_s=args.research_interval,
        price_poll_interval_s=args.poll_interval,
        db_path=args.db_path,
        skip_market_hours=args.skip_market_hours,
    )


if __name__ == "__main__":
    main()
