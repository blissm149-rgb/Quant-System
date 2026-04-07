"""Console dashboard and paper trading session runner.

Extracted from main_run.py to keep the CLI wrapper thin.
"""

import logging
import os
import signal
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.data_layer.connectors.yfinance_connector import YFinanceConnector
from quant_fund.data_layer.market_data_utils import bars_to_market_data
from quant_fund.main.engine_builder import build_engine
from quant_fund.main.trading_engine import TradingEngine
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.monitoring.system_health_monitor import SystemHealthMonitor

logger = logging.getLogger(__name__)


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
    print(f"  QUANTFUND V8 -- PAPER TRADING DASHBOARD")
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
        print(f"  {'':10} {'':>10} {chr(9472)*14}")
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


def run_paper_trading(
    tickers: List[str],
    initial_cash: float = 1_000_000.0,
    research_interval_s: float = 3600.0,
    price_poll_interval_s: float = 60.0,
    db_path: str = "quantfund_paper.db",
    skip_market_hours: bool = False,
    sector_map: Optional[Dict[str, str]] = None,
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

    # Set sector map
    if sector_map:
        engine.inject_components(sector_map=sector_map)

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

            # Compute stock returns and factor returns for factor risk model
            try:
                if isinstance(historical.index, pd.MultiIndex):
                    close_pivot = historical["close"].unstack(level="ticker")
                else:
                    close_pivot = historical[["close"]]
                stock_returns = close_pivot.pct_change().dropna()
                market_return = stock_returns.mean(axis=1)
                factor_returns = pd.DataFrame({"market": market_return})
                engine.inject_components(
                    stock_returns=stock_returns,
                    factor_returns=factor_returns,
                )
                logger.info(
                    "Stock returns and factor returns computed: %d dates",
                    len(stock_returns),
                )
            except Exception:
                logger.warning("Factor returns computation failed", exc_info=True)
        else:
            logger.warning("Historical data fetch returned empty -- research will use fallback weights")
    except Exception:
        logger.warning("Historical data fetch failed -- research will use fallback weights", exc_info=True)

    # Load champion ML models from ModelStore
    research_runner = engine._research_runner
    if research_runner is not None:
        loaded = research_runner.load_champion_models()
        if loaded > 0:
            logger.info("Loaded %d champion ML model(s) from ModelStore", loaded)

    # Seed equal-weight targets as a starting point
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

    # Install signal handlers in the main thread
    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s -- requesting graceful shutdown", sig_name)
        engine.request_shutdown()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Run the engine in a background thread so we can print the dashboard
    tick_count = [0]
    engine_thread = threading.Thread(target=engine.run, daemon=True)
    engine_thread.start()
    logger.info("TradingEngine started in background thread")

    # Dashboard loop
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
