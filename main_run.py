#!/usr/bin/env python3
"""main_run.py -- Click-to-run paper trading with live prices and dashboard.

Launch this script in VSCode (or any terminal) and leave it running.
It will:
  1. Fetch live prices from Yahoo Finance every 60 seconds
  2. Feed them into a SimulationBroker for paper-traded execution
  3. Run the full TradingEngine loop: research -> risk -> orders -> fills
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
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quant_fund.main.dashboard import run_paper_trading

# Default universe
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


def main():
    parser = argparse.ArgumentParser(
        description="QuantFund V8 -- Paper Trading with Live Prices",
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

    # Configure logging -- console + rotating file
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
    print("  QUANTFUND V8 -- PAPER TRADING")
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
        sector_map=DEFAULT_SECTORS,
    )


if __name__ == "__main__":
    main()
