"""Market data utilities — conversion helpers for broker data formats.

Extracted from main_run.py to keep the CLI wrapper thin.
"""

from typing import Dict

import pandas as pd


def bars_to_market_data(bars: pd.DataFrame) -> Dict[str, dict]:
    """Convert a (date, ticker) MultiIndex OHLCV DataFrame to broker market data format.

    Returns dict: ticker -> {bid, ask, mid, last, volume, adv}.
    """
    result: Dict[str, dict] = {}
    if bars.empty:
        return result

    # Handle MultiIndex (date, ticker)
    if isinstance(bars.index, pd.MultiIndex):
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
