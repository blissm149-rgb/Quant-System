"""Stress scenario generators for fault injection and extreme conditions.

Generates corrupted data, extreme config values, and edge-case inputs
for stress testing the system.
"""

import numpy as np
import pandas as pd

from tests.conftest import make_ohlcv


def inject_nans(df, n_nans=50, columns=None, seed=42):
    """Inject NaN values at random positions in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (will be copied, not modified in place).
    n_nans : int
        Number of NaN values to inject.
    columns : list[str], optional
        Columns to inject into. Defaults to ["close"].
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Copy of df with NaN values injected.
    """
    columns = columns or ["close"]
    rng = np.random.default_rng(seed)
    result = df.copy()

    for _ in range(n_nans):
        idx = rng.integers(len(result))
        col = rng.choice(columns)
        result.iloc[idx, result.columns.get_loc(col)] = np.nan

    return result


def inject_negative_prices(df, n_negatives=10, seed=42):
    """Inject negative prices at random positions."""
    rng = np.random.default_rng(seed)
    result = df.copy()

    for _ in range(n_negatives):
        idx = rng.integers(len(result))
        col = rng.choice(["open", "high", "low", "close"])
        result.iloc[idx, result.columns.get_loc(col)] = -abs(rng.uniform(1, 100))

    return result


def inject_future_timestamps(df, days_forward=3650):
    """Shift all timestamps into the future."""
    result = df.copy()
    result.index = result.index.set_levels(
        [result.index.levels[0] + pd.Timedelta(days=days_forward)],
        level=0,
    )
    return result


def inject_duplicates(df, n_duplicates=20, seed=42):
    """Inject duplicate rows at random positions."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(df), size=n_duplicates)
    dups = df.iloc[indices]
    result = pd.concat([df, dups]).sort_index()
    return result


def make_gap_data(tickers=None, days=100, gap_start=40, gap_length=5, seed=42):
    """Generate OHLCV with a gap (missing bars) in the middle.

    Simulates data feed interruption.
    """
    df = make_ohlcv(tickers=tickers, periods=days, seed=seed)
    dates = df.index.get_level_values("date").unique()
    gap_dates = dates[gap_start:gap_start + gap_length]
    mask = ~df.index.get_level_values("date").isin(gap_dates)
    return df[mask]


def make_stale_data(tickers=None, days=100, stale_start=40, stale_length=5, seed=42):
    """Generate OHLCV with stale (repeated) bars.

    Simulates frozen data feed.
    """
    df = make_ohlcv(tickers=tickers, periods=days, seed=seed)
    dates = df.index.get_level_values("date").unique()
    result = df.copy()

    anchor_date = dates[stale_start]
    for i in range(1, stale_length):
        if stale_start + i >= len(dates):
            break
        target_date = dates[stale_start + i]
        for ticker in (tickers or df.index.get_level_values("ticker").unique()):
            if (anchor_date, ticker) in result.index and (target_date, ticker) in result.index:
                result.loc[(target_date, ticker)] = result.loc[(anchor_date, ticker)]

    return result


def make_extreme_config():
    """Generate config with extreme but technically valid values."""
    return {
        "max_leverage": 10.0,
        "max_position_size": 0.5,
        "max_sector_exposure": 1.0,
        "dollar_neutral": False,
    }


def make_boundary_configs():
    """Generate configs at exact boundary values for testing."""
    return [
        {"max_leverage": 2.0, "max_position_size": 0.02, "max_sector_exposure": 0.20, "dollar_neutral": True},
        {"max_leverage": 0.0, "max_position_size": 0.0, "max_sector_exposure": 0.0, "dollar_neutral": True},
        {"max_leverage": 1.0, "max_position_size": 0.01, "max_sector_exposure": 0.10, "dollar_neutral": True},
    ]


def make_large_universe(n_tickers=1000, days=252, seed=42):
    """Generate OHLCV for a large ticker universe (scalability testing)."""
    tickers = [f"T{i:04d}" for i in range(n_tickers)]
    return make_ohlcv(tickers=tickers, periods=days, seed=seed)
