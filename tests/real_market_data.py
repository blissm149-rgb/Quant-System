"""Realistic historical market data based on actual stock prices.

Monthly anchor prices are sourced from real market data for 10 tickers
covering Jan 2020 – Dec 2023 (the period spanning COVID crash, recovery,
2022 bear market, and 2023 rally). Daily prices are interpolated between
anchors with realistic daily volatility per ticker.

Factor returns are based on Fama-French factor characteristics:
Market, SMB (Size), HML (Value), UMD (Momentum), QMJ (Quality).
"""

import numpy as np
import pandas as pd

# ── Monthly close prices (real data, split-adjusted) ──────────────────
# Sources: Yahoo Finance historical data through training cutoff.
# Prices are approximate monthly closes, rounded to nearest dollar.

_MONTHLY_CLOSES = {
    # fmt: off
    "AAPL": {
        "2020-01": 77, "2020-02": 68, "2020-03": 64, "2020-04": 74,
        "2020-05": 79, "2020-06": 91, "2020-07": 106, "2020-08": 129,
        "2020-09": 116, "2020-10": 109, "2020-11": 119, "2020-12": 132,
        "2021-01": 131, "2021-02": 127, "2021-03": 123, "2021-04": 132,
        "2021-05": 125, "2021-06": 137, "2021-07": 146, "2021-08": 152,
        "2021-09": 142, "2021-10": 150, "2021-11": 165, "2021-12": 178,
        "2022-01": 175, "2022-02": 165, "2022-03": 175, "2022-04": 157,
        "2022-05": 149, "2022-06": 137, "2022-07": 162, "2022-08": 157,
        "2022-09": 138, "2022-10": 153, "2022-11": 148, "2022-12": 130,
        "2023-01": 143, "2023-02": 147, "2023-03": 164, "2023-04": 170,
        "2023-05": 177, "2023-06": 193, "2023-07": 196, "2023-08": 188,
        "2023-09": 171, "2023-10": 171, "2023-11": 190, "2023-12": 192,
    },
    "MSFT": {
        "2020-01": 170, "2020-02": 163, "2020-03": 157, "2020-04": 174,
        "2020-05": 183, "2020-06": 204, "2020-07": 206, "2020-08": 226,
        "2020-09": 210, "2020-10": 202, "2020-11": 214, "2020-12": 222,
        "2021-01": 232, "2021-02": 233, "2021-03": 235, "2021-04": 252,
        "2021-05": 250, "2021-06": 271, "2021-07": 287, "2021-08": 302,
        "2021-09": 282, "2021-10": 331, "2021-11": 330, "2021-12": 336,
        "2022-01": 310, "2022-02": 298, "2022-03": 308, "2022-04": 277,
        "2022-05": 272, "2022-06": 257, "2022-07": 277, "2022-08": 264,
        "2022-09": 233, "2022-10": 233, "2022-11": 255, "2022-12": 240,
        "2023-01": 248, "2023-02": 252, "2023-03": 289, "2023-04": 308,
        "2023-05": 332, "2023-06": 340, "2023-07": 338, "2023-08": 328,
        "2023-09": 316, "2023-10": 339, "2023-11": 378, "2023-12": 376,
    },
    "GOOG": {
        "2020-01": 72, "2020-02": 66, "2020-03": 58, "2020-04": 67,
        "2020-05": 71, "2020-06": 71, "2020-07": 75, "2020-08": 84,
        "2020-09": 74, "2020-10": 81, "2020-11": 88, "2020-12": 88,
        "2021-01": 92, "2021-02": 104, "2021-03": 103, "2021-04": 116,
        "2021-05": 118, "2021-06": 122, "2021-07": 137, "2021-08": 146,
        "2021-09": 134, "2021-10": 149, "2021-11": 149, "2021-12": 145,
        "2022-01": 136, "2022-02": 136, "2022-03": 140, "2022-04": 111,
        "2022-05": 112, "2022-06": 109, "2022-07": 114, "2022-08": 109,
        "2022-09": 96, "2022-10": 94, "2022-11": 101, "2022-12": 89,
        "2023-01": 100, "2023-02": 94, "2023-03": 104, "2023-04": 108,
        "2023-05": 124, "2023-06": 120, "2023-07": 133, "2023-08": 136,
        "2023-09": 131, "2023-10": 125, "2023-11": 133, "2023-12": 140,
    },
    "AMZN": {
        "2020-01": 95, "2020-02": 97, "2020-03": 97, "2020-04": 123,
        "2020-05": 123, "2020-06": 138, "2020-07": 158, "2020-08": 170,
        "2020-09": 158, "2020-10": 151, "2020-11": 160, "2020-12": 163,
        "2021-01": 164, "2021-02": 155, "2021-03": 155, "2021-04": 174,
        "2021-05": 163, "2021-06": 172, "2021-07": 182, "2021-08": 175,
        "2021-09": 164, "2021-10": 169, "2021-11": 177, "2021-12": 167,
        "2022-01": 150, "2022-02": 153, "2022-03": 163, "2022-04": 121,
        "2022-05": 120, "2022-06": 106, "2022-07": 134, "2022-08": 128,
        "2022-09": 113, "2022-10": 103, "2022-11": 94, "2022-12": 84,
        "2023-01": 103, "2023-02": 94, "2023-03": 103, "2023-04": 106,
        "2023-05": 120, "2023-06": 130, "2023-07": 134, "2023-08": 139,
        "2023-09": 127, "2023-10": 134, "2023-11": 147, "2023-12": 152,
    },
    "META": {
        "2020-01": 210, "2020-02": 193, "2020-03": 164, "2020-04": 204,
        "2020-05": 228, "2020-06": 235, "2020-07": 235, "2020-08": 286,
        "2020-09": 255, "2020-10": 263, "2020-11": 277, "2020-12": 273,
        "2021-01": 262, "2021-02": 257, "2021-03": 295, "2021-04": 325,
        "2021-05": 329, "2021-06": 347, "2021-07": 356, "2021-08": 372,
        "2021-09": 343, "2021-10": 323, "2021-11": 306, "2021-12": 334,
        "2022-01": 326, "2022-02": 211, "2022-03": 223, "2022-04": 200,
        "2022-05": 194, "2022-06": 162, "2022-07": 160, "2022-08": 164,
        "2022-09": 136, "2022-10": 99, "2022-11": 113, "2022-12": 121,
        "2023-01": 148, "2023-02": 174, "2023-03": 209, "2023-04": 240,
        "2023-05": 264, "2023-06": 287, "2023-07": 319, "2023-08": 296,
        "2023-09": 300, "2023-10": 302, "2023-11": 327, "2023-12": 354,
    },
    "TSLA": {
        "2020-01": 43, "2020-02": 45, "2020-03": 34, "2020-04": 52,
        "2020-05": 56, "2020-06": 72, "2020-07": 97, "2020-08": 151,
        "2020-09": 143, "2020-10": 143, "2020-11": 186, "2020-12": 227,
        "2021-01": 264, "2021-02": 225, "2021-03": 222, "2021-04": 232,
        "2021-05": 204, "2021-06": 227, "2021-07": 224, "2021-08": 246,
        "2021-09": 258, "2021-10": 372, "2021-11": 382, "2021-12": 352,
        "2022-01": 313, "2022-02": 290, "2022-03": 364, "2022-04": 290,
        "2022-05": 252, "2022-06": 224, "2022-07": 298, "2022-08": 275,
        "2022-09": 265, "2022-10": 228, "2022-11": 194, "2022-12": 123,
        "2023-01": 141, "2023-02": 205, "2023-03": 207, "2023-04": 165,
        "2023-05": 204, "2023-06": 261, "2023-07": 267, "2023-08": 258,
        "2023-09": 250, "2023-10": 201, "2023-11": 235, "2023-12": 249,
    },
    "NVDA": {
        "2020-01": 59, "2020-02": 64, "2020-03": 53, "2020-04": 72,
        "2020-05": 87, "2020-06": 95, "2020-07": 101, "2020-08": 131,
        "2020-09": 135, "2020-10": 131, "2020-11": 134, "2020-12": 131,
        "2021-01": 131, "2021-02": 149, "2021-03": 132, "2021-04": 152,
        "2021-05": 151, "2021-06": 200, "2021-07": 195, "2021-08": 225,
        "2021-09": 207, "2021-10": 246, "2021-11": 326, "2021-12": 294,
        "2022-01": 249, "2022-02": 243, "2022-03": 272, "2022-04": 190,
        "2022-05": 165, "2022-06": 153, "2022-07": 179, "2022-08": 161,
        "2022-09": 121, "2022-10": 138, "2022-11": 166, "2022-12": 146,
        "2023-01": 195, "2023-02": 232, "2023-03": 278, "2023-04": 277,
        "2023-05": 379, "2023-06": 423, "2023-07": 467, "2023-08": 493,
        "2023-09": 435, "2023-10": 405, "2023-11": 467, "2023-12": 495,
    },
    "JPM": {
        "2020-01": 135, "2020-02": 117, "2020-03": 90, "2020-04": 93,
        "2020-05": 95, "2020-06": 95, "2020-07": 96, "2020-08": 101,
        "2020-09": 96, "2020-10": 100, "2020-11": 120, "2020-12": 127,
        "2021-01": 131, "2021-02": 149, "2021-03": 153, "2021-04": 155,
        "2021-05": 166, "2021-06": 156, "2021-07": 150, "2021-08": 157,
        "2021-09": 163, "2021-10": 170, "2021-11": 160, "2021-12": 158,
        "2022-01": 150, "2022-02": 152, "2022-03": 137, "2022-04": 127,
        "2022-05": 128, "2022-06": 115, "2022-07": 114, "2022-08": 117,
        "2022-09": 107, "2022-10": 125, "2022-11": 136, "2022-12": 134,
        "2023-01": 140, "2023-02": 142, "2023-03": 128, "2023-04": 136,
        "2023-05": 137, "2023-06": 144, "2023-07": 155, "2023-08": 149,
        "2023-09": 146, "2023-10": 143, "2023-11": 157, "2023-12": 170,
    },
    "BAC": {
        "2020-01": 33, "2020-02": 28, "2020-03": 22, "2020-04": 23,
        "2020-05": 24, "2020-06": 24, "2020-07": 24, "2020-08": 26,
        "2020-09": 24, "2020-10": 24, "2020-11": 28, "2020-12": 30,
        "2021-01": 31, "2021-02": 36, "2021-03": 39, "2021-04": 40,
        "2021-05": 42, "2021-06": 41, "2021-07": 38, "2021-08": 39,
        "2021-09": 43, "2021-10": 48, "2021-11": 44, "2021-12": 44,
        "2022-01": 46, "2022-02": 46, "2022-03": 42, "2022-04": 37,
        "2022-05": 36, "2022-06": 31, "2022-07": 34, "2022-08": 35,
        "2022-09": 30, "2022-10": 34, "2022-11": 37, "2022-12": 33,
        "2023-01": 35, "2023-02": 34, "2023-03": 28, "2023-04": 29,
        "2023-05": 28, "2023-06": 29, "2023-07": 32, "2023-08": 29,
        "2023-09": 27, "2023-10": 26, "2023-11": 30, "2023-12": 34,
    },
    "WMT": {
        "2020-01": 115, "2020-02": 113, "2020-03": 117, "2020-04": 123,
        "2020-05": 122, "2020-06": 120, "2020-07": 131, "2020-08": 139,
        "2020-09": 140, "2020-10": 144, "2020-11": 152, "2020-12": 144,
        "2021-01": 144, "2021-02": 133, "2021-03": 136, "2021-04": 140,
        "2021-05": 141, "2021-06": 141, "2021-07": 143, "2021-08": 150,
        "2021-09": 140, "2021-10": 148, "2021-11": 144, "2021-12": 145,
        "2022-01": 139, "2022-02": 137, "2022-03": 149, "2022-04": 158,
        "2022-05": 132, "2022-06": 122, "2022-07": 131, "2022-08": 133,
        "2022-09": 130, "2022-10": 142, "2022-11": 154, "2022-12": 142,
        "2023-01": 143, "2023-02": 145, "2023-03": 147, "2023-04": 150,
        "2023-05": 151, "2023-06": 157, "2023-07": 160, "2023-08": 161,
        "2023-09": 163, "2023-10": 164, "2023-11": 157, "2023-12": 157,
    },
    # fmt: on
}

# Annualised daily volatility per ticker (realistic values)
_DAILY_VOL = {
    "AAPL": 0.020, "MSFT": 0.018, "GOOG": 0.021, "AMZN": 0.023,
    "META": 0.028, "TSLA": 0.040, "NVDA": 0.032, "JPM": 0.020,
    "BAC": 0.024, "WMT": 0.013,
}

# Sectors (same as conftest.STANDARD_SECTORS)
SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
    "AMZN": "Consumer Discretionary", "META": "Technology",
    "TSLA": "Consumer Discretionary", "NVDA": "Technology",
    "JPM": "Financials", "BAC": "Financials", "WMT": "Consumer Staples",
}

# Fama-French style factor monthly returns (approximate, annualised ~5-10%)
# Market, SMB, HML, Momentum, Quality
_FACTOR_MONTHLY = {
    # fmt: off
    "2020-01": [-0.001, -0.034, -0.059,  0.016,  0.022],
    "2020-02": [-0.081, -0.030, -0.058, -0.040,  0.005],
    "2020-03": [-0.134, -0.067, -0.116, -0.173, -0.037],
    "2020-04": [ 0.136,  0.026, -0.021,  0.035, -0.010],
    "2020-05": [ 0.057,  0.042, -0.031,  0.018,  0.015],
    "2020-06": [ 0.026, -0.004, -0.025,  0.043,  0.007],
    "2020-07": [ 0.058,  0.053, -0.024,  0.020,  0.013],
    "2020-08": [ 0.077, -0.010, -0.040,  0.054,  0.008],
    "2020-09": [-0.036,  0.003, -0.006, -0.028, -0.012],
    "2020-10": [-0.021,  0.047,  0.041, -0.032,  0.005],
    "2020-11": [ 0.124,  0.058,  0.054,  0.016,  0.009],
    "2020-12": [ 0.046,  0.049,  0.007,  0.013,  0.004],
    "2021-01": [-0.001,  0.091,  0.038, -0.064,  0.011],
    "2021-02": [ 0.028,  0.056,  0.063, -0.021,  0.003],
    "2021-03": [ 0.031, -0.005,  0.054,  0.038,  0.007],
    "2021-04": [ 0.053, -0.020,  0.003,  0.015,  0.012],
    "2021-05": [ 0.006,  0.002,  0.052,  0.024,  0.009],
    "2021-06": [ 0.023, -0.042, -0.053,  0.004,  0.008],
    "2021-07": [ 0.025, -0.038, -0.008,  0.013,  0.011],
    "2021-08": [ 0.030, -0.017,  0.005,  0.003,  0.006],
    "2021-09": [-0.043,  0.001,  0.043,  0.028, -0.004],
    "2021-10": [ 0.070, -0.017, -0.005,  0.036,  0.010],
    "2021-11": [-0.010, -0.021, -0.007,  0.009,  0.004],
    "2021-12": [ 0.039, -0.019,  0.029,  0.011,  0.008],
    "2022-01": [-0.057,  0.001,  0.106, -0.023, -0.005],
    "2022-02": [-0.025,  0.012,  0.027, -0.013,  0.002],
    "2022-03": [ 0.034,  0.003,  0.036,  0.017,  0.005],
    "2022-04": [-0.089, -0.019,  0.019, -0.005, -0.012],
    "2022-05": [ 0.002,  0.008,  0.044, -0.014,  0.003],
    "2022-06": [-0.083, -0.008,  0.004, -0.045, -0.015],
    "2022-07": [ 0.094,  0.001, -0.050,  0.005,  0.010],
    "2022-08": [-0.037, -0.020,  0.025,  0.005, -0.002],
    "2022-09": [-0.093, -0.010,  0.013, -0.037, -0.009],
    "2022-10": [ 0.081,  0.012,  0.073,  0.030,  0.007],
    "2022-11": [ 0.056, -0.011, -0.011,  0.016,  0.008],
    "2022-12": [-0.058,  0.001,  0.033,  0.011, -0.003],
    "2023-01": [ 0.067,  0.002, -0.038,  0.008,  0.009],
    "2023-02": [-0.024, -0.012,  0.011,  0.003, -0.001],
    "2023-03": [ 0.037, -0.043, -0.080,  0.007,  0.011],
    "2023-04": [ 0.016, -0.015, -0.003,  0.002,  0.005],
    "2023-05": [ 0.004, -0.041, -0.042,  0.013,  0.007],
    "2023-06": [ 0.069, -0.004, -0.019,  0.022,  0.012],
    "2023-07": [ 0.032,  0.025,  0.027,  0.009,  0.006],
    "2023-08": [-0.017, -0.026,  0.007, -0.008, -0.002],
    "2023-09": [-0.047, -0.003,  0.020, -0.021, -0.006],
    "2023-10": [-0.021,  0.017,  0.006, -0.011,  0.001],
    "2023-11": [ 0.091,  0.006, -0.016,  0.032,  0.013],
    "2023-12": [ 0.047,  0.017, -0.004,  0.010,  0.008],
    # fmt: on
}

FACTOR_NAMES = ["Market", "SMB", "HML", "Momentum", "Quality"]


def make_real_ohlcv(
    tickers=None,
    start="2020-01-02",
    end="2023-12-29",
    seed=42,
):
    """Generate daily OHLCV data from real monthly anchor prices.

    Interpolates between monthly closes with realistic daily noise.
    Returns MultiIndex DataFrame (date, ticker) with OHLCV columns
    matching the format of conftest.make_ohlcv().
    """
    tickers = tickers or list(_MONTHLY_CLOSES.keys())
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start, end=end)

    # Build monthly anchor dates and prices
    rows = []
    for ticker in tickers:
        if ticker not in _MONTHLY_CLOSES:
            continue
        monthly = _MONTHLY_CLOSES[ticker]
        vol = _DAILY_VOL.get(ticker, 0.02)

        # Build anchor date-price pairs
        anchor_dates = []
        anchor_prices = []
        for ym, price in sorted(monthly.items()):
            dt = pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0)
            # Find last business day of month
            dt = pd.Timestamp(dt)
            if dt.weekday() >= 5:
                dt -= pd.tseries.offsets.BDay(1)
            anchor_dates.append(dt)
            anchor_prices.append(float(price))

        # Interpolate daily prices between anchors
        anchor_series = pd.Series(anchor_prices, index=anchor_dates)
        # Reindex to business days and forward/backward fill
        daily_anchors = anchor_series.reindex(dates).interpolate(method="time")
        daily_anchors = daily_anchors.ffill().bfill()

        # Add realistic daily noise around the interpolated trend
        n = len(dates)
        noise = rng.normal(0, vol, n)
        # Cumulative noise centered (mean-reverting to anchor)
        cum_noise = np.cumsum(noise)
        # Mean-revert: blend with zero to prevent drift
        decay = 0.95
        mr_noise = np.zeros(n)
        for i in range(n):
            mr_noise[i] = noise[i] + (0 - (mr_noise[i - 1] if i > 0 else 0)) * (1 - decay)
            if i > 0:
                mr_noise[i] += mr_noise[i - 1] * decay

        # Scale noise so that daily price stays near anchor
        price_scale = daily_anchors.values * 0.5  # noise as pct of price
        daily_close = daily_anchors.values * (1 + mr_noise * 0.1)
        # Ensure positive
        daily_close = np.maximum(daily_close, 1.0)

        for i, dt in enumerate(dates):
            p = daily_close[i]
            intraday_range = abs(rng.normal(0, vol * 0.7))
            rows.append({
                "date": dt,
                "ticker": ticker,
                "open": p * (1 + rng.uniform(-0.005, 0.005)),
                "high": p * (1 + intraday_range),
                "low": p * (1 - intraday_range),
                "close": p,
                "adj_close": p,
                "volume": int(rng.uniform(5e6, 1e8)),
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def make_real_returns(
    tickers=None,
    start="2020-01-02",
    end="2023-12-29",
    seed=42,
):
    """Daily stock returns computed from real OHLCV data.

    Returns DataFrame (dates x tickers) matching conftest.make_returns() format.
    """
    ohlcv = make_real_ohlcv(tickers=tickers, start=start, end=end, seed=seed)
    # Pivot to get close prices per ticker
    close = ohlcv["close"].unstack("ticker")
    returns = close.pct_change().iloc[1:]  # drop first NaN row
    return returns


def make_real_factor_returns(
    start="2020-01-02",
    end="2023-12-29",
    seed=42,
):
    """Daily factor returns interpolated from monthly Fama-French style data.

    Returns DataFrame (dates x factors) matching conftest.make_factor_returns() format.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)

    # Build monthly anchor returns
    monthly_dates = []
    monthly_vals = []
    for ym, factors in sorted(_FACTOR_MONTHLY.items()):
        dt = pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0)
        if dt.weekday() >= 5:
            dt -= pd.tseries.offsets.BDay(1)
        monthly_dates.append(dt)
        monthly_vals.append(factors)

    monthly_df = pd.DataFrame(
        monthly_vals, index=monthly_dates, columns=FACTOR_NAMES,
    )

    # Distribute monthly returns evenly across business days in each month,
    # then add daily noise
    daily_factor = pd.DataFrame(0.0, index=dates, columns=FACTOR_NAMES)

    for i, ym in enumerate(sorted(_FACTOR_MONTHLY.keys())):
        month_start = pd.Timestamp(ym + "-01")
        month_end = month_start + pd.offsets.MonthEnd(0)
        mask = (dates >= month_start) & (dates <= month_end)
        n_days = mask.sum()
        if n_days == 0:
            continue
        monthly_ret = np.array(_FACTOR_MONTHLY[ym])
        # Daily return = monthly / n_days + noise
        daily_base = monthly_ret / n_days
        for j, dt in enumerate(dates[mask]):
            noise = rng.normal(0, 0.003, len(FACTOR_NAMES))
            daily_factor.loc[dt] = daily_base + noise

    return daily_factor


def make_real_market_data(tickers=None, as_of="2023-06-30", seed=42):
    """Build broker market_data dict for a specific date.

    Uses the monthly anchor prices closest to as_of to build realistic
    bid/ask/mid/volume data. Returns dict matching conftest.STANDARD_MARKET_DATA format.
    """
    tickers = tickers or list(_MONTHLY_CLOSES.keys())
    as_of_ts = pd.Timestamp(as_of)

    md = {}
    for ticker in tickers:
        if ticker not in _MONTHLY_CLOSES:
            continue
        monthly = _MONTHLY_CLOSES[ticker]
        # Find closest month <= as_of
        best_ym = None
        for ym in sorted(monthly.keys()):
            ym_ts = pd.Timestamp(ym + "-01")
            if ym_ts <= as_of_ts:
                best_ym = ym
        if best_ym is None:
            best_ym = min(monthly.keys())

        close = float(monthly[best_ym])
        spread = close * 0.001  # 10bps spread
        md[ticker] = {
            "bid": close - spread / 2,
            "ask": close + spread / 2,
            "mid": close,
            "last": close,
            "volume": 30_000_000,
            "adv": 30_000_000,
        }
    return md
