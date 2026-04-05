"""Shared test fixtures for QuantFund V8.

Consolidates data generators and common setup previously
duplicated across 11 test files. All fixtures use deterministic
seeds for reproducibility.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderType,
)
from quant_fund.broker_interface.simulation_broker import SimulationBroker


# ── Standard universes ──────────────────────────────────────────────

STANDARD_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META",
                     "TSLA", "NVDA", "JPM", "BAC", "WMT"]

STANDARD_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
    "AMZN": "Consumer Discretionary", "META": "Technology",
    "TSLA": "Consumer Discretionary", "NVDA": "Technology",
    "JPM": "Financials", "BAC": "Financials", "WMT": "Consumer Staples",
}

STANDARD_MARKET_DATA = {
    "AAPL": {"bid": 149.0, "ask": 151.0, "mid": 150.0, "last": 150.0,
             "volume": 50_000_000, "adv": 50_000_000},
    "MSFT": {"bid": 299.0, "ask": 301.0, "mid": 300.0, "last": 300.0,
             "volume": 30_000_000, "adv": 30_000_000},
    "GOOG": {"bid": 139.0, "ask": 141.0, "mid": 140.0, "last": 140.0,
             "volume": 20_000_000, "adv": 20_000_000},
    "AMZN": {"bid": 174.0, "ask": 176.0, "mid": 175.0, "last": 175.0,
             "volume": 40_000_000, "adv": 40_000_000},
    "META": {"bid": 349.0, "ask": 351.0, "mid": 350.0, "last": 350.0,
             "volume": 15_000_000, "adv": 15_000_000},
    "TSLA": {"bid": 199.0, "ask": 201.0, "mid": 200.0, "last": 200.0,
             "volume": 80_000_000, "adv": 80_000_000},
    "NVDA": {"bid": 449.0, "ask": 451.0, "mid": 450.0, "last": 450.0,
             "volume": 25_000_000, "adv": 25_000_000},
    "JPM": {"bid": 189.0, "ask": 191.0, "mid": 190.0, "last": 190.0,
             "volume": 10_000_000, "adv": 10_000_000},
    "BAC": {"bid": 33.0, "ask": 34.0, "mid": 33.5, "last": 33.5,
            "volume": 35_000_000, "adv": 35_000_000},
    "WMT": {"bid": 159.0, "ask": 161.0, "mid": 160.0, "last": 160.0,
            "volume": 8_000_000, "adv": 8_000_000},
}


# ── Synthetic data generators ───────────────────────────────────────

def make_ohlcv(
    tickers=None, start="2019-01-02", periods=300, seed=42
):
    """Generate synthetic OHLCV data.

    Returns MultiIndex DataFrame with (date, ticker) index and
    columns: open, high, low, close, volume, adj_close.
    """
    tickers = tickers or STANDARD_TICKERS[:5]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    rows = []
    for ticker in tickers:
        base = rng.uniform(20, 200)
        rets = rng.normal(0.0005, 0.02, size=periods)
        prices = base * np.cumprod(1 + rets)
        for i, dt in enumerate(dates):
            p = prices[i]
            rows.append({
                "date": dt,
                "ticker": ticker,
                "open": p * (1 + rng.uniform(-0.005, 0.005)),
                "high": p * (1 + abs(rng.normal(0, 0.01))),
                "low": p * (1 - abs(rng.normal(0, 0.01))),
                "close": p,
                "adj_close": p,
                "volume": int(rng.uniform(1e6, 1e8)),
            })
    df = pd.DataFrame(rows)
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def make_returns(n_dates=300, tickers=None, seed=42):
    """Generate stock returns matrix (dates × tickers)."""
    tickers = tickers or STANDARD_TICKERS[:5]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_dates)
    data = rng.normal(0.0005, 0.02, (n_dates, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


def make_factor_returns(n_dates=300, factors=None, seed=42):
    """Generate factor returns for risk models."""
    factors = factors or ["Market", "Size", "Value", "Momentum", "Quality"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_dates)
    data = rng.normal(0, 0.01, (n_dates, len(factors)))
    return pd.DataFrame(data, index=dates, columns=factors)


def make_cointegrated_pair(n=500, seed=123):
    """Generate two cointegrated price series.

    Y = 1.5 * X + mean-reverting spread.
    """
    rng = np.random.default_rng(seed)
    x_rets = rng.normal(0.0005, 0.02, n)
    x = 100.0 * np.cumprod(1 + x_rets)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.95 * spread[i - 1] + rng.normal(0, 0.5)
    y = 1.5 * x + spread + 50
    dates = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"X": x, "Y": y}, index=dates)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def standard_tickers():
    return list(STANDARD_TICKERS)


@pytest.fixture
def standard_sectors():
    return dict(STANDARD_SECTORS)


@pytest.fixture
def synthetic_ohlcv():
    return make_ohlcv()


@pytest.fixture
def synthetic_returns():
    return make_returns()


@pytest.fixture
def synthetic_factor_returns():
    return make_factor_returns()


@pytest.fixture
def simulation_broker():
    """Pre-configured SimulationBroker with market data for 10 tickers."""
    broker = SimulationBroker({
        "initial_cash": 1_000_000.0,
        "enforce_cash_floor": True,
    })
    broker.set_market_data(STANDARD_MARKET_DATA)
    return broker


@pytest.fixture
def constraint_set():
    """Standard constraint parameters."""
    return {
        "max_position_size": 0.02,
        "max_sector_exposure": 0.20,
        "max_leverage": 2.0,
        "dollar_neutral": True,
    }


@pytest.fixture
def tmp_registry(tmp_path):
    """Create a temporary model registry with one production model.

    Directory layout:
        tmp_path/
        +-- production/
        |   +-- test_model -> ../versions/test_model/v1/
        +-- staging/
        +-- versions/
        |   +-- test_model/
        |       +-- v1/
        |           +-- model.pkl
        |           +-- metadata.json
        +-- retired/
    """
    (tmp_path / "production").mkdir()
    (tmp_path / "staging").mkdir()
    (tmp_path / "versions" / "test_model" / "v1").mkdir(parents=True)
    (tmp_path / "retired").mkdir()

    # Write dummy model weights
    weights = {"coefficients": [0.1, 0.2, 0.3]}
    weights_path = tmp_path / "versions" / "test_model" / "v1" / "model.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(weights, f)

    # Write metadata
    metadata = {
        "trained_at": "2026-04-04T22:00:00Z",
        "data_hash": "abc123",
        "validation": {
            "oos_sharpe": 1.2,
            "oos_max_drawdown": -0.08,
            "prediction_mean": 0.001,
            "prediction_std": 0.05,
        },
    }
    meta_path = tmp_path / "versions" / "test_model" / "v1" / "metadata.json"
    meta_path.write_text(json.dumps(metadata))

    # Create production symlink
    prod_link = tmp_path / "production" / "test_model"
    prod_link.symlink_to("../versions/test_model/v1/")

    return tmp_path
