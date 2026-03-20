"""Realistic order flow generator for execution testing.

Generates order sequences with configurable size distribution,
urgency levels, timing patterns, and order types.
"""

import numpy as np
import pandas as pd

from quant_fund.broker_interface.broker_abstraction_layer import (
    Order,
    OrderSide,
    OrderType,
)


def make_orders(
    n=20,
    tickers=None,
    seed=42,
    size_distribution="lognormal",
    include_algos=True,
):
    """Generate a sequence of realistic orders.

    Parameters
    ----------
    n : int
        Number of orders to generate.
    tickers : list[str], optional
        Ticker symbols. Defaults to 5 common tickers.
    seed : int
        Random seed for reproducibility.
    size_distribution : str
        "lognormal" (realistic) or "uniform".
    include_algos : bool
        If True, include VWAP/TWAP order types.

    Returns
    -------
    list[Order]
        List of Order objects.
    """
    tickers = tickers or ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    rng = np.random.default_rng(seed)

    order_types = [OrderType.MARKET, OrderType.LIMIT]
    if include_algos:
        order_types.extend([OrderType.VWAP, OrderType.TWAP])

    base_prices = {
        "AAPL": 150.0, "MSFT": 300.0, "GOOG": 140.0,
        "AMZN": 175.0, "META": 350.0, "TSLA": 200.0,
        "NVDA": 450.0, "JPM": 190.0, "BAC": 33.5, "WMT": 160.0,
    }

    orders = []
    base_time = pd.Timestamp("2024-01-15 09:30:00")

    for i in range(n):
        ticker = rng.choice(tickers)
        side = rng.choice([OrderSide.BUY, OrderSide.SELL])

        if size_distribution == "lognormal":
            qty = int(np.exp(rng.normal(5, 1.5)))  # median ~150, range ~10-5000
            qty = max(1, min(qty, 50000))
        else:
            qty = int(rng.uniform(10, 1000))

        otype = rng.choice(order_types)
        price = base_prices.get(ticker, 100.0)

        limit_price = None
        if otype == OrderType.LIMIT:
            offset = rng.uniform(0.001, 0.01) * price
            if side == OrderSide.BUY:
                limit_price = round(price - offset, 2)
            else:
                limit_price = round(price + offset, 2)

        algo_params = {}
        if otype == OrderType.VWAP:
            algo_params = {"participation_rate": 0.05, "horizon_minutes": 60}
        elif otype == OrderType.TWAP:
            algo_params = {"horizon_minutes": 30, "num_slices": 6}

        timestamp = base_time + pd.Timedelta(minutes=rng.uniform(0, 390))

        orders.append(Order(
            ticker=ticker,
            side=side,
            quantity=qty,
            order_type=otype,
            limit_price=limit_price,
            algo_params=algo_params,
            strategy_id=f"strategy_{rng.integers(1, 4)}",
            timestamp=timestamp,
            order_id=f"ORD-{i:05d}",
        ))

    return orders


def make_rebalance_orders(
    current_weights,
    target_weights,
    nav=1_000_000,
    prices=None,
    seed=42,
):
    """Generate orders from a portfolio rebalance.

    Parameters
    ----------
    current_weights : dict[str, float]
        Current portfolio weights by ticker.
    target_weights : dict[str, float]
        Target portfolio weights by ticker.
    nav : float
        Portfolio net asset value.
    prices : dict[str, float], optional
        Current prices per ticker.

    Returns
    -------
    list[Order]
        Orders to execute the rebalance.
    """
    prices = prices or {
        "AAPL": 150.0, "MSFT": 300.0, "GOOG": 140.0,
        "AMZN": 175.0, "META": 350.0,
    }

    all_tickers = set(current_weights) | set(target_weights)
    orders = []

    for ticker in sorted(all_tickers):
        curr = current_weights.get(ticker, 0.0)
        tgt = target_weights.get(ticker, 0.0)
        delta = tgt - curr

        if abs(delta) < 1e-6:
            continue

        price = prices.get(ticker, 100.0)
        dollar_delta = delta * nav
        qty = abs(int(dollar_delta / price))

        if qty == 0:
            continue

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        otype = OrderType.VWAP if qty > 500 else OrderType.MARKET

        orders.append(Order(
            ticker=ticker,
            side=side,
            quantity=qty,
            order_type=otype,
            strategy_id="rebalance",
            order_id=f"REB-{ticker}",
        ))

    return orders
