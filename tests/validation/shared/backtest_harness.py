"""Lightweight backtest harness for validation tests.

Iterates over dates, calls strategy.compute() with point-in-time data,
applies simple equal-weight or optimizer-based weighting, records returns.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine


@dataclass
class BacktestResult:
    """Container for backtest output."""

    daily_returns: pd.Series
    weights_history: dict  # date -> pd.Series of weights
    turnover: pd.Series
    costs: pd.Series
    nav_series: pd.Series

    @property
    def sharpe(self) -> float:
        if self.daily_returns.std() == 0:
            return 0.0
        return float(
            self.daily_returns.mean() / self.daily_returns.std() * np.sqrt(252)
        )

    @property
    def max_drawdown(self) -> float:
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        dd = (cumulative - running_max) / running_max
        return float(-dd.min())

    @property
    def total_return(self) -> float:
        return float((1 + self.daily_returns).prod() - 1)


class BacktestHarness:
    """Lightweight backtest loop for validation framework.

    Runs a single strategy over OHLCV data using point-in-time alignment.
    """

    def __init__(self, seed: int = 42, initial_nav: float = 1_000_000.0):
        self._seed = seed
        self._initial_nav = initial_nav
        self._alignment_engine = DataAlignmentEngine()

    def run(
        self,
        ohlcv: pd.DataFrame,
        strategy,
        cost_bps: float = 5.0,
        rebalance_freq: int = 1,
    ) -> BacktestResult:
        """Run a simple backtest.

        Args:
            ohlcv: MultiIndex (date, ticker) DataFrame with OHLCV columns.
            strategy: Object with compute(data, as_of) -> pd.Series of scores.
            cost_bps: Transaction cost in basis points per unit turnover.
            rebalance_freq: Rebalance every N days.

        Returns:
            BacktestResult with daily returns, weights, turnover, costs, and NAV.
        """
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        lookback = getattr(strategy, "lookback_days", 252)

        # Need at least lookback + 1 days
        if len(dates) <= lookback:
            return self._empty_result(dates)

        daily_returns = {}
        weights_history = {}
        turnover_series = {}
        cost_series = {}
        prev_weights = pd.Series(dtype=float)

        for i in range(lookback, len(dates) - 1):
            as_of = dates[i]
            next_date = dates[i + 1]

            if (i - lookback) % rebalance_freq != 0:
                # Hold existing weights
                if len(prev_weights) > 0:
                    day_data = ohlcv.loc[as_of:next_date]
                    ret = self._compute_portfolio_return(
                        prev_weights, ohlcv, as_of, next_date
                    )
                    daily_returns[next_date] = ret
                continue

            # Get point-in-time data
            try:
                aligned = self._alignment_engine.get_aligned_data(
                    ohlcv, as_of, lookback
                )
            except Exception:
                continue

            if aligned.empty:
                continue

            # Compute signals
            try:
                scores = strategy.compute(aligned, as_of)
            except Exception:
                continue

            if scores.empty or scores.isna().all():
                continue

            # Simple long-short weights from scores
            weights = self._scores_to_weights(scores)
            weights_history[as_of] = weights

            # Compute turnover
            if len(prev_weights) > 0:
                common = weights.index.union(prev_weights.index)
                w_new = weights.reindex(common, fill_value=0)
                w_old = prev_weights.reindex(common, fill_value=0)
                to = float((w_new - w_old).abs().sum())
            else:
                to = float(weights.abs().sum())

            turnover_series[as_of] = to
            cost = to * cost_bps / 10_000.0
            cost_series[as_of] = cost

            # Compute portfolio return for next day
            ret = self._compute_portfolio_return(weights, ohlcv, as_of, next_date)
            daily_returns[next_date] = ret - cost

            prev_weights = weights

        return self._build_result(
            daily_returns, weights_history, turnover_series, cost_series
        )

    def _scores_to_weights(self, scores: pd.Series) -> pd.Series:
        """Convert scores to dollar-neutral long-short weights."""
        clean = scores.dropna()
        if len(clean) == 0:
            return pd.Series(dtype=float)
        # Rank-based: top half long, bottom half short
        ranks = clean.rank(pct=True)
        weights = ranks - 0.5  # center around 0
        # Normalize to unit gross exposure
        gross = weights.abs().sum()
        if gross > 0:
            weights = weights / gross
        return weights

    def _compute_portfolio_return(
        self,
        weights: pd.Series,
        ohlcv: pd.DataFrame,
        current_date: pd.Timestamp,
        next_date: pd.Timestamp,
    ) -> float:
        """Compute weighted portfolio return between two dates."""
        ret_sum = 0.0
        for ticker in weights.index:
            try:
                cur_price = ohlcv.loc[(current_date, ticker), "close"]
                nxt_price = ohlcv.loc[(next_date, ticker), "close"]
                stock_ret = (nxt_price - cur_price) / cur_price
                ret_sum += weights[ticker] * stock_ret
            except (KeyError, ZeroDivisionError):
                continue
        return ret_sum

    def _empty_result(self, dates) -> BacktestResult:
        return BacktestResult(
            daily_returns=pd.Series(dtype=float),
            weights_history={},
            turnover=pd.Series(dtype=float),
            costs=pd.Series(dtype=float),
            nav_series=pd.Series(dtype=float),
        )

    def _build_result(
        self, daily_returns, weights_history, turnover_series, cost_series
    ) -> BacktestResult:
        rets = pd.Series(daily_returns).sort_index()
        nav = self._initial_nav * (1 + rets).cumprod()
        return BacktestResult(
            daily_returns=rets,
            weights_history=weights_history,
            turnover=pd.Series(turnover_series).sort_index(),
            costs=pd.Series(cost_series).sort_index(),
            nav_series=nav,
        )
