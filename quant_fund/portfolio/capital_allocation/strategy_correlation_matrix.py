"""Strategy correlation matrix for capital allocation.

Computes rolling pairwise correlation of strategy daily returns.
High-correlation strategies compete for capital.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyCorrelationMatrix:
    """Computes rolling pairwise correlation between strategy returns.

    Used by dynamic_strategy_allocator to penalise correlated strategies.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._rolling_window = cfg.get("correlation_window", 60)
        self._min_observations = cfg.get("correlation_min_obs", 30)

    def compute(
        self, strategy_returns: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Compute pairwise correlation matrix from strategy return series.

        Args:
            strategy_returns: Dict mapping strategy_name -> daily return Series.

        Returns:
            DataFrame (strategies x strategies) correlation matrix.
        """
        if len(strategy_returns) < 2:
            names = list(strategy_returns.keys())
            return pd.DataFrame(
                np.eye(len(names)), index=names, columns=names
            )

        # Build aligned DataFrame
        returns_df = pd.DataFrame(strategy_returns)
        # Use only trailing window
        returns_df = returns_df.iloc[-self._rolling_window :]

        # Require minimum observations
        valid_cols = [
            col for col in returns_df.columns
            if returns_df[col].notna().sum() >= self._min_observations
        ]

        if len(valid_cols) < 2:
            return pd.DataFrame(
                np.eye(len(strategy_returns)),
                index=list(strategy_returns.keys()),
                columns=list(strategy_returns.keys()),
            )

        corr = returns_df[valid_cols].corr()

        # Fill in any strategies that didn't have enough data
        all_names = list(strategy_returns.keys())
        diag = np.eye(len(all_names))
        result = pd.DataFrame(diag, index=all_names, columns=all_names)
        for i in valid_cols:
            for j in valid_cols:
                result.loc[i, j] = corr.loc[i, j]

        return result

    def get_crowding_penalty(
        self, correlation_matrix: pd.DataFrame
    ) -> pd.Series:
        """Compute a crowding penalty for each strategy.

        Penalty = average absolute off-diagonal correlation.
        Higher penalty means more overlap with other strategies.

        Returns:
            Series of penalties indexed by strategy name.
        """
        penalties = {}
        for strat in correlation_matrix.index:
            others = correlation_matrix.loc[strat].drop(strat, errors="ignore")
            penalties[strat] = others.abs().mean() if len(others) > 0 else 0.0
        return pd.Series(penalties)
