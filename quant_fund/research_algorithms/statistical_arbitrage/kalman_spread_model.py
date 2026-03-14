"""Kalman filter spread model for pairs trading.

Dynamically estimates hedge ratio via Kalman filter on the spread
between cointegrated pairs. Entry/exit based on z-score of spread.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


@dataclass
class KalmanState:
    """State of the Kalman filter for spread estimation."""

    hedge_ratio: float = 1.0
    intercept: float = 0.0
    P: np.ndarray = field(default_factory=lambda: np.eye(2) * 1.0)
    spread_mean: float = 0.0
    spread_var: float = 1.0


class KalmanSpreadModel(BaseFeatureGenerator):
    """Kalman filter-based dynamic hedge ratio and spread z-score.

    State: [hedge_ratio, intercept]
    Observation: price_a = hedge_ratio * price_b + intercept + noise

    Entry: |z-score| > entry_threshold (default 2.0)
    Exit: |z-score| < exit_threshold (default 0.5)
    Stop: |z-score| > stop_threshold (default 3.5)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._entry_threshold = cfg.get("kalman_entry_threshold", 2.0)
        self._exit_threshold = cfg.get("kalman_exit_threshold", 0.5)
        self._stop_threshold = cfg.get("kalman_stop_threshold", 3.5)
        self._observation_noise = cfg.get("kalman_obs_noise", 1.0)
        self._transition_noise = cfg.get("kalman_trans_noise", 1e-4)
        self._spread_ema_span = cfg.get("kalman_spread_ema_span", 60)
        super().__init__(
            feature_name="kalman_spread_zscore",
            lookback_days=365,
            recompute_frequency="daily",
            config=config,
        )

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Not applicable for single-ticker computation.

        Use compute_pair_signal() for pair-level signals.
        Returns empty series — pair signals are aggregated externally.
        """
        return pd.Series(dtype=float, name=self.feature_name)

    def compute_pair_signal(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> pd.DataFrame:
        """Run Kalman filter on a pair and produce signal time series.

        Args:
            prices_a: Price series for the long leg (DatetimeIndex).
            prices_b: Price series for the short leg (DatetimeIndex).

        Returns:
            DataFrame with columns: hedge_ratio, spread, z_score, signal
            signal values: 1 (long spread), -1 (short spread), 0 (flat)
        """
        common_idx = prices_a.index.intersection(prices_b.index)
        if len(common_idx) < 10:
            return pd.DataFrame()

        pa = prices_a.loc[common_idx].values
        pb = prices_b.loc[common_idx].values

        n = len(common_idx)
        hedge_ratios = np.zeros(n)
        intercepts = np.zeros(n)
        spreads = np.zeros(n)
        z_scores = np.zeros(n)
        signals = np.zeros(n)

        state = KalmanState()
        Q = np.eye(2) * self._transition_noise
        R = self._observation_noise
        position = 0  # current position: 1, -1, or 0

        for t in range(n):
            # Prediction step
            # State transition: x_t = x_{t-1} (random walk)
            P_pred = state.P + Q

            # Observation: y_t = H * x_t + noise
            # y_t = price_a[t], H = [price_b[t], 1]
            H = np.array([pb[t], 1.0])
            y = pa[t]

            # Innovation
            y_pred = H @ np.array([state.hedge_ratio, state.intercept])
            innovation = y - y_pred
            S = H @ P_pred @ H + R

            # Kalman gain
            K = P_pred @ H / S

            # Update
            x_updated = np.array([state.hedge_ratio, state.intercept]) + K * innovation
            state.hedge_ratio = x_updated[0]
            state.intercept = x_updated[1]
            state.P = (np.eye(2) - np.outer(K, H)) @ P_pred

            # Spread
            spread = pa[t] - state.hedge_ratio * pb[t] - state.intercept

            # EMA of spread mean and variance
            alpha = 2.0 / (self._spread_ema_span + 1)
            state.spread_mean = alpha * spread + (1 - alpha) * state.spread_mean
            state.spread_var = alpha * (spread - state.spread_mean) ** 2 + (1 - alpha) * state.spread_var

            spread_std = np.sqrt(max(state.spread_var, 1e-10))
            z = (spread - state.spread_mean) / spread_std

            # Signal logic
            if position == 0:
                if z > self._entry_threshold:
                    position = -1  # short spread
                elif z < -self._entry_threshold:
                    position = 1  # long spread
            elif position == 1:
                if z > -self._exit_threshold:
                    position = 0
                elif z < -self._stop_threshold:
                    position = 0  # stop loss
            elif position == -1:
                if z < self._exit_threshold:
                    position = 0
                elif z > self._stop_threshold:
                    position = 0  # stop loss

            hedge_ratios[t] = state.hedge_ratio
            intercepts[t] = state.intercept
            spreads[t] = spread
            z_scores[t] = z
            signals[t] = position

        return pd.DataFrame(
            {
                "hedge_ratio": hedge_ratios,
                "spread": spreads,
                "z_score": z_scores,
                "signal": signals,
            },
            index=common_idx,
        )

    def validate(self, feature_output: pd.Series) -> bool:
        return True
