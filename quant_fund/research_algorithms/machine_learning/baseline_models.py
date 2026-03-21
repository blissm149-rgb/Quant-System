"""Naive baseline models for benchmarking.

Provides simple baselines that ML models should beat to demonstrate
genuine predictive skill.
"""

import numpy as np
import pandas as pd


class NaiveBaselineModel:
    """Always predicts zero — the simplest possible baseline."""

    def __init__(self, config=None):
        pass

    def train_model(self, features, returns):
        return {"train_r2": 0.0, "oos_r2": 0.0}

    def predict(self, features):
        return pd.Series(0.0, index=features.index)


class RandomBaselineModel:
    """Predicts N(0, 1) random noise."""

    def __init__(self, config=None):
        cfg = config or {}
        self._seed = cfg.get("random_seed", 42)

    def train_model(self, features, returns):
        return {"train_r2": 0.0, "oos_r2": 0.0}

    def predict(self, features):
        rng = np.random.RandomState(self._seed)
        return pd.Series(rng.randn(len(features)), index=features.index)


class MeanReversionBaseline:
    """Predicts negative 5-day return (simple mean reversion)."""

    def __init__(self, config=None):
        cfg = config or {}
        self._lookback = cfg.get("mr_lookback", 5)
        self._returns = None

    def train_model(self, features, returns):
        self._returns = returns
        return {"train_r2": 0.0, "oos_r2": 0.0}

    def predict(self, features):
        if self._returns is None:
            return pd.Series(0.0, index=features.index)
        # Negative of recent return (mean reversion signal)
        recent_ret = self._returns.iloc[-self._lookback:].mean() if len(self._returns) >= self._lookback else 0.0
        return pd.Series(-recent_ret, index=features.index)


class MomentumBaseline:
    """Predicts positive 20-day return (simple momentum)."""

    def __init__(self, config=None):
        cfg = config or {}
        self._lookback = cfg.get("mom_lookback", 20)
        self._returns = None

    def train_model(self, features, returns):
        self._returns = returns
        return {"train_r2": 0.0, "oos_r2": 0.0}

    def predict(self, features):
        if self._returns is None:
            return pd.Series(0.0, index=features.index)
        recent_ret = self._returns.iloc[-self._lookback:].mean() if len(self._returns) >= self._lookback else 0.0
        return pd.Series(recent_ret, index=features.index)
