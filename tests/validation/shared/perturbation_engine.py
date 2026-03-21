"""Perturbation engine for robustness testing.

Generates perturbed variants of parameters, data, and execution assumptions.
"""

import numpy as np
import pandas as pd


class PerturbationEngine:
    """Generates perturbed variants for robustness testing."""

    def perturb_parameter(
        self,
        base: float,
        pct_range: float = 0.20,
        n_samples: int = 20,
        seed: int = 42,
    ) -> list[float]:
        """Generate n_samples values uniformly distributed in [base*(1-pct), base*(1+pct)]."""
        rng = np.random.default_rng(seed)
        low = base * (1 - pct_range)
        high = base * (1 + pct_range)
        return sorted(rng.uniform(low, high, n_samples).tolist())

    def perturb_returns(
        self,
        returns: pd.DataFrame,
        noise_std: float = 0.001,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Add Gaussian noise to a returns DataFrame."""
        rng = np.random.default_rng(seed)
        noise = pd.DataFrame(
            rng.normal(0, noise_std, size=returns.shape),
            index=returns.index,
            columns=returns.columns,
        )
        return returns + noise

    def add_execution_noise(
        self,
        fill_prices: pd.Series,
        noise_bps: float = 5.0,
        seed: int = 42,
    ) -> pd.Series:
        """Add random execution noise to fill prices."""
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, noise_bps / 10_000.0, size=len(fill_prices))
        return fill_prices * (1 + noise)

    def shuffle_dates(
        self, returns: pd.DataFrame, seed: int = 42
    ) -> pd.DataFrame:
        """Shuffle date ordering, destroying temporal structure."""
        rng = np.random.default_rng(seed)
        shuffled_values = returns.values.copy()
        rng.shuffle(shuffled_values)
        return pd.DataFrame(
            shuffled_values, index=returns.index, columns=returns.columns
        )

    def inject_stale_prices(
        self,
        ohlcv: pd.DataFrame,
        stale_pct: float = 0.05,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Replace a fraction of prices with the previous day's value."""
        rng = np.random.default_rng(seed)
        result = ohlcv.copy()
        dates = result.index.get_level_values(0).unique().sort_values()
        n_stale = max(1, int(len(dates) * stale_pct))
        stale_dates = rng.choice(dates[1:], size=n_stale, replace=False)

        for d in stale_dates:
            idx = dates.get_loc(d)
            if idx > 0:
                prev_d = dates[idx - 1]
                for col in ["open", "high", "low", "close"]:
                    if col in result.columns:
                        try:
                            result.loc[d, col] = result.loc[prev_d, col].values
                        except (KeyError, ValueError):
                            pass
        return result
