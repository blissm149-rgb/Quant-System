"""Enhanced factor returns generator with regime-dependent behavior.

Extends the base make_factor_returns() from conftest with regime-specific
factor dynamics and stress scenarios.
"""

import numpy as np
import pandas as pd


def make_regime_factor_returns(
    regime="normal",
    n_dates=252,
    factors=None,
    seed=42,
):
    """Generate factor returns with regime-specific characteristics.

    Parameters
    ----------
    regime : str
        One of "normal", "crisis", "momentum_crash", "value_rally", "low_vol".
    n_dates : int
        Number of trading days.
    factors : list[str], optional
        Factor names. Defaults to Fama-French 5.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Factor returns (dates x factors).
    """
    factors = factors or ["Market", "Size", "Value", "Momentum", "Quality"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)

    regime_params = {
        "normal": {
            "means": [0.0003, 0.0001, 0.0001, 0.0002, 0.0001],
            "vols": [0.01, 0.005, 0.005, 0.007, 0.004],
        },
        "crisis": {
            "means": [-0.003, -0.001, -0.002, -0.005, 0.001],
            "vols": [0.03, 0.015, 0.015, 0.025, 0.008],
        },
        "momentum_crash": {
            "means": [0.0001, 0.0001, 0.001, -0.005, 0.0001],
            "vols": [0.012, 0.006, 0.006, 0.020, 0.005],
        },
        "value_rally": {
            "means": [0.0003, 0.0003, 0.003, 0.0001, 0.0001],
            "vols": [0.01, 0.007, 0.010, 0.007, 0.004],
        },
        "low_vol": {
            "means": [0.0002, 0.0001, 0.0001, 0.0001, 0.0001],
            "vols": [0.005, 0.003, 0.003, 0.004, 0.002],
        },
    }

    params = regime_params.get(regime, regime_params["normal"])
    means = np.array(params["means"][:len(factors)])
    vols = np.array(params["vols"][:len(factors)])

    data = rng.normal(means, vols, size=(n_dates, len(factors)))
    return pd.DataFrame(data, index=dates, columns=factors)


def make_correlated_factor_returns(
    n_dates=252,
    factors=None,
    correlation=0.5,
    seed=42,
):
    """Generate factor returns with specified cross-factor correlation.

    Useful for testing high-correlation regimes where diversification breaks down.
    """
    factors = factors or ["Market", "Size", "Value", "Momentum", "Quality"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    n_factors = len(factors)

    # Build correlation matrix
    corr = np.full((n_factors, n_factors), correlation)
    np.fill_diagonal(corr, 1.0)
    vols = np.array([0.01, 0.005, 0.005, 0.007, 0.004])[:n_factors]
    cov = np.outer(vols, vols) * corr

    data = rng.multivariate_normal(np.zeros(n_factors), cov, size=n_dates)
    return pd.DataFrame(data, index=dates, columns=factors)
