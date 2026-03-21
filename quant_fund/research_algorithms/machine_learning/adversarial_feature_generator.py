"""Adversarial feature generators for stress testing ML models.

Generates noise features, lookahead features, and non-stationary
features to test model robustness.
"""

import numpy as np
import pandas as pd


def generate_noise_features(
    n_samples: int,
    n_features: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate pure random noise features.

    These should have zero predictive power. A model that improves
    when these are added is likely overfitting.
    """
    rng = np.random.RandomState(seed)
    data = rng.randn(n_samples, n_features)
    return pd.DataFrame(
        data, columns=[f"noise_{i}" for i in range(n_features)]
    )


def generate_lookahead_feature(
    forward_returns: pd.Series,
    noise_level: float = 0.1,
    seed: int = 42,
) -> pd.Series:
    """Generate a feature that contains future information.

    This feature is a noisy version of forward returns — it should
    never be allowed into a production pipeline. Used to verify
    that leakage detection catches it.
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(forward_returns)) * noise_level
    return forward_returns + noise


def generate_nonstationary_feature(
    n_samples: int,
    break_point: float = 0.5,
    seed: int = 42,
) -> pd.Series:
    """Generate a feature with a structural break.

    The feature distribution changes at break_point fraction of samples.
    Tests whether models degrade gracefully under regime changes.
    """
    rng = np.random.RandomState(seed)
    bp = int(n_samples * break_point)
    first_half = rng.randn(bp) * 1.0
    second_half = rng.randn(n_samples - bp) * 3.0 + 2.0  # shifted distribution
    return pd.Series(np.concatenate([first_half, second_half]), name="nonstationary")


def perturb_features(
    features: pd.DataFrame,
    noise_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Add Gaussian noise to features (perturbation test).

    Args:
        features: Original feature DataFrame.
        noise_fraction: Noise magnitude as fraction of each feature's std.
        seed: Random seed.

    Returns:
        Perturbed copy of features.
    """
    rng = np.random.RandomState(seed)
    noise = pd.DataFrame(
        rng.randn(*features.shape) * features.std().values * noise_fraction,
        index=features.index,
        columns=features.columns,
    )
    return features + noise


def add_label_noise(
    returns: pd.Series,
    noise_fraction: float = 0.1,
    seed: int = 42,
) -> pd.Series:
    """Add noise to return labels (label noise test)."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(returns)) * returns.std() * noise_fraction
    return returns + noise
