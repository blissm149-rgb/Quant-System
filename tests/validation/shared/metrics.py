"""Common statistical metrics for validation framework."""

import numpy as np
import pandas as pd
from typing import Callable, Optional


def annualized_sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily returns."""
    excess = returns - rf / 252.0
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from daily returns (returned as positive number)."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(-drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    ann_ret = (1 + returns).prod() ** (252 / max(len(returns), 1)) - 1
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(ann_ret / mdd)


def information_coefficient(
    signal: pd.Series, forward_returns: pd.Series
) -> float:
    """Rank correlation (Spearman) between signal and forward returns."""
    aligned = pd.concat([signal, forward_returns], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def ic_tstat(ic_series: pd.Series) -> float:
    """T-statistic for mean IC being different from zero."""
    n = len(ic_series)
    if n < 2 or ic_series.std() == 0:
        return 0.0
    return float(ic_series.mean() / (ic_series.std() / np.sqrt(n)))


def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).dropna()


def bootstrap_ci(
    statistic_fn: Callable,
    data: pd.Series,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Bootstrap confidence interval for a statistic.

    Returns (lower, upper) bounds.
    """
    rng = np.random.default_rng(seed)
    stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = data.iloc[rng.integers(0, n, size=n)]
        sample.index = range(n)
        stats.append(statistic_fn(sample))
    stats = sorted(stats)
    alpha = (1 - ci) / 2
    lower = stats[int(alpha * len(stats))]
    upper = stats[int((1 - alpha) * len(stats))]
    return (lower, upper)


def deflated_sharpe_ratio(
    sharpe: float, n_trials: int, skew: float, kurtosis: float, T: int
) -> float:
    """Deflated Sharpe Ratio (Harvey et al., 2016).

    Probability that the observed Sharpe is not due to multiple testing.
    Returns value in [0, 1]; values > 0.5 suggest genuine skill.
    """
    from scipy import stats as scipy_stats

    e_max_sharpe = _expected_max_sharpe(n_trials, T)
    se_sharpe = np.sqrt(
        (1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2) / T
    )
    if se_sharpe == 0:
        return 0.0
    test_stat = (sharpe - e_max_sharpe) / se_sharpe
    return float(scipy_stats.norm.cdf(test_stat))


def _expected_max_sharpe(n_trials: int, T: int) -> float:
    """Expected maximum Sharpe under the null (all strategies have zero alpha)."""
    from scipy import stats as scipy_stats

    gamma = 0.5772156649  # Euler-Mascheroni
    z = scipy_stats.norm.ppf(1 - 1 / n_trials)
    return float(
        (1 - gamma) * scipy_stats.norm.ppf(1 - 1 / n_trials)
        + gamma * scipy_stats.norm.ppf(1 - 1 / (n_trials * np.e))
    ) / np.sqrt(T / 252)


def block_bootstrap(
    data: pd.Series,
    block_size: int,
    n_bootstrap: int,
    statistic_fn: Callable,
    seed: int = 42,
) -> list[float]:
    """Block bootstrap resampling for time-series data."""
    rng = np.random.default_rng(seed)
    n = len(data)
    n_blocks = max(1, n // block_size)
    results = []
    for _ in range(n_bootstrap):
        blocks = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample_indices = np.concatenate(
            [np.arange(b, b + block_size) for b in blocks]
        )[:n]
        sample = data.iloc[sample_indices].reset_index(drop=True)
        results.append(statistic_fn(sample))
    return results


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Bonferroni correction: returns list of bool (significant or not)."""
    m = len(p_values)
    return [p < alpha / m for p in p_values]


def benjamini_hochberg(p_values: list[float], fdr: float = 0.10) -> list[bool]:
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    thresholds = np.array([(i + 1) / m * fdr for i in range(m)])
    reject = sorted_p <= thresholds
    # Find largest k where p(k) <= k/m * fdr
    if not np.any(reject):
        return [False] * m
    max_k = np.max(np.where(reject))
    result = [False] * m
    for i in range(max_k + 1):
        result[sorted_indices[i]] = True
    return result
