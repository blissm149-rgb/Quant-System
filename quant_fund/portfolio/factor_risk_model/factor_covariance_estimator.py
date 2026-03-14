"""Factor covariance estimator using Ledoit-Wolf shrinkage.

Estimates the factor covariance matrix and stock-specific idiosyncratic
variance from residuals. Full covariance: Sigma = B F B^T + D.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorCovarianceEstimator:
    """Estimates factor covariance matrix with shrinkage.

    Uses Ledoit-Wolf shrinkage to regularise the factor covariance matrix.
    Idiosyncratic variance is estimated from regression residuals.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._estimation_window = cfg.get("cov_estimation_window", 252)
        self._min_observations = cfg.get("cov_min_observations", 60)
        self._shrinkage_target = cfg.get("shrinkage_target", "identity")
        self._annualise = cfg.get("annualise", True)

    def estimate(
        self,
        factor_returns: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """Estimate factor covariance matrix with Ledoit-Wolf shrinkage.

        Args:
            factor_returns: DataFrame (dates x factors) of daily factor returns.
            as_of: Point-in-time boundary.

        Returns:
            DataFrame (factors x factors) covariance matrix.
        """
        F = factor_returns[factor_returns.index < as_of].iloc[
            -self._estimation_window :
        ]

        if len(F) < self._min_observations:
            logger.warning("Insufficient data for covariance estimation")
            n = len(factor_returns.columns)
            return pd.DataFrame(
                np.eye(n) * 0.04 / 252,
                index=factor_returns.columns,
                columns=factor_returns.columns,
            )

        sample_cov = F.cov().values.copy()
        shrunk = self._ledoit_wolf_shrinkage(sample_cov, len(F))

        if self._annualise:
            shrunk *= 252

        return pd.DataFrame(
            shrunk, index=F.columns, columns=F.columns
        )

    def estimate_idiosyncratic(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        factor_exposures: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Estimate idiosyncratic (stock-specific) variance from residuals.

        Args:
            stock_returns: DataFrame (dates x tickers).
            factor_returns: DataFrame (dates x factors).
            factor_exposures: DataFrame (tickers x factors).
            as_of: Point-in-time boundary.

        Returns:
            Series of idiosyncratic variances indexed by ticker.
        """
        R = stock_returns[stock_returns.index < as_of].iloc[
            -self._estimation_window :
        ]
        F = factor_returns[factor_returns.index < as_of].iloc[
            -self._estimation_window :
        ]
        common_dates = R.index.intersection(F.index)
        R = R.loc[common_dates]
        F = F.loc[common_dates]

        idio_var = {}
        common_factors = factor_exposures.columns.intersection(F.columns)

        for ticker in R.columns:
            if ticker not in factor_exposures.index:
                idio_var[ticker] = R[ticker].var() if not R[ticker].isna().all() else 0.0
                continue

            betas = factor_exposures.loc[ticker, common_factors].values
            factor_contribution = F[common_factors].values @ betas
            residuals = R[ticker].values - factor_contribution

            valid = np.isfinite(residuals)
            if valid.sum() < 10:
                idio_var[ticker] = 0.0
            else:
                var = np.var(residuals[valid], ddof=1)
                idio_var[ticker] = var * 252 if self._annualise else var

        return pd.Series(idio_var)

    def build_full_covariance(
        self,
        factor_exposures: pd.DataFrame,
        factor_covariance: pd.DataFrame,
        idiosyncratic_variance: pd.Series,
    ) -> pd.DataFrame:
        """Build full stock covariance matrix: Sigma = B F B^T + D.

        Args:
            factor_exposures: DataFrame (tickers x factors).
            factor_covariance: DataFrame (factors x factors).
            idiosyncratic_variance: Series of per-stock variances.

        Returns:
            DataFrame (tickers x tickers) covariance matrix.
        """
        common_factors = factor_exposures.columns.intersection(
            factor_covariance.columns
        )
        B = factor_exposures[common_factors].values
        F_cov = factor_covariance.loc[common_factors, common_factors].values

        # B F B^T
        systematic = B @ F_cov @ B.T

        # D (diagonal)
        tickers = factor_exposures.index
        D = np.diag(idiosyncratic_variance.reindex(tickers, fill_value=0.0).values)

        full_cov = systematic + D
        return pd.DataFrame(full_cov, index=tickers, columns=tickers)

    def _ledoit_wolf_shrinkage(
        self, sample_cov: np.ndarray, n_obs: int
    ) -> np.ndarray:
        """Apply Ledoit-Wolf shrinkage to the sample covariance matrix.

        Shrinks toward a scaled identity matrix (constant correlation target).
        """
        p = sample_cov.shape[0]

        if self._shrinkage_target == "identity":
            mu = np.trace(sample_cov) / p
            target = mu * np.eye(p)
        else:
            # Constant correlation target
            var = np.diag(sample_cov)
            std = np.sqrt(var)
            corr = (sample_cov / np.outer(std, std)).copy()
            np.fill_diagonal(corr, 1.0)
            avg_corr = (corr.sum() - p) / (p * (p - 1))
            target = (np.outer(std, std) * avg_corr).copy()
            np.fill_diagonal(target, var)

        # Compute optimal shrinkage intensity (simplified)
        delta = sample_cov - target
        delta_sq_sum = np.sum(delta ** 2)
        # Approximate shrinkage intensity
        alpha = min(1.0, max(0.0, delta_sq_sum / (n_obs * np.sum(sample_cov ** 2))))
        # Clamp to reasonable range
        alpha = min(alpha, 0.5)

        shrunk = (1 - alpha) * sample_cov + alpha * target
        return shrunk
