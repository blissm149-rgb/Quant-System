"""Risk decomposition into factor and idiosyncratic components.

Decomposes portfolio variance and risk contributions for attribution
and exposure monitoring.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskDecomposition:
    """Decomposes portfolio risk into factor and idiosyncratic components.

    Total variance = factor variance + idiosyncratic variance
    Factor variance = w^T B F B^T w
    Idiosyncratic variance = w^T D w
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}

    def decompose(
        self,
        weights: pd.Series,
        factor_exposures: pd.DataFrame,
        factor_covariance: pd.DataFrame,
        idiosyncratic_variance: pd.Series,
    ) -> Dict[str, float]:
        """Decompose portfolio variance into components.

        Args:
            weights: Portfolio weights indexed by ticker.
            factor_exposures: DataFrame (tickers x factors).
            factor_covariance: DataFrame (factors x factors).
            idiosyncratic_variance: Series of per-stock variances.

        Returns:
            Dict with total_variance, factor_variance, idiosyncratic_variance,
            and per-factor contributions.
        """
        tickers = weights.index
        w = weights.reindex(tickers, fill_value=0.0).values

        common_factors = factor_exposures.columns.intersection(
            factor_covariance.columns
        )
        B = factor_exposures.reindex(tickers, fill_value=0.0)[common_factors].values
        F_cov = factor_covariance.loc[common_factors, common_factors].values
        D = np.diag(
            idiosyncratic_variance.reindex(tickers, fill_value=0.0).values
        )

        # Portfolio factor exposure: B^T w
        portfolio_factor_exposure = B.T @ w

        # Factor variance: w^T B F B^T w = (B^T w)^T F (B^T w)
        factor_var = float(
            portfolio_factor_exposure @ F_cov @ portfolio_factor_exposure
        )

        # Idiosyncratic variance: w^T D w
        idio_var = float(w @ D @ w)

        total_var = factor_var + idio_var

        # Per-factor marginal contribution
        factor_contributions = {}
        for i, factor in enumerate(common_factors):
            # Marginal contribution of factor i
            e_i = np.zeros(len(common_factors))
            e_i[i] = portfolio_factor_exposure[i]
            contrib = float(e_i @ F_cov @ portfolio_factor_exposure)
            factor_contributions[factor] = contrib

        return {
            "total_variance": total_var,
            "total_volatility": np.sqrt(max(0, total_var)),
            "factor_variance": factor_var,
            "factor_volatility": np.sqrt(max(0, factor_var)),
            "idiosyncratic_variance": idio_var,
            "idiosyncratic_volatility": np.sqrt(max(0, idio_var)),
            "factor_pct": factor_var / total_var if total_var > 0 else 0.0,
            "factor_contributions": factor_contributions,
            "portfolio_factor_exposures": dict(
                zip(common_factors, portfolio_factor_exposure)
            ),
        }

    def top_risk_contributors(
        self,
        weights: pd.Series,
        full_covariance: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.Series:
        """Find the top individual stock risk contributors.

        Marginal contribution to risk (MCTR) = (Sigma w)_i / sigma_p

        Args:
            weights: Portfolio weights.
            full_covariance: Full stock covariance matrix.
            top_n: Number of top contributors to return.

        Returns:
            Series of risk contributions indexed by ticker, sorted descending.
        """
        tickers = weights.index
        w = weights.reindex(tickers, fill_value=0.0).values
        Sigma = full_covariance.reindex(
            index=tickers, columns=tickers, fill_value=0.0
        ).values

        portfolio_var = w @ Sigma @ w
        if portfolio_var <= 0:
            return pd.Series(dtype=float)

        sigma_p = np.sqrt(portfolio_var)

        # MCTR for each stock
        mctr = (Sigma @ w) / sigma_p
        # Risk contribution = w_i * MCTR_i
        risk_contrib = w * mctr

        result = pd.Series(risk_contrib, index=tickers)
        return result.abs().nlargest(top_n)
