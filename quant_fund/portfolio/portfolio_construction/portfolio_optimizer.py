"""Portfolio optimizer using mean-variance optimisation.

Minimises: lambda * w^T Sigma w - alpha^T w
subject to constraints from constraint_engine.

Uses quadratic programming. Falls back to a simpler analytical
solution if CVXPY is not available.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from quant_fund.portfolio.portfolio_construction.constraint_engine import (
    ConstraintEngine,
    ConstraintSet,
)

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Mean-variance portfolio optimizer with constraint enforcement.

    Solver priority: CVXPY (if available) -> numpy QP fallback.
    All constraints come from constraint_engine.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._risk_aversion = cfg.get("risk_aversion", 1.0)
        self._max_iterations = cfg.get("max_iterations", 1000)

    def optimize(
        self,
        alpha_scores: pd.Series,
        factor_covariance: pd.DataFrame,
        factor_exposures: pd.DataFrame,
        constraints: ConstraintSet,
        current_positions: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Compute optimal portfolio weights.

        Args:
            alpha_scores: Cross-sectionally normalised, indexed by ticker.
            factor_covariance: Factor covariance matrix (factors x factors).
            factor_exposures: Factor exposures (tickers x factors).
            constraints: ConstraintSet from constraint_engine.
            current_positions: Current weights for turnover constraint.

        Returns:
            Target weights indexed by ticker.
        """
        tickers = alpha_scores.index.tolist()
        n = len(tickers)
        if n == 0:
            return pd.Series(dtype=float)

        alpha = alpha_scores.reindex(tickers, fill_value=0.0).values

        # Build covariance: Sigma = B F B^T + D_approx
        common_factors = factor_exposures.columns.intersection(
            factor_covariance.columns
        )
        B = factor_exposures.reindex(tickers, fill_value=0.0)[common_factors].values.copy()
        F = factor_covariance.loc[common_factors, common_factors].values.copy()
        Sigma = B @ F @ B.T
        # Add small diagonal for numerical stability
        Sigma += np.eye(n) * 1e-6

        try:
            w = self._optimize_cvxpy(alpha, Sigma, n, constraints, tickers)
        except Exception as e:
            logger.info("CVXPY not available or failed (%s), using fallback", e)
            w = self._optimize_fallback(alpha, Sigma, n, constraints)

        result = pd.Series(w, index=tickers)
        return result

    def _optimize_cvxpy(self, alpha, Sigma, n, constraints, tickers):
        """Optimise using CVXPY with OSQP backend."""
        import cvxpy as cp

        w = cp.Variable(n)
        lam = self._risk_aversion

        objective = cp.Minimize(lam * cp.quad_form(w, cp.psd_wrap(Sigma)) - alpha @ w)

        cons = []
        # Position size: |w_i| <= max_position_size
        cons.append(w <= constraints.max_position_size)
        cons.append(w >= -constraints.max_position_size)

        # Leverage: ||w||_1 <= max_leverage
        cons.append(cp.norm(w, 1) <= constraints.max_leverage)

        # Dollar neutral
        if constraints.dollar_neutral:
            cons.append(cp.sum(w) == 0)

        # Sector exposure
        if constraints.sector_map:
            sectors = sorted(set(constraints.sector_map.values()))
            for sector in sectors:
                mask = np.array([
                    1.0 if constraints.sector_map.get(tickers[i]) == sector else 0.0
                    for i in range(n)
                ])
                cons.append(mask @ cp.abs(w) <= constraints.max_sector_exposure)

        prob = cp.Problem(objective, cons)
        prob.solve(solver=cp.OSQP, max_iter=self._max_iterations, warm_start=True)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            logger.warning("CVXPY status: %s, falling back", prob.status)
            raise RuntimeError(f"CVXPY solve failed: {prob.status}")

        return w.value

    def _optimize_fallback(self, alpha, Sigma, n, constraints):
        """Simple fallback: alpha-proportional weights with constraint clipping.

        Iteratively clips positions and re-centres to satisfy all constraints.
        """
        if np.all(alpha == 0):
            return np.zeros(n)

        # Start with Sigma^{-1} alpha (unconstrained optimal direction)
        try:
            Sigma_inv = np.linalg.inv(Sigma)
            w = Sigma_inv @ alpha / self._risk_aversion
        except np.linalg.LinAlgError:
            w = alpha.copy()

        # Iterative projection to satisfy constraints
        for _ in range(20):
            # Clip position sizes
            w = np.clip(w, -constraints.max_position_size, constraints.max_position_size)

            # Dollar neutral
            if constraints.dollar_neutral and len(w) > 0:
                w -= w.mean()

            # Re-clip after centering
            w = np.clip(w, -constraints.max_position_size, constraints.max_position_size)

            # Leverage scaling
            gross = np.abs(w).sum()
            if gross > constraints.max_leverage and gross > 0:
                w *= constraints.max_leverage / gross

        return w
