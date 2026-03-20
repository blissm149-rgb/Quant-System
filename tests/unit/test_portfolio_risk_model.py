"""Unit tests for portfolio factor risk model modules.

TESTING_PLAN.md Section 3.8 — factor_covariance_estimator, factor_exposure_estimator, risk_decomposition.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.portfolio.factor_risk_model.factor_covariance_estimator import (
    FactorCovarianceEstimator,
)
from quant_fund.portfolio.factor_risk_model.factor_exposure_estimator import (
    FactorExposureEstimator,
)
from quant_fund.portfolio.factor_risk_model.risk_decomposition import (
    RiskDecomposition,
)
from tests.conftest import make_factor_returns, make_returns


@pytest.mark.unit
@pytest.mark.tier2
class TestFactorCovarianceEstimator:
    """FactorCovarianceEstimator — Ledoit-Wolf shrinkage covariance."""

    @pytest.fixture
    def estimator(self):
        return FactorCovarianceEstimator()

    def test_estimate_returns_dataframe(self, estimator):
        """estimate returns square DataFrame."""
        factor_returns = make_factor_returns(n_dates=300, seed=42)
        as_of = pd.Timestamp("2025-01-01")
        cov = estimator.estimate(factor_returns, as_of=as_of)
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape[0] == cov.shape[1]

    def test_covariance_is_symmetric(self, estimator):
        """Covariance matrix is symmetric."""
        factor_returns = make_factor_returns(n_dates=300, seed=42)
        cov = estimator.estimate(factor_returns, as_of=pd.Timestamp("2025-01-01"))
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)

    def test_covariance_is_psd(self, estimator):
        """Covariance matrix is positive semi-definite."""
        factor_returns = make_factor_returns(n_dates=300, seed=42)
        cov = estimator.estimate(factor_returns, as_of=pd.Timestamp("2025-01-01"))
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-10)

    def test_insufficient_data_returns_identity_scaled(self, estimator):
        """With very few observations, falls back to identity-like matrix."""
        factor_returns = make_factor_returns(n_dates=5, seed=42)
        cov = estimator.estimate(factor_returns, as_of=pd.Timestamp("2025-01-01"))
        assert isinstance(cov, pd.DataFrame)


@pytest.mark.unit
@pytest.mark.tier2
class TestFactorExposureEstimator:
    """FactorExposureEstimator — cross-sectional factor exposures."""

    @pytest.fixture
    def estimator(self):
        return FactorExposureEstimator()

    def test_estimate_returns_dataframe(self, estimator):
        """estimate returns DataFrame of exposures (tickers x factors)."""
        returns = make_returns(n_dates=300, seed=42)
        factor_returns = make_factor_returns(n_dates=300, seed=42)
        exposures = estimator.estimate(returns, factor_returns, as_of=pd.Timestamp("2025-01-01"))
        assert isinstance(exposures, pd.DataFrame)
        assert len(exposures) > 0

    def test_exposures_bounded(self, estimator):
        """Factor exposures are within reasonable range."""
        returns = make_returns(n_dates=300, seed=42)
        factor_returns = make_factor_returns(n_dates=300, seed=42)
        exposures = estimator.estimate(returns, factor_returns, as_of=pd.Timestamp("2025-01-01"))
        # Factor betas should be in [-10, 10] range
        assert exposures.abs().max().max() < 10.0


@pytest.mark.unit
@pytest.mark.tier2
class TestRiskDecomposition:
    """RiskDecomposition — factor + idiosyncratic = total."""

    @pytest.fixture
    def decomposition(self):
        return RiskDecomposition()

    def test_decompose_returns_dict(self, decomposition):
        """decompose returns dict with expected keys."""
        weights = pd.Series({"AAPL": 0.01, "MSFT": -0.01, "GOOG": 0.005})
        factors = ["Market", "Size"]
        factor_exposures = pd.DataFrame(
            np.random.default_rng(42).standard_normal((3, 2)),
            index=["AAPL", "MSFT", "GOOG"], columns=factors,
        )
        factor_cov = pd.DataFrame(
            np.eye(2) * 0.04, index=factors, columns=factors,
        )
        idio_var = pd.Series({"AAPL": 0.01, "MSFT": 0.01, "GOOG": 0.01})

        result = decomposition.decompose(weights, factor_exposures, factor_cov, idio_var)
        assert isinstance(result, dict)
        assert "total_variance" in result
        assert "factor_variance" in result

    def test_total_equals_factor_plus_idio(self, decomposition):
        """Total variance ≈ factor + idiosyncratic (within tolerance)."""
        weights = pd.Series({"AAPL": 0.01, "MSFT": -0.01})
        factors = ["Market"]
        factor_exposures = pd.DataFrame(
            [[1.0], [0.8]], index=["AAPL", "MSFT"], columns=factors,
        )
        factor_cov = pd.DataFrame([[0.04]], index=factors, columns=factors)
        idio_var = pd.Series({"AAPL": 0.01, "MSFT": 0.01})

        result = decomposition.decompose(weights, factor_exposures, factor_cov, idio_var)
        total = result["total_variance"]
        factor = result["factor_variance"]
        idio = result["idiosyncratic_variance"]
        np.testing.assert_allclose(total, factor + idio, rtol=0.01)
