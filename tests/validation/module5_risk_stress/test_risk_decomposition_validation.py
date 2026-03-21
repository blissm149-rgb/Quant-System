"""Test risk decomposition consistency.

Validates that factor and idiosyncratic variance components sum
correctly and that per-factor contributions are self-consistent.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.portfolio.factor_risk_model.risk_decomposition import (
    RiskDecomposition,
)


pytestmark = [pytest.mark.validation]


def _make_risk_inputs():
    """Create synthetic risk model inputs with known properties."""
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    factors = ["market", "momentum", "value"]

    rng = np.random.default_rng(42)

    weights = pd.Series([0.10, 0.08, -0.05, 0.07, -0.06], index=tickers)

    exposures = pd.DataFrame(
        rng.normal(0, 0.5, (len(tickers), len(factors))),
        index=tickers,
        columns=factors,
    )

    # Create a valid PSD factor covariance matrix
    raw = rng.normal(0, 0.01, (len(factors), len(factors)))
    factor_cov = pd.DataFrame(
        raw @ raw.T + np.eye(len(factors)) * 0.001,
        index=factors,
        columns=factors,
    )

    idio_var = pd.Series(
        rng.uniform(0.0001, 0.001, len(tickers)), index=tickers
    )

    return weights, exposures, factor_cov, idio_var


class TestRiskDecompositionValidation:
    """Verify risk decomposition produces consistent results."""

    def test_total_variance_equals_factor_plus_idiosyncratic(self):
        """Total variance must equal factor_variance + idiosyncratic_variance."""
        weights, exposures, factor_cov, idio_var = _make_risk_inputs()
        decomp = RiskDecomposition()

        result = decomp.decompose(weights, exposures, factor_cov, idio_var)

        expected_total = result["factor_variance"] + result["idiosyncratic_variance"]
        assert abs(result["total_variance"] - expected_total) < 1e-12, (
            f"Total variance ({result['total_variance']:.10f}) should equal "
            f"factor ({result['factor_variance']:.10f}) + idio "
            f"({result['idiosyncratic_variance']:.10f}) = {expected_total:.10f}"
        )

    def test_all_variance_components_non_negative(self):
        """All variance components should be non-negative (by construction)."""
        weights, exposures, factor_cov, idio_var = _make_risk_inputs()
        decomp = RiskDecomposition()

        result = decomp.decompose(weights, exposures, factor_cov, idio_var)

        assert result["total_variance"] >= 0, "Total variance should be >= 0"
        assert result["factor_variance"] >= -1e-12, "Factor variance should be >= 0"
        assert result["idiosyncratic_variance"] >= 0, "Idio variance should be >= 0"

    def test_factor_contributions_sum_to_factor_variance(self):
        """Per-factor marginal contributions should sum approximately
        to total factor variance."""
        weights, exposures, factor_cov, idio_var = _make_risk_inputs()
        decomp = RiskDecomposition()

        result = decomp.decompose(weights, exposures, factor_cov, idio_var)

        contrib_sum = sum(result["factor_contributions"].values())
        factor_var = result["factor_variance"]

        assert abs(contrib_sum - factor_var) < 1e-10, (
            f"Factor contributions sum ({contrib_sum:.10f}) should equal "
            f"factor variance ({factor_var:.10f})"
        )
