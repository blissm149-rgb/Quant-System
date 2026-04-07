"""Test bootstrap confidence interval estimation.

Validates that bootstrap CI and block bootstrap produce correct
statistical properties on known distributions.
"""

import pytest
import numpy as np
import pandas as pd

from tests.validation.shared.metrics import (
    annualized_sharpe,
    bootstrap_ci,
    block_bootstrap,
)


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestBootstrapConfidence:
    """Verify bootstrap CI calibration and block bootstrap consistency."""

    def test_bootstrap_ci_covers_sample_mean(self):
        """95% bootstrap CI should contain the sample mean (by construction,
        since bootstrap resamples from the data)."""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(0.05, 0.5, 2000))
        sample_mean = float(data.mean())

        lower, upper = bootstrap_ci(
            lambda x: float(x.mean()), data, n_bootstrap=1000, ci=0.95
        )

        assert lower < sample_mean < upper, (
            f"95% CI [{lower:.4f}, {upper:.4f}] should contain "
            f"sample mean {sample_mean:.4f}"
        )

    def test_wider_ci_with_higher_confidence(self):
        """99% CI should be wider than 90% CI on the same data."""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(0, 1, 500))

        lo_90, hi_90 = bootstrap_ci(
            lambda x: float(x.mean()), data, n_bootstrap=500, ci=0.90
        )
        lo_99, hi_99 = bootstrap_ci(
            lambda x: float(x.mean()), data, n_bootstrap=500, ci=0.99
        )

        width_90 = hi_90 - lo_90
        width_99 = hi_99 - lo_99

        assert width_99 > width_90, (
            f"99% CI width ({width_99:.4f}) should exceed "
            f"90% CI width ({width_90:.4f})"
        )

    def test_block_bootstrap_preserves_autocorrelation(self):
        """Block bootstrap with large blocks should preserve more temporal
        structure than iid bootstrap (blocks=1)."""
        rng = np.random.default_rng(42)
        # AR(1) process with autocorrelation
        n = 500
        data = np.zeros(n)
        for i in range(1, n):
            data[i] = 0.7 * data[i - 1] + rng.normal(0, 0.01)
        series = pd.Series(data)

        # Block bootstrap with block_size=20 (preserves autocorrelation)
        block_results = block_bootstrap(
            series, block_size=20, n_bootstrap=200,
            statistic_fn=lambda x: float(x.autocorr(lag=1)),
            seed=42,
        )

        # IID bootstrap (block_size=1, destroys autocorrelation)
        iid_results = block_bootstrap(
            series, block_size=1, n_bootstrap=200,
            statistic_fn=lambda x: float(x.autocorr(lag=1)),
            seed=42,
        )

        mean_block_ac = np.mean(block_results)
        mean_iid_ac = np.mean(iid_results)

        assert mean_block_ac > mean_iid_ac, (
            f"Block bootstrap mean autocorr ({mean_block_ac:.4f}) should "
            f"exceed iid bootstrap ({mean_iid_ac:.4f})"
        )

    def test_bootstrap_sharpe_ci_reasonable_width(self):
        """Bootstrap CI for annualized Sharpe should have a width between
        0.1 and 5.0 for a 500-day strategy with moderate Sharpe."""
        rng = np.random.default_rng(42)
        daily_rets = pd.Series(rng.normal(0.0003, 0.01, 500))

        lower, upper = bootstrap_ci(
            annualized_sharpe, daily_rets, n_bootstrap=500, ci=0.95
        )

        width = upper - lower
        assert 0.1 < width < 5.0, (
            f"Sharpe CI width ({width:.2f}) should be between 0.1 and 5.0 "
            f"for a 500-day sample"
        )
