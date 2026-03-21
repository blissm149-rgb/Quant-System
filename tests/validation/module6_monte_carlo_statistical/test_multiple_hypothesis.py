"""Test multiple hypothesis correction methods.

Validates Bonferroni and Benjamini-Hochberg FDR corrections produce
correct rejection decisions relative to known p-value distributions.
"""

import pytest
import numpy as np

from tests.validation.shared.metrics import (
    bonferroni_correction,
    benjamini_hochberg,
)


pytestmark = [pytest.mark.validation]


class TestMultipleHypothesis:
    """Verify multiple testing correction methods."""

    def test_bonferroni_rejects_only_strong_signals(self):
        """Bonferroni should only reject p-values below alpha/m."""
        p_values = [0.001, 0.01, 0.03, 0.10, 0.50]
        alpha = 0.05

        results = bonferroni_correction(p_values, alpha)
        threshold = alpha / len(p_values)  # 0.01

        for i, (p, rejected) in enumerate(zip(p_values, results)):
            if p < threshold:
                assert rejected, f"p={p} < {threshold} should be rejected"
            else:
                assert not rejected, f"p={p} >= {threshold} should not be rejected"

    def test_bonferroni_more_conservative_than_bh(self):
        """Bonferroni should reject fewer or equal hypotheses than BH
        at the same significance level."""
        rng = np.random.default_rng(42)
        # Mix of truly significant and null p-values
        p_values = list(rng.uniform(0, 0.02, 5)) + list(rng.uniform(0.1, 1.0, 15))

        bonf_rejected = sum(bonferroni_correction(p_values, alpha=0.05))
        bh_rejected = sum(benjamini_hochberg(p_values, fdr=0.05))

        assert bonf_rejected <= bh_rejected, (
            f"Bonferroni rejections ({bonf_rejected}) should be <= "
            f"BH rejections ({bh_rejected})"
        )

    def test_bh_controls_false_discovery_rate(self):
        """Under the global null (all p-values from uniform), BH at FDR=0.10
        should reject at most ~10% of hypotheses on average."""
        rng = np.random.default_rng(42)

        rejection_rates = []
        for trial in range(100):
            p_values = list(rng.uniform(0, 1, 20))
            rejected = benjamini_hochberg(p_values, fdr=0.10)
            rejection_rates.append(sum(rejected) / len(p_values))

        mean_rate = np.mean(rejection_rates)
        # Under global null, FDR should be controlled at 10%
        # Allow generous tolerance for finite-sample variability
        assert mean_rate < 0.20, (
            f"Mean rejection rate under null ({mean_rate:.3f}) should be "
            f"well below 20% with FDR=0.10"
        )

    def test_no_rejections_with_high_p_values(self):
        """When all p-values are high, neither method should reject any."""
        p_values = [0.30, 0.45, 0.60, 0.80, 0.95]

        bonf = bonferroni_correction(p_values, alpha=0.05)
        bh = benjamini_hochberg(p_values, fdr=0.10)

        assert not any(bonf), "No Bonferroni rejections expected with high p-values"
        assert not any(bh), "No BH rejections expected with high p-values"
