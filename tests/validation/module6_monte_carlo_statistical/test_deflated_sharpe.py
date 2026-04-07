"""Test deflated Sharpe ratio implementation.

Validates that the DSR correctly penalizes multiple testing and
that its output is calibrated relative to the number of trials.
"""

import pytest
import numpy as np

from tests.validation.shared.metrics import deflated_sharpe_ratio


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestDeflatedSharpe:
    """Verify deflated Sharpe ratio properties."""

    def test_dsr_decreases_with_more_trials(self):
        """DSR should decrease as the number of tested strategies increases
        (more multiple testing → higher haircut)."""
        sharpe = 1.5
        T = 252 * 3  # 3 years
        skew = 0.0
        kurtosis = 3.0

        dsr_10 = deflated_sharpe_ratio(sharpe, n_trials=10, skew=skew,
                                        kurtosis=kurtosis, T=T)
        dsr_100 = deflated_sharpe_ratio(sharpe, n_trials=100, skew=skew,
                                         kurtosis=kurtosis, T=T)
        dsr_1000 = deflated_sharpe_ratio(sharpe, n_trials=1000, skew=skew,
                                          kurtosis=kurtosis, T=T)

        assert dsr_10 > dsr_100 > dsr_1000, (
            f"DSR should decrease with more trials: "
            f"10 trials={dsr_10:.4f}, 100={dsr_100:.4f}, 1000={dsr_1000:.4f}"
        )

    def test_dsr_increases_with_sharpe(self):
        """Higher observed Sharpe should produce higher DSR (more likely
        to be genuine skill)."""
        T = 252 * 3
        n_trials = 50
        skew = 0.0
        kurtosis = 3.0

        dsr_low = deflated_sharpe_ratio(0.5, n_trials, skew, kurtosis, T)
        dsr_high = deflated_sharpe_ratio(2.0, n_trials, skew, kurtosis, T)

        assert dsr_high > dsr_low, (
            f"Higher Sharpe should yield higher DSR: "
            f"Sharpe=0.5 → {dsr_low:.4f}, Sharpe=2.0 → {dsr_high:.4f}"
        )

    def test_dsr_output_bounded_zero_one(self):
        """DSR (a probability) should always be in [0, 1]."""
        test_cases = [
            (0.5, 10, 0.0, 3.0, 252),
            (2.0, 100, -0.5, 4.0, 756),
            (0.1, 1000, 0.3, 3.5, 126),
            (3.0, 5, 0.0, 3.0, 1260),
        ]

        for sharpe, n, skew, kurt, T in test_cases:
            dsr = deflated_sharpe_ratio(sharpe, n, skew, kurt, T)
            assert 0.0 <= dsr <= 1.0, (
                f"DSR should be in [0,1], got {dsr:.4f} for "
                f"sharpe={sharpe}, n_trials={n}, T={T}"
            )
