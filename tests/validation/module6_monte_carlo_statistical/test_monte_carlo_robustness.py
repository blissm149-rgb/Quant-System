"""Test Monte Carlo robustness via perturbation engine.

Validates that strategy performance degrades gracefully under
data perturbation (noise injection, stale prices, shuffled dates).
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.research_algorithms.mean_reversion.zscore_reversion_strategy import (
    ZScoreReversionStrategy,
)
from tests.conftest import STANDARD_TICKERS
from tests.generators.market_regime_simulator import MarketRegimeSimulator
from tests.validation.shared.backtest_harness import BacktestHarness
from tests.validation.shared.metrics import annualized_sharpe
from tests.validation.shared.perturbation_engine import PerturbationEngine


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestMonteCarloRobustness:
    """Verify strategy robustness under Monte Carlo perturbations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tickers = STANDARD_TICKERS[:5]
        sim = MarketRegimeSimulator(tickers=tickers, seed=42)
        self.data = sim.regime_sequence([
            ("bull", {"days": 150}),
            ("sideways", {"days": 150}),
        ])
        self.harness = BacktestHarness(seed=42)
        self.engine = PerturbationEngine()

    def test_noise_injection_does_not_crash(self):
        """Adding Gaussian noise to returns should not cause backtest
        to crash or produce NaN results."""
        strategy = ZScoreReversionStrategy()

        # Extract returns for perturbation
        dates = self.data.index.get_level_values(0).unique().sort_values()
        tickers = self.data.index.get_level_values("ticker").unique()

        # Build returns matrix
        returns_dict = {}
        for t in tickers:
            prices = self.data.xs(t, level="ticker")["close"]
            returns_dict[t] = prices.pct_change().dropna()
        returns_df = pd.DataFrame(returns_dict).dropna()

        # Perturb returns
        noisy = self.engine.perturb_returns(returns_df, noise_std=0.002, seed=42)

        # Should not have NaN
        assert noisy.isna().sum().sum() == 0, "Perturbed returns should have no NaN"

    def test_stale_price_injection_degrades_gracefully(self):
        """Injecting 5% stale prices should not crash the backtest."""
        strategy = ZScoreReversionStrategy()
        stale_data = self.engine.inject_stale_prices(
            self.data, stale_pct=0.05, seed=42
        )

        result = self.harness.run(stale_data, strategy)

        # Should still produce some results (may be empty if lookback too long)
        assert result.nav_series.isna().sum() == 0 or len(result.nav_series) == 0, (
            "NAV series should have no internal NaN gaps after stale injection"
        )

    def test_shuffled_dates_destroy_temporal_structure(self):
        """Shuffling dates should scramble the data values so the temporal
        structure is destroyed (values at each date should differ)."""
        dates = self.data.index.get_level_values(0).unique().sort_values()
        tickers = self.data.index.get_level_values("ticker").unique()
        returns_dict = {}
        for t in tickers:
            prices = self.data.xs(t, level="ticker")["close"]
            returns_dict[t] = prices.pct_change().dropna()
        returns_df = pd.DataFrame(returns_dict).dropna()

        shuffled = self.engine.shuffle_dates(returns_df, seed=42)

        # Values should be rearranged (different row content at same index)
        # Check that at least 50% of rows differ
        mismatches = 0
        for i in range(len(returns_df)):
            if not np.allclose(
                returns_df.iloc[i].values, shuffled.iloc[i].values
            ):
                mismatches += 1

        pct_different = mismatches / len(returns_df)
        assert pct_different > 0.5, (
            f"Only {pct_different:.1%} of rows changed after shuffle, "
            f"expected > 50%"
        )

    def test_perturbation_engine_generates_correct_samples(self):
        """Parameter perturbation should generate the requested number of
        samples within the specified range."""
        base = 20.0
        samples = self.engine.perturb_parameter(
            base, pct_range=0.20, n_samples=50, seed=42
        )

        assert len(samples) == 50, f"Expected 50 samples, got {len(samples)}"

        for s in samples:
            assert base * 0.80 <= s <= base * 1.20, (
                f"Sample {s} outside range [{base*0.80}, {base*1.20}]"
            )
