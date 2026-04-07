"""Test parameter sensitivity of factor models.

Validates that small perturbations to strategy parameters do not cause
catastrophic changes in output, and that strategies degrade gracefully.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine
from quant_fund.research_algorithms.factor_models.momentum_factor import MomentumFactor
from quant_fund.research_algorithms.factor_models.low_volatility_factor import (
    LowVolatilityFactor,
)
from quant_fund.research_algorithms.mean_reversion.zscore_reversion_strategy import (
    ZScoreReversionStrategy,
)
from quant_fund.research_algorithms.mean_reversion.short_term_reversal_strategy import (
    ShortTermReversalStrategy,
)
from tests.conftest import make_ohlcv, STANDARD_TICKERS
from tests.validation.shared.perturbation_engine import PerturbationEngine


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


def _aligned_data(ohlcv, as_of_idx=400, lookback=300):
    """Helper to get pre-filtered aligned data."""
    dates = ohlcv.index.get_level_values(0).unique().sort_values()
    as_of = dates[as_of_idx]
    safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]
    engine = DataAlignmentEngine()
    return engine.get_aligned_data(safe, as_of, lookback), as_of


class TestParameterSensitivity:
    """Verify strategies are not brittle to small parameter changes."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=500, seed=42)
        self.aligned, self.as_of = _aligned_data(self.ohlcv)

    def test_momentum_window_perturbation(self):
        """Varying momentum_long_window by +/- 20% should produce rank-correlated
        outputs (Spearman > 0.5)."""
        base_cfg = {"momentum_long_window": 252, "momentum_skip_window": 21}
        base = MomentumFactor(config=base_cfg)
        base_scores = base.compute(self.aligned, self.as_of)

        pe = PerturbationEngine()
        windows = pe.perturb_parameter(252, pct_range=0.20, n_samples=10, seed=42)

        rank_corrs = []
        for w in windows:
            cfg = {"momentum_long_window": int(w), "momentum_skip_window": 21}
            variant = MomentumFactor(config=cfg)
            variant_scores = variant.compute(self.aligned, self.as_of)

            merged = pd.concat([base_scores, variant_scores], axis=1).dropna()
            if len(merged) >= 3:
                corr = merged.iloc[:, 0].corr(merged.iloc[:, 1], method="spearman")
                rank_corrs.append(corr)

        assert len(rank_corrs) > 0, "Need at least one valid correlation"
        mean_corr = np.mean(rank_corrs)
        assert mean_corr > 0.5, (
            f"Mean rank correlation across momentum window perturbations = "
            f"{mean_corr:.3f}, expected > 0.5"
        )

    def test_reversal_window_perturbation(self):
        """Varying reversal return_window by +/- 40% should still produce
        correlated outputs."""
        base = ZScoreReversionStrategy(
            config={"reversion_return_window": 5, "reversion_zscore_window": 60}
        )
        base_scores = base.compute(self.aligned, self.as_of)

        pe = PerturbationEngine()
        windows = pe.perturb_parameter(5, pct_range=0.40, n_samples=8, seed=42)

        rank_corrs = []
        for w in windows:
            w_int = max(2, int(w))
            variant = ZScoreReversionStrategy(
                config={"reversion_return_window": w_int, "reversion_zscore_window": 60}
            )
            variant_scores = variant.compute(self.aligned, self.as_of)

            merged = pd.concat([base_scores, variant_scores], axis=1).dropna()
            if len(merged) >= 3:
                corr = merged.iloc[:, 0].corr(merged.iloc[:, 1], method="spearman")
                rank_corrs.append(corr)

        assert len(rank_corrs) > 0
        mean_corr = np.mean(rank_corrs)
        assert mean_corr > 0.3, (
            f"Mean rank correlation across reversal window perturbations = "
            f"{mean_corr:.3f}, expected > 0.3"
        )

    def test_low_vol_window_perturbation(self):
        """LowVolatilityFactor with vol windows 200-300 should produce highly
        correlated scores (Spearman > 0.7) since underlying vol is similar."""
        base = LowVolatilityFactor(config={"low_vol_window": 252})
        base_scores = base.compute(self.aligned, self.as_of)

        rank_corrs = []
        for w in [200, 220, 240, 260, 280]:
            variant = LowVolatilityFactor(config={"low_vol_window": w})
            variant_scores = variant.compute(self.aligned, self.as_of)

            merged = pd.concat([base_scores, variant_scores], axis=1).dropna()
            if len(merged) >= 3:
                corr = merged.iloc[:, 0].corr(merged.iloc[:, 1], method="spearman")
                rank_corrs.append(corr)

        assert len(rank_corrs) > 0
        mean_corr = np.mean(rank_corrs)
        assert mean_corr > 0.7, (
            f"Mean rank correlation for low-vol window perturbation = "
            f"{mean_corr:.3f}, expected > 0.7"
        )

    def test_no_strategy_returns_nan_for_all_tickers(self):
        """No factor model should return all-NaN output when given clean,
        sufficiently long data."""
        strategies = [
            MomentumFactor(),
            LowVolatilityFactor(),
            ZScoreReversionStrategy(),
            ShortTermReversalStrategy(),
        ]

        for strategy in strategies:
            scores = strategy.compute(self.aligned, self.as_of)
            non_nan_pct = scores.notna().mean()
            assert non_nan_pct > 0.5, (
                f"{strategy.feature_name}: only {non_nan_pct:.0%} non-NaN, "
                f"expected > 50% on clean 500-day data"
            )
