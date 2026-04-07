"""Test data leakage pathways in the backtest pipeline.

Validates that no information from the future contaminates feature
computation, normalisation, or corporate action adjustments.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine
from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer
from quant_fund.data_layer.corporate_action_adjuster import (
    CorporateActionAdjuster,
    CorporateAction,
)
from tests.conftest import make_ohlcv, STANDARD_TICKERS


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestDataLeakage:
    """Verify that no information leaks from the future into feature computation."""

    def test_normalizer_uses_only_current_cross_section(self):
        """Z-score normalisation must use only the cross-section at the
        current rebalance date, not future cross-sections."""
        normalizer = FeatureNormalizer()

        rng = np.random.default_rng(42)
        tickers = STANDARD_TICKERS[:10]

        scores_t1 = pd.Series(rng.normal(0, 1, len(tickers)), index=tickers)
        scores_t2 = pd.Series(
            rng.normal(5, 2, len(tickers)), index=tickers
        )

        # Normalise t1 in isolation
        norm_t1_solo = normalizer.normalize(scores_t1)

        # Normalise t1 as part of a DataFrame containing t2
        combined = pd.DataFrame({"t1": scores_t1, "t2": scores_t2})
        norm_combined = normalizer.normalize_dataframe(combined)

        # Normalised values for t1 should be identical
        pd.testing.assert_series_equal(
            norm_t1_solo,
            norm_combined["t1"],
            check_names=False,
            atol=1e-10,
            obj="Z-score normalisation must be independent per cross-section",
        )

    def test_corporate_actions_applied_backward_only(self):
        """Corporate action adjustments must only modify prices BEFORE the
        ex_date, not after."""
        adjuster = CorporateActionAdjuster()
        tickers = ["AAPL"]
        ohlcv = make_ohlcv(tickers=tickers, periods=200, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()

        ex_date = dates[100]
        split_factor = 0.5  # 2-for-1 split

        action = CorporateAction(
            ticker="AAPL",
            ex_date=ex_date,
            action_type="split",
            adjustment_factor=split_factor,
        )
        adjuster.register_actions([action])

        pre_split_mask = ohlcv.index.get_level_values(0) < ex_date
        post_split_mask = ohlcv.index.get_level_values(0) >= ex_date

        original_pre = ohlcv.loc[pre_split_mask, "close"].copy()
        original_post = ohlcv.loc[post_split_mask, "close"].copy()

        adjusted = adjuster.adjust_dataframe(ohlcv)
        adjusted_pre = adjusted.loc[pre_split_mask, "close"]
        adjusted_post = adjusted.loc[post_split_mask, "close"]

        # Pre-split prices should be adjusted
        assert not np.allclose(adjusted_pre.values, original_pre.values), (
            "Pre-split prices should be adjusted"
        )

        # Post-split prices should remain unchanged
        np.testing.assert_array_almost_equal(
            adjusted_post.values,
            original_post.values,
            decimal=6,
            err_msg="Post-split prices must NOT be modified",
        )

    def test_feature_scores_invariant_to_future_data_truncation(self):
        """Scores at T must be identical whether we pass data up to T+50
        (using permissive alignment) or exactly up to T."""
        engine = DataAlignmentEngine()
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=400, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()

        from quant_fund.research_algorithms.factor_models.momentum_factor import (
            MomentumFactor,
        )

        strategy = MomentumFactor()
        as_of = dates[300]
        lookback = strategy.lookback_days

        # Approach 1: pre-filter to < as_of, then align (strict)
        safe = ohlcv[ohlcv.index.get_level_values(0) < as_of]
        aligned_strict = engine.get_aligned_data(safe, as_of, lookback)
        scores_strict = strategy.compute(aligned_strict, as_of)

        # Approach 2: use permissive alignment on full dataset
        aligned_permissive = engine.get_aligned_data_permissive(ohlcv, as_of, lookback)
        scores_permissive = strategy.compute(aligned_permissive, as_of)

        # Both should produce identical scores
        pd.testing.assert_series_equal(
            scores_strict.sort_index(),
            scores_permissive.sort_index(),
            check_names=False,
            atol=1e-10,
            obj="Feature scores at T must not depend on data after T",
        )
