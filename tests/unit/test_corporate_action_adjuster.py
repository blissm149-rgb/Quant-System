"""Unit tests for corporate_action_adjuster module.

TESTING_PLAN.md Section 3.2 — Data Layer.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.data_layer.corporate_action_adjuster import (
    CorporateAction,
    CorporateActionAdjuster,
)
from tests.conftest import make_ohlcv


@pytest.mark.unit
@pytest.mark.tier1
class TestCorporateActionAdjuster:
    """CorporateActionAdjuster — adjusts prices for splits, dividends, spinoffs."""

    @pytest.fixture
    def adjuster(self):
        return CorporateActionAdjuster()

    @pytest.fixture
    def ohlcv(self):
        return make_ohlcv(tickers=["AAPL"], periods=50, seed=42)

    def test_two_to_one_split_adjusts_pre_split_prices(self, adjuster, ohlcv):
        """2:1 split correctly adjusts pre-split prices by factor."""
        dates = ohlcv.index.get_level_values("date").unique()
        ex_date = dates[25]  # middle of the data

        # Factor=0.5 means pre-split prices are halved (2:1 split)
        action = CorporateAction(
            ticker="AAPL",
            ex_date=ex_date,
            action_type="split",
            adjustment_factor=0.5,
        )
        adjuster.register_actions([action])

        adjusted = adjuster.adjust_dataframe(ohlcv)

        # Pre-split prices should be multiplied by factor (0.5 = halved)
        pre_split_original = ohlcv.loc[ohlcv.index.get_level_values("date") < ex_date, "close"]
        pre_split_adjusted = adjusted.loc[adjusted.index.get_level_values("date") < ex_date, "close"]

        np.testing.assert_allclose(
            pre_split_adjusted.values,
            pre_split_original.values * 0.5,
            rtol=1e-6,
        )

    def test_post_split_prices_unchanged(self, adjuster, ohlcv):
        """Post-split prices remain unchanged after adjustment."""
        dates = ohlcv.index.get_level_values("date").unique()
        ex_date = dates[25]

        action = CorporateAction(
            ticker="AAPL",
            ex_date=ex_date,
            action_type="split",
            adjustment_factor=0.5,
        )
        adjuster.register_actions([action])

        adjusted = adjuster.adjust_dataframe(ohlcv)

        post_split_original = ohlcv.loc[ohlcv.index.get_level_values("date") >= ex_date, "close"]
        post_split_adjusted = adjusted.loc[adjusted.index.get_level_values("date") >= ex_date, "close"]

        np.testing.assert_allclose(
            post_split_adjusted.values,
            post_split_original.values,
            rtol=1e-6,
        )

    def test_raw_data_not_modified(self, adjuster, ohlcv):
        """Original DataFrame is never modified (copy-on-write)."""
        dates = ohlcv.index.get_level_values("date").unique()
        original_close = ohlcv["close"].copy()

        action = CorporateAction(
            ticker="AAPL",
            ex_date=dates[25],
            action_type="split",
            adjustment_factor=0.5,
        )
        adjuster.register_actions([action])
        adjuster.adjust_dataframe(ohlcv)

        pd.testing.assert_series_equal(ohlcv["close"], original_close)

    def test_dividend_adjustment_backward_only(self, adjuster, ohlcv):
        """Dividend adjustment applied to pre-ex-date data only."""
        dates = ohlcv.index.get_level_values("date").unique()
        ex_date = dates[25]

        action = CorporateAction(
            ticker="AAPL",
            ex_date=ex_date,
            action_type="dividend",
            adjustment_factor=0.98,  # 2% dividend
        )
        adjuster.register_actions([action])

        adjusted = adjuster.adjust_dataframe(ohlcv)

        # Post-ex-date should be unchanged
        post_original = ohlcv.loc[ohlcv.index.get_level_values("date") >= ex_date, "close"]
        post_adjusted = adjusted.loc[adjusted.index.get_level_values("date") >= ex_date, "close"]
        np.testing.assert_allclose(post_adjusted.values, post_original.values, rtol=1e-6)

    def test_multiple_actions_chain_correctly(self, adjuster, ohlcv):
        """Multiple adjustments applied cumulatively in date order."""
        dates = ohlcv.index.get_level_values("date").unique()

        actions = [
            CorporateAction("AAPL", dates[15], "split", 0.5),
            CorporateAction("AAPL", dates[30], "split", 0.5),
        ]
        adjuster.register_actions(actions)

        adjusted = adjuster.adjust_dataframe(ohlcv)

        # Earliest data adjusted by both splits (cumulative factor = 0.5 * 0.5 = 0.25)
        earliest_original = ohlcv.loc[(dates[0], "AAPL"), "close"]
        earliest_adjusted = adjusted.loc[(dates[0], "AAPL"), "close"]
        np.testing.assert_allclose(earliest_adjusted, earliest_original * 0.25, rtol=1e-6)

    def test_register_and_clear_actions(self, adjuster):
        """Actions can be registered and cleared."""
        action = CorporateAction("AAPL", pd.Timestamp("2023-01-15"), "split", 2.0)
        adjuster.register_actions([action])
        assert len(adjuster.get_actions_for_ticker("AAPL")) == 1

        adjuster.clear_actions()
        assert len(adjuster.get_actions_for_ticker("AAPL")) == 0

    def test_no_actions_returns_unchanged(self, adjuster, ohlcv):
        """With no registered actions, data values are unchanged."""
        adjusted = adjuster.adjust_dataframe(ohlcv)
        # Check price columns are identical (volume dtype may differ due to division)
        for col in ["open", "high", "low", "close"]:
            np.testing.assert_allclose(adjusted[col].values, ohlcv[col].values, rtol=1e-10)
