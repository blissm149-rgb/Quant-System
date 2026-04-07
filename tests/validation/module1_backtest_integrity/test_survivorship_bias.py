"""Test survivorship bias detection in backtests.

Validates that delisted stocks are included in the historical universe
and that the universe is not constructed with forward-looking information.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine
from tests.conftest import make_ohlcv, STANDARD_TICKERS


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestSurvivorshipBias:
    """Verify that the backtest universe handles delisted stocks correctly."""

    def _make_ohlcv_with_delisting(
        self, tickers, periods=300, delist_ticker="META", delist_day=150, seed=42
    ):
        """Create OHLCV where one stock 'delists' partway through."""
        ohlcv = make_ohlcv(tickers=tickers, periods=periods, seed=seed)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        delist_date = dates[delist_day]

        mask = ~(
            (ohlcv.index.get_level_values("ticker") == delist_ticker)
            & (ohlcv.index.get_level_values(0) > delist_date)
        )
        return ohlcv[mask], delist_date, dates

    def test_delisted_stocks_included_in_historical_universe(self):
        """A stock that 'delists' at day 150 should appear in the universe
        for days 1-150 and be absent afterward."""
        tickers = STANDARD_TICKERS[:5]
        ohlcv, delist_date, dates = self._make_ohlcv_with_delisting(
            tickers, periods=300, delist_ticker="META", delist_day=150
        )

        engine = DataAlignmentEngine()

        # Before delisting: pre-filter to dates < pre_delist_date
        pre_delist_date = dates[100]
        safe_pre = ohlcv[ohlcv.index.get_level_values(0) < pre_delist_date]
        aligned = engine.get_aligned_data(safe_pre, pre_delist_date, lookback_days=50)
        tickers_in_data = aligned.index.get_level_values("ticker").unique()
        assert "META" in tickers_in_data, (
            "META should appear in universe before delist date"
        )

        # After delisting: pre-filter and use lookback that doesn't overlap
        post_delist_date = dates[250]
        safe_post = ohlcv[ohlcv.index.get_level_values(0) < post_delist_date]
        aligned_post = engine.get_aligned_data(
            safe_post, post_delist_date, lookback_days=50
        )
        tickers_post = aligned_post.index.get_level_values("ticker").unique()

        assert "META" not in tickers_post, (
            "META should be absent from universe 100 days after delisting "
            "when lookback is only 50 days"
        )

    def test_universe_not_forward_looking(self):
        """The universe at day T=100 must not contain any stock that
        first appears at T > 100."""
        tickers = STANDARD_TICKERS[:5]
        ohlcv = make_ohlcv(tickers=tickers, periods=300, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()

        # Add a 'late entry' stock that only has data from day 200 onward
        late_dates = dates[200:]
        late_data = []
        rng = np.random.default_rng(99)
        base_price = 50.0
        for d in late_dates:
            ret = rng.normal(0, 0.02)
            base_price *= 1 + ret
            late_data.append(
                {
                    "open": base_price * 0.99,
                    "high": base_price * 1.01,
                    "low": base_price * 0.98,
                    "close": base_price,
                    "volume": 1_000_000,
                    "adj_close": base_price,
                }
            )

        late_df = pd.DataFrame(late_data, index=late_dates)
        late_df["ticker"] = "LATE_ENTRY"
        late_df = late_df.set_index("ticker", append=True)
        late_df.index.names = ohlcv.index.names

        combined = pd.concat([ohlcv, late_df]).sort_index()

        engine = DataAlignmentEngine()

        # At T=100, LATE_ENTRY should NOT be in the universe
        as_of_early = dates[100]
        safe_early = combined[combined.index.get_level_values(0) < as_of_early]
        aligned = engine.get_aligned_data(safe_early, as_of_early, lookback_days=50)
        tickers_at_100 = aligned.index.get_level_values("ticker").unique()
        assert "LATE_ENTRY" not in tickers_at_100, (
            "LATE_ENTRY should not appear in universe at day 100"
        )

        # At T=250, LATE_ENTRY SHOULD be in the universe
        as_of_late = dates[250]
        safe_late = combined[combined.index.get_level_values(0) < as_of_late]
        aligned_late = engine.get_aligned_data(
            safe_late, as_of_late, lookback_days=50
        )
        tickers_at_250 = aligned_late.index.get_level_values("ticker").unique()
        assert "LATE_ENTRY" in tickers_at_250, (
            "LATE_ENTRY should appear in universe at day 250"
        )

    def test_backtest_returns_differ_with_survivorship(self):
        """Running backtests with vs. without delisted stocks must produce
        different returns, proving survivorship bias matters."""
        from tests.validation.shared.backtest_harness import BacktestHarness
        from quant_fund.research_algorithms.factor_models.momentum_factor import (
            MomentumFactor,
        )

        tickers = STANDARD_TICKERS[:5]
        full_ohlcv = make_ohlcv(tickers=tickers, periods=500, seed=42)

        ohlcv_with_delist, _, _ = self._make_ohlcv_with_delisting(
            tickers, periods=500, delist_ticker="META", delist_day=200
        )

        # Survivorship-biased: remove META entirely
        survivor_only = full_ohlcv[
            full_ohlcv.index.get_level_values("ticker") != "META"
        ]

        harness = BacktestHarness(seed=42)
        strategy = MomentumFactor()

        result_with_delist = harness.run(ohlcv_with_delist, strategy)
        result_survivor = harness.run(survivor_only, strategy)

        if len(result_with_delist.daily_returns) > 0 and len(result_survivor.daily_returns) > 0:
            ret_diff = abs(
                result_with_delist.total_return - result_survivor.total_return
            )
            assert ret_diff > 0.001, (
                f"Survivorship bias should matter: returns differ by only "
                f"{ret_diff:.6f} (expected > 0.001)"
            )
