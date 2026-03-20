"""Unit tests for capacity model modules.

TESTING_PLAN.md Section 3.8 — market_impact_model, liquidity_estimator, capacity_simulator.
"""

import numpy as np
import pandas as pd
import pytest

from quant_fund.portfolio.capacity_model.market_impact_model import MarketImpactModel
from quant_fund.portfolio.capacity_model.liquidity_estimator import LiquidityEstimator
from quant_fund.portfolio.capacity_model.capacity_simulator import CapacitySimulator
from tests.conftest import make_ohlcv


@pytest.mark.unit
@pytest.mark.tier2
class TestMarketImpactModel:
    """MarketImpactModel — Almgren-Chriss impact estimation."""

    @pytest.fixture
    def model(self):
        return MarketImpactModel()

    def test_impact_positive(self, model):
        """Market impact is always positive for valid inputs."""
        impact = model.estimate_impact_bps(
            order_size_usd=100_000, adv_usd=50_000_000, daily_volatility=0.02,
        )
        assert impact > 0

    def test_larger_order_higher_impact(self, model):
        """Larger orders have higher impact."""
        small = model.estimate_impact_bps(10_000, 50_000_000, 0.02)
        large = model.estimate_impact_bps(1_000_000, 50_000_000, 0.02)
        assert large > small

    def test_zero_order_zero_impact(self, model):
        """Zero order size returns zero impact."""
        impact = model.estimate_impact_bps(0, 50_000_000, 0.02)
        assert impact == 0.0

    def test_portfolio_impact(self, model):
        """estimate_impact_portfolio returns per-ticker impacts."""
        trades = pd.Series({"AAPL": 100_000, "MSFT": 50_000})
        adv = pd.Series({"AAPL": 50_000_000, "MSFT": 30_000_000})
        vol = pd.Series({"AAPL": 0.02, "MSFT": 0.018})
        impacts = model.estimate_impact_portfolio(trades, adv, vol)
        assert len(impacts) == 2
        assert (impacts >= 0).all()


@pytest.mark.unit
@pytest.mark.tier2
class TestLiquidityEstimator:
    """LiquidityEstimator — ADV and liquidity metrics."""

    @pytest.fixture
    def estimator(self):
        return LiquidityEstimator()

    def test_estimate_returns_dataframe(self, estimator):
        """estimate returns DataFrame with liquidity metrics."""
        data = make_ohlcv(tickers=["AAPL", "MSFT"], periods=50, seed=42)
        volume = data[["volume"]].unstack("ticker").droplevel(0, axis=1)
        prices = data[["close"]].unstack("ticker").droplevel(0, axis=1)
        result = estimator.estimate(volume, prices, as_of=pd.Timestamp("2025-01-01"))
        assert isinstance(result, pd.DataFrame)
        assert "adv_usd" in result.columns

    def test_adv_usd_positive(self, estimator):
        """ADV in USD is positive for valid data."""
        data = make_ohlcv(tickers=["AAPL"], periods=50, seed=42)
        volume = data[["volume"]].unstack("ticker").droplevel(0, axis=1)
        prices = data[["close"]].unstack("ticker").droplevel(0, axis=1)
        result = estimator.estimate(volume, prices, as_of=pd.Timestamp("2025-01-01"))
        assert (result["adv_usd"] > 0).all()


@pytest.mark.unit
@pytest.mark.tier2
class TestCapacitySimulator:
    """CapacitySimulator — max AUM estimation."""

    @pytest.fixture
    def simulator(self):
        return CapacitySimulator()

    def test_estimate_capacity_returns_dict(self, simulator):
        """estimate_capacity returns dict with max_capacity_usd."""
        target = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        current = pd.Series({"AAPL": 0.0, "MSFT": 0.0})
        adv = pd.Series({"AAPL": 50_000_000, "MSFT": 30_000_000})
        vol = pd.Series({"AAPL": 0.02, "MSFT": 0.018})
        result = simulator.estimate_capacity(
            target, current, adv, vol, expected_alpha_bps=50,
        )
        assert isinstance(result, dict)
        assert "max_capacity_usd" in result
