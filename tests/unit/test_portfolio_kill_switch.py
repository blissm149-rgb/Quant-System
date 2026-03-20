"""Unit tests for portfolio_kill_switch module.

TESTING_PLAN.md Section 3.9 — CRITICAL: simplest module, no dependencies.
"""

import pandas as pd
import pytest

from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch, StrategyKillSwitch


@pytest.mark.unit
@pytest.mark.tier1
class TestKillSwitch:
    """KillSwitch — portfolio-level hard stop on drawdown breach."""

    def test_fires_at_20_percent(self):
        """Kill switch fires at exactly 20% drawdown."""
        ks = KillSwitch(config={"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        ks.update_peak(1_000_000)
        assert ks.check(800_000) is True  # exactly 20% DD

    def test_does_not_fire_below_threshold(self):
        """Kill switch does not fire just under 20%."""
        ks = KillSwitch(config={"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        ks.update_peak(1_000_000)
        assert ks.check(800_001) is False

    def test_peak_tracking(self):
        """update_peak tracks high-water mark (never decreases)."""
        ks = KillSwitch(config={"initial_nav": 900_000})
        ks.update_peak(900_000)
        ks.update_peak(1_000_000)
        ks.update_peak(950_000)  # should not lower peak
        # Peak should be 1,000,000
        assert not ks.check(850_000)  # 15% from 1M — under 20%

    def test_is_halted_after_trigger(self):
        """is_halted is True after kill switch fires."""
        ks = KillSwitch(config={"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        ks.update_peak(1_000_000)
        ks.check(700_000)  # 30% DD
        assert ks.is_halted is True

    def test_reset_clears_halt(self):
        """reset clears the halted state."""
        ks = KillSwitch(config={"drawdown_limit": 0.20, "initial_nav": 1_000_000})
        ks.update_peak(1_000_000)
        ks.check(700_000)
        assert ks.is_halted is True
        ks.reset(700_000)
        assert ks.is_halted is False

    def test_extreme_crash_90_percent(self):
        """Kill switch fires cleanly on 90% crash."""
        ks = KillSwitch(config={"drawdown_limit": 0.20, "initial_nav": 10_000_000})
        ks.update_peak(10_000_000)
        assert ks.check(1_000_000) is True  # 90% DD


@pytest.mark.unit
@pytest.mark.tier1
class TestStrategyKillSwitch:
    """StrategyKillSwitch — per-strategy kill switch."""

    @pytest.fixture
    def sks(self):
        return StrategyKillSwitch(config={"drawdown_limit": 0.20})

    def test_register_and_check(self, sks):
        """Register strategy and check drawdown."""
        sks.register_strategy("strat_1", initial_nav=1_000_000)
        assert sks.check("strat_1", 900_000) is False  # 10%
        assert sks.check("strat_1", 800_000) is True   # 20%

    def test_is_halted(self, sks):
        """is_halted returns True for halted strategy."""
        sks.register_strategy("strat_1", initial_nav=1_000_000)
        sks.check("strat_1", 700_000)
        assert sks.is_halted("strat_1") is True

    def test_generate_liquidation_orders(self, sks):
        """Generates sell/buy orders to flatten positions."""
        sks.register_strategy("strat_1", initial_nav=1_000_000)
        sks.check("strat_1", 700_000)
        positions = pd.Series({"AAPL": 100, "MSFT": -50})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})
        orders = sks.generate_liquidation_orders("strat_1", positions, prices)
        assert len(orders) == 2  # one per position

    def test_halted_strategies_list(self, sks):
        """halted_strategies returns list of halted strategy IDs."""
        sks.register_strategy("a", 1_000_000)
        sks.register_strategy("b", 1_000_000)
        sks.check("a", 700_000)  # halt a
        assert "a" in sks.halted_strategies
        assert "b" not in sks.halted_strategies
