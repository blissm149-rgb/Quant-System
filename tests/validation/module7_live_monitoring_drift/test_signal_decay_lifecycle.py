"""Test alpha signal decay detection and strategy lifecycle management.

Validates SignalDecayDetector IC half-life estimation,
InformationCoefficientMonitor degradation alerts,
AlphaPerformanceTracker metric accuracy, and
StrategyRetirementManager state transitions.
"""

import pytest
import numpy as np
import pandas as pd

from quant_fund.alpha_monitoring.signal_decay_detector import (
    DecayAction,
    DecayResult,
    SignalDecayDetector,
)
from quant_fund.alpha_monitoring.information_coefficient_monitor import (
    AlertLevel,
    InformationCoefficientMonitor,
    SignalDegradationAlert,
)
from quant_fund.alpha_monitoring.alpha_performance_tracker import (
    AlphaPerformanceTracker,
    SignalPerformance,
)
from quant_fund.alpha_monitoring.strategy_retirement_manager import (
    StrategyRetirementManager,
    StrategyStatus,
)


pytestmark = [pytest.mark.validation, pytest.mark.tier4]


class TestSignalDecayDetection:
    """Validate IC decay detection under controlled scenarios."""

    def test_fast_decay_triggers_retire(self):
        """Rapidly decaying IC should trigger RETIRE action."""
        detector = SignalDecayDetector(config={
            "decay_review_threshold": 20,
            "decay_retire_threshold": 10,
            "decay_min_history": 30,
        })
        dates = pd.bdate_range("2023-01-02", periods=100)

        for i, dt in enumerate(dates):
            # Very fast decay: half-life ~ 5 days
            ic = 0.1 * np.exp(-i / 5.0)
            detector.update("fast_decay", ic, dt)

        result = detector.detect("fast_decay")
        assert result is not None
        assert result.action in (DecayAction.REVIEW, DecayAction.RETIRE)

    def test_stable_signal_no_action(self):
        """Highly autocorrelated (slow mean-reverting) IC should not trigger RETIRE."""
        detector = SignalDecayDetector(config={
            "decay_review_threshold": 20,
            "decay_retire_threshold": 10,
            "decay_min_history": 30,
            "decay_estimation_window": 100,
        })
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-02", periods=100)

        # AR(1) process with phi=0.98 -> half-life ~34 days, well above thresholds
        ic = 0.08
        for dt in dates:
            ic = 0.98 * ic + 0.02 * 0.08 + rng.normal(0, 0.002)
            detector.update("stable", ic, dt)

        result = detector.detect("stable")
        assert result is not None
        assert result.action == DecayAction.NONE

    def test_insufficient_history_returns_none(self):
        """Signal with less than min_history should return None."""
        detector = SignalDecayDetector(config={"decay_min_history": 60})
        dates = pd.bdate_range("2023-01-02", periods=20)

        for dt in dates:
            detector.update("short", 0.05, dt)

        result = detector.detect("short")
        assert result is None

    def test_detect_all_covers_multiple_signals(self):
        """detect_all should return results for all signals with sufficient history."""
        detector = SignalDecayDetector(config={"decay_min_history": 30})
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-02", periods=60)

        for dt in dates:
            detector.update("sig_a", 0.05 + rng.normal(0, 0.01), dt)
            detector.update("sig_b", 0.03 + rng.normal(0, 0.01), dt)

        results = detector.detect_all()
        assert len(results) == 2
        names = {r.signal_name for r in results}
        assert names == {"sig_a", "sig_b"}


class TestICMonitorAlerts:
    """Validate IC monitor generates correct alerts under degradation."""

    def test_negative_ic_triggers_warning(self):
        """Anti-correlated signal should trigger at least one WARNING."""
        monitor = InformationCoefficientMonitor(config={
            "ic_short_window": 20,
            "ic_long_window": 60,
        })
        rng = np.random.default_rng(42)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        dates = pd.bdate_range("2023-01-02", periods=60)
        all_alerts = []

        for dt in dates:
            signal = pd.Series(rng.standard_normal(5), index=tickers)
            # Anti-correlated returns
            returns = pd.Series(-signal * 0.5 + rng.standard_normal(5) * 0.1, index=tickers)
            alerts = monitor.update("bad_signal", signal, returns, dt)
            all_alerts.extend(alerts)

        warnings = [a for a in all_alerts if a.alert_level == AlertLevel.WARNING]
        assert len(warnings) > 0

    def test_strong_signal_no_warnings(self):
        """Strongly positive IC signal should produce no WARNING/CRITICAL alerts."""
        monitor = InformationCoefficientMonitor(config={
            "ic_short_window": 10,
            "ic_long_window": 30,
            "ic_floor": -0.05,  # lenient floor
            "ic_tstat_floor": 0.5,  # lenient t-stat
        })
        rng = np.random.default_rng(42)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        dates = pd.bdate_range("2023-01-02", periods=60)
        all_alerts = []

        for dt in dates:
            signal = pd.Series(rng.standard_normal(5), index=tickers)
            returns = pd.Series(signal * 0.8 + rng.standard_normal(5) * 0.1, index=tickers)
            alerts = monitor.update("good_signal", signal, returns, dt)
            all_alerts.extend(alerts)

        bad_alerts = [
            a for a in all_alerts
            if a.alert_level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
        ]
        assert len(bad_alerts) == 0

    def test_rolling_ic_series_populated(self):
        """get_rolling_ic should return a populated Series after updates."""
        monitor = InformationCoefficientMonitor()
        rng = np.random.default_rng(42)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

        for i in range(30):
            dt = pd.Timestamp("2023-01-02") + pd.Timedelta(days=i)
            signal = pd.Series(rng.standard_normal(5), index=tickers)
            returns = pd.Series(rng.standard_normal(5), index=tickers)
            monitor.update("test_sig", signal, returns, dt)

        rolling_ic = monitor.get_rolling_ic("test_sig")
        assert isinstance(rolling_ic, pd.Series)
        assert len(rolling_ic) == 30


class TestAlphaPerformanceTracking:
    """Validate per-signal performance computation."""

    def test_performance_metrics_populated(self):
        """Performance should include realised IC, IR, and PnL."""
        tracker = AlphaPerformanceTracker()
        rng = np.random.default_rng(42)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        dates = pd.bdate_range("2023-01-02", periods=60)

        for dt in dates:
            signal = pd.Series(rng.standard_normal(5), index=tickers)
            returns = pd.Series(signal * 0.3 + rng.standard_normal(5) * 0.1, index=tickers)
            tracker.update("alpha_1", signal, returns, dt)

        perf = tracker.get_performance("alpha_1")
        assert isinstance(perf, SignalPerformance)
        assert perf.n_days == 60
        assert perf.realised_ic > 0  # signal is positively correlated

    def test_multiple_signals_tracked_independently(self):
        """Each signal should have independent performance metrics."""
        tracker = AlphaPerformanceTracker()
        rng = np.random.default_rng(42)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

        for i in range(30):
            dt = pd.Timestamp("2023-01-02") + pd.Timedelta(days=i)
            sig_a = pd.Series(rng.standard_normal(5), index=tickers)
            sig_b = pd.Series(rng.standard_normal(5), index=tickers)
            rets = pd.Series(rng.standard_normal(5) * 0.02, index=tickers)
            tracker.update("sig_a", sig_a, rets, dt)
            tracker.update("sig_b", sig_b, rets, dt)

        all_perfs = tracker.get_all_performances()
        assert len(all_perfs) == 2
        names = {p.signal_name for p in all_perfs}
        assert names == {"sig_a", "sig_b"}


class TestStrategyLifecycle:
    """Validate strategy retirement state machine transitions."""

    def test_active_to_review_on_low_ic(self):
        """IC below review threshold should transition ACTIVE -> UNDER_REVIEW."""
        mgr = StrategyRetirementManager(config={"review_ic_threshold": 0.01})
        mgr.register_strategy("s1", pd.Timestamp("2023-01-01"))

        state = mgr.update("s1", current_ic=0.005, date=pd.Timestamp("2023-06-01"))
        assert state.status == StrategyStatus.UNDER_REVIEW
        assert state.allocation_multiplier < 1.0

    def test_review_to_active_on_recovery(self):
        """IC recovery above threshold should transition UNDER_REVIEW -> ACTIVE."""
        mgr = StrategyRetirementManager(config={
            "review_ic_threshold": 0.01,
            "recovery_ic_threshold": 0.03,
        })
        mgr.register_strategy("s1", pd.Timestamp("2023-01-01"))
        mgr.update("s1", current_ic=0.005, date=pd.Timestamp("2023-06-01"))

        state = mgr.update("s1", current_ic=0.05, date=pd.Timestamp("2023-06-15"))
        assert state.status == StrategyStatus.ACTIVE
        assert state.allocation_multiplier == 1.0

    def test_manual_retirement(self):
        """Manual retirement should transition to RETIRED with zero allocation."""
        mgr = StrategyRetirementManager()
        mgr.register_strategy("s1", pd.Timestamp("2023-01-01"))

        state = mgr.retire_strategy(
            "s1", pd.Timestamp("2023-12-01"), reason="persistent decay"
        )
        assert state.status == StrategyStatus.RETIRED
        assert state.allocation_multiplier == 0.0

    def test_retired_strategy_excluded_from_active(self):
        """Retired strategies should not appear in get_active_strategies."""
        mgr = StrategyRetirementManager()
        mgr.register_strategy("active_1", pd.Timestamp("2023-01-01"))
        mgr.register_strategy("to_retire", pd.Timestamp("2023-01-01"))
        mgr.retire_strategy("to_retire", pd.Timestamp("2023-12-01"))

        active = mgr.get_active_strategies()
        names = [s.strategy_name for s in active]
        assert "active_1" in names
        assert "to_retire" not in names

    def test_allocation_multipliers_reflect_state(self):
        """Allocation multipliers should match current lifecycle state."""
        mgr = StrategyRetirementManager(config={
            "review_ic_threshold": 0.01,
            "review_allocation_multiplier": 0.5,
        })
        mgr.register_strategy("s1", pd.Timestamp("2023-01-01"))
        mgr.register_strategy("s2", pd.Timestamp("2023-01-01"))

        mgr.update("s1", current_ic=0.005, date=pd.Timestamp("2023-06-01"))

        mults = mgr.get_allocation_multipliers()
        assert mults["s1"] == pytest.approx(0.5)
        assert mults["s2"] == 1.0

    def test_state_history_records_transitions(self):
        """Strategy state should record a history of transitions."""
        mgr = StrategyRetirementManager(config={"review_ic_threshold": 0.01})
        mgr.register_strategy("s1", pd.Timestamp("2023-01-01"))
        mgr.update("s1", current_ic=0.005, date=pd.Timestamp("2023-06-01"))

        state = mgr.get_state("s1")
        assert len(state.history) >= 2  # registration + transition
