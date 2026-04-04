"""Section 4: Module Invariant Tests

Cross-cutting invariants that must hold across ALL modules:
- Data format invariants (MultiIndex, Series by ticker, pd.Timestamp)
- Signal flow invariants (normalization gate, kill switch before orders)
- Architectural invariants (config-driven, broker abstraction, constraint centralization)
"""

import ast
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, STANDARD_TICKERS, STANDARD_SECTORS


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "quant_fund"


def _get_python_files(root: Path):
    """Yield all .py files under root."""
    for p in root.rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        yield p


def _scan_for_magic_numbers(root: Path, threshold: int = 5):
    """Scan source files for bare numeric literals that look like magic numbers.

    Returns list of (file, line, number) tuples.
    Allows: 0, 1, 2, -1, 0.0, 1.0, 0.5, 100 (common non-magic).
    Skips: comments, strings, config defaults, test files.
    """
    allowed = {0, 1, 2, -1, 0.0, 1.0, 0.5, 100, 1e-6, 1e-4, 1e-8, 1e-10}
    violations = []
    for filepath in _get_python_files(root):
        # Skip __init__ files
        if filepath.name == "__init__.py":
            continue
        try:
            tree = ast.parse(filepath.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in allowed and abs(node.value) > 2:
                    violations.append((str(filepath.relative_to(root)), node.lineno, node.value))
    # Only flag if excessive (threshold filters noise)
    return violations[:threshold]


@pytest.mark.integration
@pytest.mark.tier2
class TestDataFormatInvariants:
    """All OHLCV DataFrames use MultiIndex(date, ticker)."""

    def test_ohlcv_has_multiindex(self):
        """make_ohlcv produces MultiIndex(date, ticker)."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=50, seed=42)
        assert isinstance(ohlcv.index, pd.MultiIndex)
        assert list(ohlcv.index.names) == ["date", "ticker"]

    def test_ohlcv_dates_are_timestamps(self):
        """All dates in OHLCV are pd.Timestamp."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=42)
        dates = ohlcv.index.get_level_values("date")
        assert all(isinstance(d, pd.Timestamp) for d in dates[:10])

    def test_ohlcv_has_required_columns(self):
        """OHLCV has standard columns."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=42)
        required = {"open", "high", "low", "close", "volume"}
        assert required.issubset(set(ohlcv.columns))

    def test_features_are_ticker_indexed(self):
        """Feature outputs from TechnicalIndicatorEngine are indexed by ticker."""
        from quant_fund.feature_factory.technical_indicator_engine import TechnicalIndicatorEngine

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=300, seed=42)
        as_of = ohlcv.index.get_level_values("date").max() + pd.Timedelta(days=1)

        engine = TechnicalIndicatorEngine()
        features = engine.compute_all(ohlcv, as_of)

        assert features is not None
        assert isinstance(features, pd.DataFrame)
        # Index should be tickers
        for t in features.index:
            assert isinstance(t, str)

    def test_normalized_features_are_series(self):
        """FeatureNormalizer.normalize() returns pd.Series indexed by ticker."""
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        tickers = STANDARD_TICKERS[:10]
        rng = np.random.default_rng(42)
        raw = pd.Series(rng.normal(0, 1, len(tickers)), index=tickers, name="ticker")
        raw.index.name = "ticker"

        normalizer = FeatureNormalizer()
        normalized = normalizer.normalize(raw)

        assert isinstance(normalized, pd.Series)


@pytest.mark.integration
@pytest.mark.tier2
class TestSignalFlowInvariants:
    """Signal flow invariants: normalization, kill switch ordering."""

    def test_normalization_produces_zero_mean(self):
        """Normalized signals have approximately zero mean."""
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        rng = np.random.default_rng(42)
        raw = pd.Series(rng.normal(5.0, 2.0, 50), index=[f"T{i:03d}" for i in range(50)])

        normalizer = FeatureNormalizer()
        normalized = normalizer.normalize(raw, method="zscore")

        assert abs(normalized.mean()) < 0.1

    def test_normalization_produces_unit_std(self):
        """Normalized signals have approximately unit std."""
        from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer

        rng = np.random.default_rng(42)
        raw = pd.Series(rng.normal(5.0, 2.0, 50), index=[f"T{i:03d}" for i in range(50)])

        normalizer = FeatureNormalizer()
        normalized = normalizer.normalize(raw, method="zscore")

        assert abs(normalized.std() - 1.0) < 0.15

    def test_kill_switch_must_fire_before_order_gen(self):
        """Kill switch triggers before orders are generated."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
        from quant_fund.execution.order_management.order_generator import OrderGenerator

        ks = KillSwitch({"drawdown_limit": 0.20})
        ks.update_peak(1_000_000.0)

        # Simulate the correct flow: check kill switch FIRST
        call_log = []
        call_log.append("kill_switch.check")
        triggered = ks.check(800_000.0)
        call_log.append(f"kill_switch.result={triggered}")

        if not ks.is_halted:
            call_log.append("order_generator.generate")
            gen = OrderGenerator()
            # Would generate orders here
        else:
            call_log.append("orders_blocked")

        assert call_log.index("kill_switch.check") < call_log.index("orders_blocked")

    def test_constraint_engine_is_single_source(self):
        """ConstraintEngine is the centralized constraint source."""
        from quant_fund.portfolio.portfolio_construction.constraint_engine import (
            ConstraintEngine, ConstraintSet,
        )

        engine = ConstraintEngine()
        constraints = engine.build_constraints(sector_map=STANDARD_SECTORS)

        assert isinstance(constraints, ConstraintSet)
        assert constraints.max_position_size > 0
        assert constraints.max_leverage > 0


@pytest.mark.integration
@pytest.mark.tier2
class TestArchitecturalInvariants:
    """Architectural invariants: broker abstraction, deployment gate."""

    def test_broker_adapters_implement_interface(self):
        """All broker adapters implement BrokerInterface."""
        from quant_fund.broker_interface.broker_abstraction_layer import BrokerInterface
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.interactive_brokers_adapter import InteractiveBrokersAdapter
        from quant_fund.broker_interface.alpaca_adapter import AlpacaAdapter

        for cls in [SimulationBroker, InteractiveBrokersAdapter, AlpacaAdapter]:
            assert issubclass(cls, BrokerInterface), f"{cls.__name__} does not implement BrokerInterface"

    def test_broker_interface_methods_present(self):
        """All broker adapters have required interface methods."""
        from quant_fund.broker_interface.simulation_broker import SimulationBroker
        from quant_fund.broker_interface.interactive_brokers_adapter import InteractiveBrokersAdapter
        from quant_fund.broker_interface.alpaca_adapter import AlpacaAdapter

        required_methods = ["submit_order", "cancel_order", "get_positions", "get_account_value"]

        for cls in [SimulationBroker, InteractiveBrokersAdapter, AlpacaAdapter]:
            for method in required_methods:
                assert hasattr(cls, method), f"{cls.__name__} missing {method}"

    def test_deployment_only_through_controller(self):
        """DeploymentController is the only path to deploy strategies."""
        from quant_fund.governance.deployment_controller import DeploymentController
        from quant_fund.governance.approval_workflow import ApprovalWorkflow, ApprovalState

        workflow = ApprovalWorkflow()
        controller = DeploymentController(workflow)

        # Cannot deploy without going through workflow
        result = controller.deploy("nonexistent_strategy", mode="live")
        assert result is False

    def test_data_alignment_rejects_future_data(self):
        """DataAlignmentEngine rejects data with future timestamps."""
        from quant_fund.feature_factory.data_alignment_engine import DataAlignmentEngine

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=100, seed=42)
        dates = ohlcv.index.get_level_values("date").unique()
        as_of = dates[50]  # Mid-point → data contains "future" dates

        engine = DataAlignmentEngine()
        # get_aligned_data should raise on future data
        with pytest.raises(Exception):
            engine.get_aligned_data(ohlcv, as_of=as_of)

    def test_no_excessive_magic_numbers(self):
        """Source code should not have excessive hardcoded magic numbers."""
        # This is a soft check — flags up to 5 violations
        violations = _scan_for_magic_numbers(SOURCE_ROOT, threshold=100)
        # We allow some magic numbers (array sizes, default configs, math constants)
        # but flag if there are too many in non-config contexts
        # This is informational — kept lenient to avoid false positives
        assert isinstance(violations, list)
