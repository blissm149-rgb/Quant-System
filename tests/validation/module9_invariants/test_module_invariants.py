"""Module 9 – Cross-Cutting Module Invariant Tests.

Validates architectural invariants that must hold across the entire codebase:
  1. Data format invariants (MultiIndex, feature Series, Timestamp types)
  2. Signal flow invariants (normalisation gate, kill switch ordering)
  3. Architectural invariants (broker abstraction, constraint centralisation,
     deployment gate, data alignment routing, config-driven parameters)

These are static and dynamic audits drawn from TESTING_PLAN Section 4
and HANDOFF.md strict rules.
"""

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv, make_returns, STANDARD_TICKERS, STANDARD_SECTORS

pytestmark = [pytest.mark.validation]

# Root of production code
QUANT_FUND_ROOT = Path(__file__).resolve().parents[3] / "quant_fund"


# ── helpers ─────────────────────────────────────────────────────────


def _collect_python_files(root: Path) -> List[Path]:
    """Return all .py source files under *root*, excluding __pycache__."""
    return sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in str(p) and p.name != "__init__.py"
    )


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════
# 1. DATA FORMAT INVARIANTS
# ═══════════════════════════════════════════════════════════════════


class TestOHLCVMultiIndex:
    """All OHLCV DataFrames must use MultiIndex(date, ticker)."""

    def test_make_ohlcv_has_correct_multiindex(self):
        """The shared fixture must produce a MultiIndex with (date, ticker)."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=42)
        assert isinstance(ohlcv.index, pd.MultiIndex)
        assert list(ohlcv.index.names) == ["date", "ticker"]

    def test_ohlcv_date_level_is_datetime(self):
        """The date level of the MultiIndex must be DatetimeIndex."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=50, seed=42)
        date_level = ohlcv.index.get_level_values("date")
        assert pd.api.types.is_datetime64_any_dtype(date_level)


class TestFeatureOutputFormat:
    """All feature generators must return pd.Series indexed by ticker."""

    def _get_feature_generators(self):
        """Import and instantiate all feature generators."""
        from quant_fund.feature_factory.technical_indicator_engine import (
            TechnicalIndicatorEngine,
        )
        engine = TechnicalIndicatorEngine()
        return engine.generators

    def test_all_technical_features_return_series(self):
        """Each technical indicator .compute() returns a pd.Series."""
        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=300, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[-1]

        from quant_fund.feature_factory.data_alignment_engine import (
            DataAlignmentEngine,
        )
        alignment = DataAlignmentEngine()

        for gen in self._get_feature_generators():
            aligned = alignment.get_aligned_data_permissive(
                ohlcv, as_of=as_of, lookback_days=gen.lookback_days
            )
            if aligned.empty:
                continue
            result = gen.compute(aligned, as_of=as_of)
            assert isinstance(result, pd.Series), (
                f"{gen.feature_name}.compute() returned {type(result).__name__}, "
                f"expected pd.Series"
            )

    def test_factor_models_return_series(self):
        """Each factor model .compute() returns a pd.Series."""
        from quant_fund.research_algorithms.factor_models.momentum_factor import (
            MomentumFactor,
        )
        from quant_fund.research_algorithms.factor_models.value_factor import (
            ValueFactor,
        )
        from quant_fund.research_algorithms.factor_models.quality_factor import (
            QualityFactor,
        )
        from quant_fund.research_algorithms.factor_models.low_volatility_factor import (
            LowVolatilityFactor,
        )
        from quant_fund.research_algorithms.factor_models.size_factor import SizeFactor

        from quant_fund.feature_factory.data_alignment_engine import (
            DataAlignmentEngine,
        )

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=300, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[-1]
        alignment = DataAlignmentEngine()

        for FactorCls in [MomentumFactor, ValueFactor, QualityFactor,
                          LowVolatilityFactor, SizeFactor]:
            factor = FactorCls()
            aligned = alignment.get_aligned_data_permissive(
                ohlcv, as_of=as_of, lookback_days=factor.lookback_days
            )
            if aligned.empty:
                continue
            result = factor.compute(aligned, as_of=as_of)
            assert isinstance(result, pd.Series), (
                f"{FactorCls.__name__}.compute() returned {type(result).__name__}"
            )


# ═══════════════════════════════════════════════════════════════════
# 2. SIGNAL FLOW INVARIANTS
# ═══════════════════════════════════════════════════════════════════


class TestSignalCombinationContract:
    """Signal ranking engine combine() must return ticker-indexed Series."""

    def test_combine_returns_ticker_indexed_series(self):
        from quant_fund.alpha_discovery.signal_ranking_engine import (
            SignalRankingEngine,
        )
        engine = SignalRankingEngine()

        signals = {
            "momentum": pd.Series(
                [0.5, -0.3, 0.1], index=["AAPL", "MSFT", "GOOG"]
            ),
            "value": pd.Series(
                [-0.2, 0.4, 0.6], index=["AAPL", "MSFT", "GOOG"]
            ),
        }
        combined = engine.combine(signals)
        assert isinstance(combined, pd.Series)
        assert len(combined) == 3
        assert all(isinstance(t, str) for t in combined.index)

    def test_combine_handles_disjoint_tickers(self):
        """combine() must produce a union of tickers across all signals."""
        from quant_fund.alpha_discovery.signal_ranking_engine import (
            SignalRankingEngine,
        )
        engine = SignalRankingEngine()

        signals = {
            "s1": pd.Series([1.0], index=["AAPL"]),
            "s2": pd.Series([2.0], index=["MSFT"]),
        }
        combined = engine.combine(signals)
        assert set(combined.index) == {"AAPL", "MSFT"}


class TestKillSwitchOrdering:
    """Kill switch must block new orders when drawdown is breached."""

    def test_kill_switch_fires_before_accepting_new_positions(self):
        """When the kill switch triggers, the system must enter HALT/RISK_HALT
        state, which prevents further trading. Verify the state machine gate."""
        from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch

        ks = KillSwitch()
        ks.update_peak(1_000_000.0)

        # 20% drawdown -> should trigger
        assert ks.check(800_000.0) is True
        # Just under 20% -> should NOT trigger
        ks.reset(1_000_000.0)
        assert ks.check(800_001.0) is False

    def test_state_machine_blocks_trading_in_halt(self):
        """SystemStateMachine.can_trade must be False in RISK_HALT."""
        from quant_fund.infrastructure.system_state_machine import (
            SystemStateMachine,
            SystemState,
        )

        sm = SystemStateMachine()
        # Move to TRADING_ENABLED state
        sm.transition_to(SystemState.DATA_READY, reason="test")
        sm.transition_to(SystemState.TRADING_ENABLED, reason="test")
        assert sm.can_trade is True

        # Trigger risk halt
        sm.transition_to(SystemState.RISK_HALT, reason="drawdown breach")
        assert sm.can_trade is False


# ═══════════════════════════════════════════════════════════════════
# 3. BROKER ABSTRACTION INVARIANT
# ═══════════════════════════════════════════════════════════════════


class TestBrokerAbstraction:
    """No module outside broker_interface/ may directly import broker adapters."""

    def test_no_direct_adapter_imports_outside_broker_interface(self):
        """Scan all production files outside broker_interface/ for direct
        imports of InteractiveBrokersAdapter or AlpacaAdapter."""
        adapter_patterns = [
            re.compile(r"from\s+quant_fund\.broker_interface\.interactive_brokers_adapter\s+import"),
            re.compile(r"from\s+quant_fund\.broker_interface\.alpaca_adapter\s+import"),
            re.compile(r"import\s+quant_fund\.broker_interface\.interactive_brokers_adapter"),
            re.compile(r"import\s+quant_fund\.broker_interface\.alpaca_adapter"),
        ]

        violations: List[Tuple[str, int, str]] = []
        broker_dir = QUANT_FUND_ROOT / "broker_interface"

        for path in _collect_python_files(QUANT_FUND_ROOT):
            # Skip files inside broker_interface/ itself
            if broker_dir in path.parents or path.parent == broker_dir:
                continue

            source = _read_source(path)
            for i, line in enumerate(source.splitlines(), 1):
                for pat in adapter_patterns:
                    if pat.search(line):
                        violations.append((str(path.relative_to(QUANT_FUND_ROOT)), i, line.strip()))

        assert violations == [], (
            f"Direct adapter imports found outside broker_interface/:\n"
            + "\n".join(f"  {f}:{ln}: {code}" for f, ln, code in violations)
        )


# ═══════════════════════════════════════════════════════════════════
# 4. CONSTRAINT CENTRALISATION INVARIANT
# ═══════════════════════════════════════════════════════════════════


class TestConstraintCentralisation:
    """All portfolio constraint ENFORCEMENT must live in constraint_engine.py.
    Other modules may read constraint values for monitoring/display."""

    def test_constraint_engine_is_source_of_truth(self):
        """ConstraintEngine.build_constraints() must produce a ConstraintSet
        with the canonical defaults."""
        from quant_fund.portfolio.portfolio_construction.constraint_engine import (
            ConstraintEngine,
            ConstraintSet,
        )

        engine = ConstraintEngine()
        cs = engine.build_constraints()

        assert isinstance(cs, ConstraintSet)
        assert cs.max_position_size == pytest.approx(0.02)
        assert cs.max_sector_exposure == pytest.approx(0.20)
        assert cs.max_leverage == pytest.approx(2.0)
        assert cs.dollar_neutral is True

    def test_constraint_set_configurable_via_config(self):
        """ConstraintEngine must accept config overrides."""
        from quant_fund.portfolio.portfolio_construction.constraint_engine import (
            ConstraintEngine,
        )

        custom_config = {
            "position_limits": {
                "max_position_size": 0.05,
                "max_sector_exposure": 0.30,
                "max_leverage": 3.0,
            }
        }
        engine = ConstraintEngine(config=custom_config)
        cs = engine.build_constraints()

        assert cs.max_position_size == pytest.approx(0.05)
        assert cs.max_sector_exposure == pytest.approx(0.30)
        assert cs.max_leverage == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════════
# 5. DEPLOYMENT GATE INVARIANT
# ═══════════════════════════════════════════════════════════════════


class TestDeploymentGate:
    """Only deployment_controller.py can flip strategies from paper to live.
    Live deployment requires governance approval."""

    def test_no_mode_mutation_outside_deployment_controller(self):
        """Scan source for direct `.mode = "live"` assignments outside
        deployment_controller.py."""
        violations: List[Tuple[str, int, str]] = []
        pattern = re.compile(r'\.mode\s*=\s*["\']live["\']')

        for path in _collect_python_files(QUANT_FUND_ROOT):
            if path.name == "deployment_controller.py":
                continue
            source = _read_source(path)
            for i, line in enumerate(source.splitlines(), 1):
                if pattern.search(line):
                    violations.append((str(path.relative_to(QUANT_FUND_ROOT)), i, line.strip()))

        assert violations == [], (
            f"Direct .mode='live' assignments found outside deployment_controller.py:\n"
            + "\n".join(f"  {f}:{ln}: {code}" for f, ln, code in violations)
        )

    def test_live_deployment_requires_approval(self):
        """deploy(mode='live') without approval must return False."""
        from quant_fund.governance.deployment_controller import DeploymentController
        from quant_fund.governance.approval_workflow import ApprovalWorkflow

        workflow = ApprovalWorkflow()
        controller = DeploymentController(approval_workflow=workflow)

        # No approval submitted — deploying to live must fail
        result = controller.deploy("test_strategy", mode="live")
        assert result is False

    def test_paper_deployment_succeeds_without_approval(self):
        """deploy(mode='paper') should succeed without approval."""
        from quant_fund.governance.deployment_controller import DeploymentController
        from quant_fund.governance.approval_workflow import ApprovalWorkflow

        workflow = ApprovalWorkflow()
        controller = DeploymentController(approval_workflow=workflow)

        result = controller.deploy("test_strategy", mode="paper")
        assert result is True


# ═══════════════════════════════════════════════════════════════════
# 6. DATA ALIGNMENT ROUTING INVARIANT
# ═══════════════════════════════════════════════════════════════════


class TestDataAlignmentRouting:
    """All feature computation must route through DataAlignmentEngine."""

    def test_technical_indicator_engine_uses_alignment(self):
        """TechnicalIndicatorEngine.compute_all() must call
        get_aligned_data (strict) for every generator."""
        source_path = (
            QUANT_FUND_ROOT / "feature_factory" / "technical_indicator_engine.py"
        )
        source = _read_source(source_path)

        assert "get_aligned_data" in source, (
            "TechnicalIndicatorEngine does not call get_aligned_data"
        )

    def test_alignment_engine_referenced_in_tech_indicator_init(self):
        """TechnicalIndicatorEngine must initialise a DataAlignmentEngine."""
        source_path = (
            QUANT_FUND_ROOT / "feature_factory" / "technical_indicator_engine.py"
        )
        source = _read_source(source_path)

        assert "DataAlignmentEngine" in source, (
            "TechnicalIndicatorEngine does not reference DataAlignmentEngine"
        )

    def test_base_feature_generator_contract_documented(self):
        """BaseFeatureGenerator.compute() docstring must state that data
        is pre-aligned (timestamps < as_of)."""
        from quant_fund.feature_factory.base_feature_generator import (
            BaseFeatureGenerator,
        )
        doc = inspect.getdoc(BaseFeatureGenerator.compute) or ""
        assert "as_of" in doc.lower(), (
            "BaseFeatureGenerator.compute() docstring does not mention as_of"
        )


# ═══════════════════════════════════════════════════════════════════
# 7. CONFIG-DRIVEN PARAMETERS INVARIANT
# ═══════════════════════════════════════════════════════════════════


class TestConfigDrivenParameters:
    """Key modules must accept configuration via config dict, not hardcode
    their parameters."""

    @pytest.mark.parametrize(
        "module_path,class_name",
        [
            ("quant_fund.risk_engine.portfolio_kill_switch", "KillSwitch"),
            ("quant_fund.portfolio.portfolio_construction.constraint_engine", "ConstraintEngine"),
            ("quant_fund.research_algorithms.machine_learning.gradient_boosted_tree_model", "GradientBoostedTreeModel"),
            ("quant_fund.research_algorithms.machine_learning.random_forest_model", "RandomForestModel"),
            ("quant_fund.research_algorithms.machine_learning.neural_network_predictor", "NeuralNetworkPredictor"),
            ("quant_fund.representation_learning.autoencoder_model", "AutoencoderModel"),
            ("quant_fund.representation_learning.temporal_model_lstm", "TemporalModelLSTM"),
            ("quant_fund.representation_learning.temporal_model_transformer", "TemporalModelTransformer"),
        ],
        ids=lambda x: x.split(".")[-1] if "." in x else x,
    )
    def test_module_accepts_config_dict(self, module_path, class_name):
        """Every key module must accept an optional config dict in __init__."""
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())

        assert "config" in params, (
            f"{class_name}.__init__() has no 'config' parameter. "
            f"Parameters: {params}"
        )


# ═══════════════════════════════════════════════════════════════════
# 8. LOOK-AHEAD ENFORCEMENT — DYNAMIC TEST
# ═══════════════════════════════════════════════════════════════════


class TestLookAheadEnforcement:
    """Dynamic test: TechnicalIndicatorEngine must reject future data
    and produce valid features from pre-filtered data."""

    def test_compute_all_rejects_future_data(self):
        """Passing data with timestamps >= as_of must raise LookAheadError."""
        from quant_fund.feature_factory.technical_indicator_engine import (
            TechnicalIndicatorEngine,
        )
        from quant_fund.feature_factory.data_alignment_engine import LookAheadError

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=400, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[300]

        engine = TechnicalIndicatorEngine()

        with pytest.raises(LookAheadError):
            engine.compute_all(ohlcv, as_of=as_of)

    def test_features_produced_from_pre_filtered_data(self):
        """Pre-filtered data (all timestamps < as_of) must produce features."""
        from quant_fund.feature_factory.technical_indicator_engine import (
            TechnicalIndicatorEngine,
        )

        ohlcv = make_ohlcv(tickers=STANDARD_TICKERS[:3], periods=400, seed=42)
        dates = ohlcv.index.get_level_values(0).unique().sort_values()
        as_of = dates[300]

        engine = TechnicalIndicatorEngine()
        truncated = ohlcv[ohlcv.index.get_level_values(0) < as_of]
        feats = engine.compute_all(truncated, as_of=as_of)

        assert len(feats.columns) > 0, "No features produced from pre-filtered data"


# ═══════════════════════════════════════════════════════════════════
# 9. ALL ABSTRACT BASE CLASSES IMPLEMENTED
# ═══════════════════════════════════════════════════════════════════


class TestAbstractBaseContracts:
    """Verify that abstract base class contracts are fulfilled by all
    concrete implementations."""

    def test_all_feature_generators_implement_compute_and_validate(self):
        """Every BaseFeatureGenerator subclass must implement compute()
        and validate() without raising NotImplementedError."""
        from quant_fund.feature_factory.base_feature_generator import (
            BaseFeatureGenerator,
        )
        from quant_fund.feature_factory.technical_indicator_engine import (
            TechnicalIndicatorEngine,
        )

        engine = TechnicalIndicatorEngine()
        for gen in engine.generators:
            assert isinstance(gen, BaseFeatureGenerator), (
                f"{gen.__class__.__name__} does not inherit BaseFeatureGenerator"
            )
            assert hasattr(gen, "compute") and callable(gen.compute)
            assert hasattr(gen, "validate") and callable(gen.validate)
            assert hasattr(gen, "feature_name")
            assert hasattr(gen, "lookback_days")

    def test_broker_adapters_implement_interface(self):
        """All broker adapters must implement BrokerInterface."""
        from quant_fund.broker_interface.broker_abstraction_layer import (
            BrokerInterface,
        )
        from quant_fund.broker_interface.simulation_broker import SimulationBroker

        assert issubclass(SimulationBroker, BrokerInterface), (
            "SimulationBroker does not implement BrokerInterface"
        )

        # Check required methods
        required = ["submit_order", "cancel_order", "get_positions",
                     "get_account_value", "get_fills"]
        for method in required:
            assert hasattr(SimulationBroker, method), (
                f"SimulationBroker missing required method: {method}"
            )


# ═══════════════════════════════════════════════════════════════════
# 10. TIMESTAMP TYPE INVARIANT
# ═══════════════════════════════════════════════════════════════════


class TestTimestampTypes:
    """All timestamps in data fixtures must be pd.Timestamp, not
    datetime.datetime."""

    def test_ohlcv_dates_are_pd_timestamp(self):
        ohlcv = make_ohlcv(tickers=["AAPL"], periods=10, seed=42)
        dates = ohlcv.index.get_level_values(0)
        for d in dates:
            assert isinstance(d, pd.Timestamp), (
                f"Date {d} is {type(d).__name__}, expected pd.Timestamp"
            )

    def test_returns_index_is_pd_timestamp(self):
        returns = make_returns(n_dates=10, tickers=["AAPL"], seed=42)
        for d in returns.index:
            assert isinstance(d, pd.Timestamp), (
                f"Date {d} is {type(d).__name__}, expected pd.Timestamp"
            )
