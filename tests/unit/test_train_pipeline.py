"""Integration tests for the offline training pipeline."""

import ast
import importlib
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.tier2]


class TestTrainingPipeline:
    """Tests for the offline training pipeline."""

    def test_pipeline_never_imports_main_run(self):
        """Verify no import dependency on the live trading loop.

        The offline pipeline must be completely independent of main_run.py.
        """
        pipeline_path = Path(__file__).parents[2] / "train_pipeline.py"
        source = pipeline_path.read_text()
        tree = ast.parse(source)

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)

        # Must not import main_run or trading_engine
        assert "main_run" not in imported_modules
        for mod in imported_modules:
            assert "main_run" not in mod, f"Pipeline imports main_run via {mod}"
            assert "trading_engine" not in mod, f"Pipeline imports trading_engine via {mod}"

    def test_pipeline_modules_importable(self):
        """Verify all pipeline dependencies can be imported."""
        from quant_fund.infrastructure.data_builder import DataBuilder
        from quant_fund.infrastructure.promotion_gate import PromotionGate

        assert DataBuilder is not None
        assert PromotionGate is not None

    def test_load_config_missing_file(self):
        """load_config raises FileNotFoundError for missing file."""
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from train_pipeline import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_run_training_pipeline_empty_dataset(self, tmp_path):
        """Pipeline handles empty dataset gracefully."""
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from train_pipeline import run_training_pipeline

        config = {
            "data": {"warehouse_path": str(tmp_path)},
            "models": {},
            "promotion_gate": {},
            "model_registry_path": str(tmp_path / "registry"),
        }

        results = run_training_pipeline(config)
        assert results.get("status") == "skipped"

    def test_main_run_never_imports_train_pipeline(self):
        """Verify main_run.py does not import train_pipeline.

        This is the key invariant: the live loop has no dependency
        on the training pipeline.
        """
        main_run_path = Path(__file__).parents[2] / "main_run.py"
        source = main_run_path.read_text()
        tree = ast.parse(source)

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)

        assert "train_pipeline" not in imported_modules
        for mod in imported_modules:
            assert "train_pipeline" not in mod

    def test_research_runner_no_retrain_in_loop(self):
        """Verify research_runner.run_cycle does not call model.fit/train."""
        runner_path = (
            Path(__file__).parents[2]
            / "quant_fund"
            / "main"
            / "research_runner.py"
        )
        source = runner_path.read_text()
        tree = ast.parse(source)

        # Find the run_cycle method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_cycle":
                # Check that run_cycle doesn't call _retrain_models
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            assert child.func.attr != "_retrain_models", (
                                "run_cycle must not call _retrain_models"
                            )
