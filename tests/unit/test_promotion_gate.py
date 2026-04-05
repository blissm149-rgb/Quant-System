"""Tests for the candidate model promotion system."""

import json
import os
import pickle

import pytest

from quant_fund.infrastructure.promotion_gate import PromotionGate


class TestPromotionGate:
    """Tests for the candidate model promotion system."""

    def _make_gate(self, tmp_registry, **overrides):
        config = {
            "model_registry_path": str(tmp_registry),
            "min_oos_sharpe": 0.5,
            "max_oos_drawdown": -0.15,
            "min_improvement_pct": 0.02,
            "require_human_approval": False,
        }
        config.update(overrides)
        return PromotionGate(config)

    def test_candidate_promoted_when_all_gates_pass(self, tmp_registry):
        """Good candidate model is promoted and symlink is updated."""
        gate = self._make_gate(tmp_registry)
        candidate = {"weights": [1, 2, 3]}
        validation = {
            "oos_sharpe": 1.5,
            "oos_max_drawdown": -0.05,
        }

        version = gate.evaluate("new_model", candidate, validation, "datahash123")
        assert version is not None
        assert version.startswith("v")

        # Verify symlink exists
        prod_link = tmp_registry / "production" / "new_model"
        assert prod_link.exists() or prod_link.is_symlink()

    def test_candidate_rejected_below_min_sharpe(self, tmp_registry):
        """Candidate with Sharpe < threshold is rejected."""
        gate = self._make_gate(tmp_registry)
        candidate = {"weights": [1, 2, 3]}
        validation = {
            "oos_sharpe": 0.3,  # Below 0.5 threshold
            "oos_max_drawdown": -0.05,
        }

        version = gate.evaluate("test_model", candidate, validation, "hash")
        assert version is None

    def test_candidate_rejected_excessive_drawdown(self, tmp_registry):
        """Candidate exceeding max drawdown is rejected."""
        gate = self._make_gate(tmp_registry)
        candidate = {"weights": [1, 2, 3]}
        validation = {
            "oos_sharpe": 1.5,
            "oos_max_drawdown": -0.25,  # Worse than -0.15 threshold
        }

        version = gate.evaluate("test_model", candidate, validation, "hash")
        assert version is None

    def test_candidate_rejected_insufficient_improvement(self, tmp_registry):
        """Candidate that doesn't beat incumbent by min_improvement is rejected."""
        gate = self._make_gate(tmp_registry)

        # The existing test_model in tmp_registry has oos_sharpe=1.2
        # A candidate with 1.21 only improves by ~0.8%, below 2% threshold
        candidate = {"weights": [4, 5, 6]}
        validation = {
            "oos_sharpe": 1.21,
            "oos_max_drawdown": -0.05,
        }

        version = gate.evaluate("test_model", candidate, validation, "hash")
        assert version is None

    def test_candidate_beats_incumbent_and_promoted(self, tmp_registry):
        """Candidate that beats incumbent by enough is promoted."""
        gate = self._make_gate(tmp_registry)

        # Incumbent has oos_sharpe=1.2, need > 2% improvement = 1.224+
        candidate = {"weights": [4, 5, 6]}
        validation = {
            "oos_sharpe": 1.5,  # 25% improvement
            "oos_max_drawdown": -0.05,
        }

        version = gate.evaluate("test_model", candidate, validation, "hash")
        assert version is not None

    def test_atomic_symlink_swap(self, tmp_registry):
        """Verify os.rename atomicity -- no partial state visible."""
        gate = self._make_gate(tmp_registry)
        candidate = {"weights": [1, 2, 3]}
        validation = {
            "oos_sharpe": 1.5,
            "oos_max_drawdown": -0.05,
        }

        version = gate.evaluate("atomic_test", candidate, validation, "hash")
        assert version is not None

        prod_link = tmp_registry / "production" / "atomic_test"
        assert prod_link.is_symlink()
        resolved = prod_link.resolve()
        assert resolved.exists()
        assert (resolved / "model.pkl").exists()

    def test_rejected_candidate_logged(self, tmp_registry):
        """Failed candidates are logged to rejections file."""
        gate = self._make_gate(tmp_registry)
        candidate = {"weights": [1, 2, 3]}
        validation = {
            "oos_sharpe": 0.1,  # Will be rejected
            "oos_max_drawdown": -0.05,
        }

        gate.evaluate("bad_model", candidate, validation, "hash")

        rejection_log = tmp_registry / "retired" / "rejections.jsonl"
        assert rejection_log.exists()
        lines = rejection_log.read_text().strip().split("\n")
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["model_name"] == "bad_model"

    def test_metadata_written_with_promotion(self, tmp_registry):
        """Verify metadata.json and validation.json written to version dir."""
        gate = self._make_gate(tmp_registry)
        candidate = {"weights": [1, 2, 3]}
        validation = {
            "oos_sharpe": 1.5,
            "oos_max_drawdown": -0.05,
        }

        version = gate.evaluate("meta_test", candidate, validation, "datahash_xyz")
        assert version is not None

        version_dir = tmp_registry / "versions" / "meta_test" / version
        assert (version_dir / "metadata.json").exists()
        assert (version_dir / "validation.json").exists()

        meta = json.loads((version_dir / "metadata.json").read_text())
        assert meta["data_hash"] == "datahash_xyz"
        assert meta["validation"]["oos_sharpe"] == 1.5

    def test_data_hash_recorded_for_reproducibility(self, tmp_registry):
        """Training data hash is stored in metadata for audit trail."""
        gate = self._make_gate(tmp_registry)
        candidate = {"weights": [1, 2, 3]}
        validation = {
            "oos_sharpe": 2.0,
            "oos_max_drawdown": -0.03,
        }

        version = gate.evaluate("hash_test", candidate, validation, "unique_hash_42")
        version_dir = tmp_registry / "versions" / "hash_test" / version
        meta = json.loads((version_dir / "metadata.json").read_text())
        assert meta["data_hash"] == "unique_hash_42"

    def test_next_version_increments(self, tmp_registry):
        """Version numbering increments correctly."""
        gate = self._make_gate(tmp_registry)
        # test_model already has v1
        next_v = gate._next_version("test_model")
        assert next_v == "v2"

    def test_regime_robustness_pass(self, tmp_registry):
        """Walk-forward with enough positive windows passes."""
        gate = self._make_gate(tmp_registry)
        validation = {
            "walk_forward_sharpes": [0.5, 0.3, -0.1, 0.8, 0.2],
        }
        assert gate._check_regime_robustness(validation) is True

    def test_regime_robustness_fail(self, tmp_registry):
        """Walk-forward with too many negative windows fails."""
        gate = self._make_gate(tmp_registry)
        validation = {
            "walk_forward_sharpes": [0.5, -0.3, -0.1, -0.8, -0.2],
        }
        assert gate._check_regime_robustness(validation) is False

    def test_regime_robustness_no_data_passes(self, tmp_registry):
        """No walk-forward data passes by default."""
        gate = self._make_gate(tmp_registry)
        assert gate._check_regime_robustness({}) is True
