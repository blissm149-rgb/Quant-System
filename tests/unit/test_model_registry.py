"""Tests for the read-only model artifact store (ModelRegistry)."""

import json
import os
import pickle
import threading

import pytest

from quant_fund.infrastructure.model_loader import LoadedModel, ModelRegistry


class TestLoadedModel:
    """Tests for the LoadedModel dataclass."""

    def test_loaded_model_requires_checksum(self):
        with pytest.raises(ValueError, match="loaded without checksum"):
            LoadedModel(
                name="test",
                version="v1",
                weights={"w": 1},
                metadata={},
                checksum="",
            )

    def test_loaded_model_with_checksum(self):
        m = LoadedModel(
            name="test",
            version="v1",
            weights={"w": 1},
            metadata={"key": "val"},
            checksum="abc123",
        )
        assert m.name == "test"
        assert m.version == "v1"
        assert m.checksum == "abc123"


class TestModelRegistry:
    """Tests for the read-only model artifact store."""

    def test_load_model_from_production_symlink(self, tmp_registry):
        """Verify that loading follows symlink to correct versioned directory."""
        registry = ModelRegistry(registry_path=str(tmp_registry))
        loaded = registry.load_model("test_model")
        assert loaded.name == "test_model"
        assert loaded.version == "v1"
        assert loaded.weights == {"coefficients": [0.1, 0.2, 0.3]}

    def test_load_model_returns_correct_checksum(self, tmp_registry):
        """Verify SHA-256 checksum matches the actual weights file."""
        registry = ModelRegistry(registry_path=str(tmp_registry))
        loaded = registry.load_model("test_model")
        assert len(loaded.checksum) == 64  # SHA-256 hex digest
        assert loaded.checksum != ""

    def test_check_for_updates_detects_symlink_change(self, tmp_registry):
        """Simulate promotion (symlink swap) and verify hot-reload detects it."""
        registry = ModelRegistry(registry_path=str(tmp_registry))
        registry.load_model("test_model")
        original_checksum = registry.get_model("test_model").checksum

        # Create v2 with different weights
        v2_dir = tmp_registry / "versions" / "test_model" / "v2"
        v2_dir.mkdir(parents=True)
        new_weights = {"coefficients": [0.4, 0.5, 0.6]}
        with open(v2_dir / "model.pkl", "wb") as f:
            pickle.dump(new_weights, f)
        (v2_dir / "metadata.json").write_text(json.dumps({"version": "v2"}))

        # Atomic symlink swap
        prod_link = tmp_registry / "production" / "test_model"
        tmp_link = prod_link.with_suffix(".tmp")
        os.symlink("../versions/test_model/v2/", tmp_link)
        os.rename(str(tmp_link), str(prod_link))

        reloaded = registry.check_for_updates()
        assert "test_model" in reloaded
        assert registry.get_model("test_model").checksum != original_checksum
        assert registry.get_model("test_model").weights == new_weights

    def test_check_for_updates_no_reload_if_unchanged(self, tmp_registry):
        """Verify no unnecessary deserialization when nothing changed."""
        registry = ModelRegistry(registry_path=str(tmp_registry))
        registry.load_model("test_model")

        reloaded = registry.check_for_updates()
        assert reloaded == []

    def test_load_missing_model_raises(self, tmp_registry):
        """FileNotFoundError when model_name has no production symlink."""
        registry = ModelRegistry(registry_path=str(tmp_registry))
        with pytest.raises(FileNotFoundError, match="No production model"):
            registry.load_model("nonexistent_model")

    def test_load_model_with_corrupt_weights_raises(self, tmp_registry):
        """Verify graceful failure on corrupt/truncated weights file."""
        # Corrupt the weights file
        weights_path = tmp_registry / "versions" / "test_model" / "v1" / "model.pkl"
        weights_path.write_bytes(b"corrupted data")

        registry = ModelRegistry(registry_path=str(tmp_registry))
        with pytest.raises(Exception):
            registry.load_model("test_model")

    def test_concurrent_read_during_symlink_swap(self, tmp_registry):
        """Simulate atomic rename race condition -- should always get valid model."""
        registry = ModelRegistry(registry_path=str(tmp_registry))

        errors = []
        results = []

        def reader():
            try:
                loaded = registry.load_model("test_model")
                results.append(loaded)
            except Exception as e:
                errors.append(e)

        # Create v2
        v2_dir = tmp_registry / "versions" / "test_model" / "v2"
        v2_dir.mkdir(parents=True)
        new_weights = {"coefficients": [0.7, 0.8, 0.9]}
        with open(v2_dir / "model.pkl", "wb") as f:
            pickle.dump(new_weights, f)
        (v2_dir / "metadata.json").write_text(json.dumps({}))

        # Read concurrently while swapping
        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()

        # Swap in the middle
        prod_link = tmp_registry / "production" / "test_model"
        tmp_link = prod_link.with_suffix(".tmp")
        os.symlink("../versions/test_model/v2/", tmp_link)
        os.rename(str(tmp_link), str(prod_link))

        for t in threads:
            t.join()

        # All reads should succeed (get either v1 or v2, never corrupt)
        assert len(errors) == 0
        assert len(results) == 10

    def test_checksum_mismatch_triggers_reload(self, tmp_registry):
        """Verify that modifying weights file (same path) triggers reload."""
        registry = ModelRegistry(registry_path=str(tmp_registry))
        registry.load_model("test_model")
        original_checksum = registry.get_model("test_model").checksum

        # Overwrite weights in-place
        weights_path = tmp_registry / "versions" / "test_model" / "v1" / "model.pkl"
        new_weights = {"coefficients": [9.9, 8.8, 7.7]}
        with open(weights_path, "wb") as f:
            pickle.dump(new_weights, f)

        reloaded = registry.check_for_updates()
        assert "test_model" in reloaded
        assert registry.get_model("test_model").checksum != original_checksum

    def test_get_model_loads_on_first_access(self, tmp_registry):
        """get_model loads from disk if not previously loaded."""
        registry = ModelRegistry(registry_path=str(tmp_registry))
        loaded = registry.get_model("test_model")
        assert loaded.name == "test_model"

    def test_load_model_reads_metadata(self, tmp_registry):
        """Verify metadata is loaded from metadata.json."""
        registry = ModelRegistry(registry_path=str(tmp_registry))
        loaded = registry.load_model("test_model")
        assert loaded.metadata.get("data_hash") == "abc123"
        assert loaded.metadata["validation"]["oos_sharpe"] == 1.2

    def test_load_specific_version(self, tmp_registry):
        """Verify loading a specific version for fallback."""
        # Create v2
        v2_dir = tmp_registry / "versions" / "test_model" / "v2"
        v2_dir.mkdir(parents=True)
        v2_weights = {"coefficients": [0.4, 0.5, 0.6]}
        with open(v2_dir / "model.pkl", "wb") as f:
            pickle.dump(v2_weights, f)
        (v2_dir / "metadata.json").write_text(json.dumps({"version": "v2"}))

        registry = ModelRegistry(registry_path=str(tmp_registry))
        loaded = registry.load_specific_version("test_model", "v2")
        assert loaded.version == "v2"
        assert loaded.weights == v2_weights

    def test_load_specific_version_missing_raises(self, tmp_registry):
        """FileNotFoundError for nonexistent version."""
        registry = ModelRegistry(registry_path=str(tmp_registry))
        with pytest.raises(FileNotFoundError):
            registry.load_specific_version("test_model", "v999")
