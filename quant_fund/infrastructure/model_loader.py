"""Read-only model registry for the live trading loop.

Loads versioned model artifacts from the production artifact store.
Supports hot-reload via symlink change detection (cheap stat() calls).
The live loop NEVER writes to this store — only the offline pipeline does.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Immutable container for a loaded model and its metadata."""

    name: str
    version: str
    weights: Any
    metadata: Dict
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            raise ValueError(
                f"Model {self.name} v{self.version} loaded without checksum"
            )


class ModelRegistry:
    """Read-only interface to the model artifact store.

    Used ONLY by the live trading loop. The offline pipeline writes
    versioned artifacts; this class reads them via production symlinks.

    Directory layout::

        <registry_path>/
        +-- production/           # symlinks to current live models
        |   +-- alpha_momentum -> ../versions/alpha_momentum/v23/
        +-- staging/              # candidate models awaiting promotion
        +-- versions/             # immutable versioned artifacts
        |   +-- alpha_momentum/
        |       +-- v23/
        |           +-- model.pkl
        |           +-- metadata.json
        +-- retired/              # superseded or failed models
    """

    def __init__(self, registry_path: str, auto_reload: bool = True):
        self.registry_path = Path(registry_path)
        self.production_path = self.registry_path / "production"
        self._models: Dict[str, LoadedModel] = {}
        self._auto_reload = auto_reload

    def load_model(self, model_name: str) -> LoadedModel:
        """Load the current production version of a model."""
        prod_link = self.production_path / model_name
        if not prod_link.exists():
            raise FileNotFoundError(f"No production model: {model_name}")

        resolved = prod_link.resolve()
        weights_file = self._find_weights_file(resolved)
        metadata_file = resolved / "metadata.json"

        checksum = self._compute_checksum(weights_file)
        version = resolved.name

        # Skip reload if same checksum
        if model_name in self._models:
            if self._models[model_name].checksum == checksum:
                return self._models[model_name]

        weights = self._deserialize_weights(weights_file)
        metadata = (
            json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        )

        loaded = LoadedModel(
            name=model_name,
            version=version,
            weights=weights,
            metadata=metadata,
            checksum=checksum,
        )
        self._models[model_name] = loaded
        logger.info(
            "Loaded model %s %s (checksum: %s)", model_name, version, checksum[:12]
        )
        return loaded

    def check_for_updates(self) -> List[str]:
        """Check if any production symlinks have changed.

        Returns list of model names that were reloaded.
        Called once per iteration of the main loop -- cheap stat() calls only.
        """
        reloaded = []
        for model_name, current in list(self._models.items()):
            prod_link = self.production_path / model_name
            if not prod_link.exists():
                continue
            resolved = prod_link.resolve()
            weights_file = self._find_weights_file(resolved)
            checksum = self._compute_checksum(weights_file)
            if checksum != current.checksum:
                self.load_model(model_name)
                reloaded.append(model_name)
                logger.info("Hot-reloaded model %s -> %s", model_name, resolved.name)
        return reloaded

    def get_model(self, model_name: str) -> LoadedModel:
        """Get a previously loaded model. Does not hit disk."""
        if model_name not in self._models:
            return self.load_model(model_name)
        return self._models[model_name]

    def load_specific_version(self, model_name: str, version: str) -> LoadedModel:
        """Load a specific version (for fallback on degradation)."""
        version_dir = self.registry_path / "versions" / model_name / version
        if not version_dir.exists():
            raise FileNotFoundError(
                f"Version {version} not found for model {model_name}"
            )

        weights_file = self._find_weights_file(version_dir)
        metadata_file = version_dir / "metadata.json"
        checksum = self._compute_checksum(weights_file)
        weights = self._deserialize_weights(weights_file)
        metadata = (
            json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        )

        loaded = LoadedModel(
            name=model_name,
            version=version,
            weights=weights,
            metadata=metadata,
            checksum=checksum,
        )
        self._models[model_name] = loaded
        logger.info("Loaded fallback model %s %s", model_name, version)
        return loaded

    @staticmethod
    def _compute_checksum(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _find_weights_file(model_dir: Path) -> Path:
        for ext in (".pkl", ".pt", ".onnx", ".joblib", ".h5"):
            candidate = model_dir / f"model{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No weights file in {model_dir}")

    @staticmethod
    def _deserialize_weights(path: Path) -> Any:
        """Deserialize model weights based on file extension."""
        suffix = path.suffix
        if suffix == ".pkl":
            import pickle

            with open(path, "rb") as f:
                return pickle.load(f)
        elif suffix == ".joblib":
            import joblib

            return joblib.load(path)
        elif suffix == ".pt":
            import torch

            return torch.load(path, map_location="cpu", weights_only=True)
        else:
            raise ValueError(f"Unsupported weights format: {suffix}")
