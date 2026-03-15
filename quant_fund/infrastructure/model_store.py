"""Model persistence and versioning layer.

Saves and loads trained ML models with metadata (training dates,
feature lists, performance scores). Supports version comparison
and champion/challenger promotion.

Uses joblib for sklearn models and numpy .npz for raw-weight models
(LSTM, Transformer, Autoencoder).
"""

import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a single model version."""

    version_id: str
    model_name: str
    model_type: str  # "sklearn", "numpy_weights"
    created_at: str
    train_start_date: str
    train_end_date: str
    feature_names: List[str]
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    artifact_path: str
    is_champion: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "created_at": self.created_at,
            "train_start_date": self.train_start_date,
            "train_end_date": self.train_end_date,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "artifact_path": self.artifact_path,
            "is_champion": self.is_champion,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        return cls(**data)


class ModelStore:
    """Persists trained models and manages version history.

    Directory layout::

        <model_dir>/
        ├── <model_name>/
        │   ├── versions/
        │   │   ├── <version_id>/
        │   │   │   ├── model.joblib   (sklearn) or weights.npz (numpy)
        │   │   │   └── metadata.json
        │   │   └── ...
        │   └── champion.json          (points to current champion version)
        └── ...

    Usage:
        store = ModelStore(model_dir="./models")
        vid = store.save_sklearn_model(model, "gbt_alpha", ...)
        loaded = store.load_sklearn_model("gbt_alpha", vid)
        store.promote_to_champion("gbt_alpha", vid)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self._model_dir = Path(cfg.get("model_dir", "./models"))
        self._max_versions: int = cfg.get("max_versions_per_model", 20)
        logger.info(
            "ModelStore initialized model_dir=%s max_versions=%d",
            self._model_dir,
            self._max_versions,
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_sklearn_model(
        self,
        model: Any,
        model_name: str,
        train_start_date: str,
        train_end_date: str,
        feature_names: List[str],
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a sklearn model to disk via joblib.

        Args:
            model: Trained sklearn estimator.
            model_name: Logical model name (e.g., ``"gbt_alpha"``).
            train_start_date: Start of training window (ISO string).
            train_end_date: End of training window (ISO string).
            feature_names: Feature columns used for training.
            metrics: Performance metrics (e.g., ``{"ic": 0.05}``).
            hyperparameters: Model hyperparameters.
            metadata: Additional metadata.

        Returns:
            Version ID string.
        """
        import joblib

        version_id = uuid.uuid4().hex[:16]
        version_dir = self._version_dir(model_name, version_id)
        version_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = str(version_dir / "model.joblib")
        joblib.dump(model, artifact_path)

        version = self._create_version_record(
            version_id=version_id,
            model_name=model_name,
            model_type="sklearn",
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            feature_names=feature_names,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            artifact_path=artifact_path,
            metadata=metadata or {},
        )

        self._save_metadata(version_dir, version)
        self._enforce_retention(model_name)

        logger.info(
            "Saved sklearn model '%s' version %s (%d features, %d bytes)",
            model_name,
            version_id,
            len(feature_names),
            os.path.getsize(artifact_path),
        )
        return version_id

    def save_numpy_weights(
        self,
        weights: Dict[str, np.ndarray],
        model_name: str,
        train_start_date: str,
        train_end_date: str,
        feature_names: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save numpy weight arrays to disk as ``.npz``.

        Suitable for LSTM, Transformer, Autoencoder, and Embedding
        models that store weights as dicts of numpy arrays.

        Args:
            weights: Dict mapping weight names to numpy arrays.
            model_name: Logical model name.
            train_start_date: Start of training window.
            train_end_date: End of training window.
            feature_names: Feature columns (optional for some models).
            metrics: Performance metrics.
            hyperparameters: Model hyperparameters.
            metadata: Additional metadata.

        Returns:
            Version ID string.
        """
        version_id = uuid.uuid4().hex[:16]
        version_dir = self._version_dir(model_name, version_id)
        version_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = str(version_dir / "weights.npz")
        np.savez(artifact_path, **weights)

        version = self._create_version_record(
            version_id=version_id,
            model_name=model_name,
            model_type="numpy_weights",
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            feature_names=feature_names or [],
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            artifact_path=artifact_path,
            metadata=metadata or {},
        )

        self._save_metadata(version_dir, version)
        self._enforce_retention(model_name)

        logger.info(
            "Saved numpy weights '%s' version %s (%d arrays)",
            model_name,
            version_id,
            len(weights),
        )
        return version_id

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_sklearn_model(
        self, model_name: str, version_id: Optional[str] = None
    ) -> Any:
        """Load a sklearn model from disk.

        Args:
            model_name: Logical model name.
            version_id: Specific version to load. If ``None``, loads
                the current champion. If no champion, loads latest.

        Returns:
            The deserialized sklearn estimator.

        Raises:
            FileNotFoundError: If the model or version doesn't exist.
        """
        import joblib

        version = self._resolve_version(model_name, version_id)
        return joblib.load(version.artifact_path)

    def load_numpy_weights(
        self, model_name: str, version_id: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Load numpy weight arrays from disk.

        Args:
            model_name: Logical model name.
            version_id: Specific version to load. If ``None``, loads
                the current champion or latest.

        Returns:
            Dict mapping weight names to numpy arrays.

        Raises:
            FileNotFoundError: If the model or version doesn't exist.
        """
        version = self._resolve_version(model_name, version_id)
        data = np.load(version.artifact_path, allow_pickle=False)
        return dict(data)

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def list_versions(self, model_name: str) -> List[Dict]:
        """List all versions of a model, newest first."""
        model_dir = self._model_dir / model_name / "versions"
        if not model_dir.exists():
            return []

        versions = []
        for vdir in sorted(model_dir.iterdir(), reverse=True):
            meta_path = vdir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    versions.append(json.load(f))

        # Sort by created_at descending
        versions.sort(key=lambda v: v.get("created_at", ""), reverse=True)
        return versions

    def get_version(self, model_name: str, version_id: str) -> Dict:
        """Get metadata for a specific version."""
        meta_path = self._version_dir(model_name, version_id) / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Version {version_id} of model '{model_name}' not found"
            )
        with open(meta_path) as f:
            return json.load(f)

    def get_champion(self, model_name: str) -> Optional[Dict]:
        """Get the current champion version metadata, or None."""
        champion_path = self._model_dir / model_name / "champion.json"
        if not champion_path.exists():
            return None
        with open(champion_path) as f:
            data = json.load(f)
        version_id = data.get("version_id")
        if version_id:
            try:
                return self.get_version(model_name, version_id)
            except FileNotFoundError:
                return None
        return None

    def promote_to_champion(self, model_name: str, version_id: str) -> None:
        """Promote a version to champion.

        The champion is the model version used for live signal
        generation. Only one champion can exist per model name.

        Args:
            model_name: Logical model name.
            version_id: Version to promote.

        Raises:
            FileNotFoundError: If the version doesn't exist.
        """
        # Verify version exists
        self.get_version(model_name, version_id)

        champion_path = self._model_dir / model_name / "champion.json"
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        with open(champion_path, "w") as f:
            json.dump({"version_id": version_id}, f)

        logger.info(
            "Promoted model '%s' version %s to champion",
            model_name,
            version_id,
        )

    def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """Compare two model versions by their recorded metrics.

        Args:
            model_name: Logical model name.
            version_a: First version ID.
            version_b: Second version ID.

        Returns:
            Dict with side-by-side metric comparison and a ``winner``
            field based on primary metric (first metric key).
        """
        meta_a = self.get_version(model_name, version_a)
        meta_b = self.get_version(model_name, version_b)

        metrics_a = meta_a.get("metrics", {})
        metrics_b = meta_b.get("metrics", {})

        all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))

        comparison: Dict[str, Any] = {
            "version_a": version_a,
            "version_b": version_b,
            "metrics": {},
        }

        for key in all_keys:
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            comparison["metrics"][key] = {
                "a": val_a,
                "b": val_b,
                "diff": (val_b - val_a) if val_a is not None and val_b is not None else None,
            }

        # Determine winner by first metric (higher is better assumed)
        if all_keys:
            primary = all_keys[0]
            val_a = metrics_a.get(primary, float("-inf"))
            val_b = metrics_b.get(primary, float("-inf"))
            comparison["winner"] = version_a if val_a >= val_b else version_b
        else:
            comparison["winner"] = None

        return comparison

    def check_challenger_beats_champion(
        self,
        model_name: str,
        challenger_id: str,
        metric_name: str = "ic",
        min_improvement: float = 0.0,
    ) -> Tuple[bool, Dict]:
        """Check if a challenger version outperforms the current champion.

        Args:
            model_name: Logical model name.
            challenger_id: Version ID of the challenger.
            metric_name: Metric to compare on (higher is better).
            min_improvement: Minimum improvement required to beat champion.

        Returns:
            Tuple of (beats_champion, comparison_details).
        """
        champion = self.get_champion(model_name)
        if champion is None:
            return True, {"reason": "no_existing_champion"}

        champion_id = champion["version_id"]
        comparison = self.compare_versions(model_name, champion_id, challenger_id)

        champion_val = champion.get("metrics", {}).get(metric_name, float("-inf"))
        challenger_meta = self.get_version(model_name, challenger_id)
        challenger_val = challenger_meta.get("metrics", {}).get(metric_name, float("-inf"))

        beats = (challenger_val - champion_val) >= min_improvement

        return beats, {
            "champion_id": champion_id,
            "challenger_id": challenger_id,
            "metric": metric_name,
            "champion_value": champion_val,
            "challenger_value": challenger_val,
            "improvement": challenger_val - champion_val,
            "min_required": min_improvement,
            "beats_champion": beats,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _version_dir(self, model_name: str, version_id: str) -> Path:
        return self._model_dir / model_name / "versions" / version_id

    def _create_version_record(self, **kwargs) -> ModelVersion:
        kwargs["created_at"] = datetime.now(timezone.utc).isoformat()
        return ModelVersion(**kwargs)

    def _save_metadata(self, version_dir: Path, version: ModelVersion) -> None:
        meta_path = version_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

    def _resolve_version(
        self, model_name: str, version_id: Optional[str]
    ) -> ModelVersion:
        """Resolve which version to load."""
        if version_id:
            meta = self.get_version(model_name, version_id)
        else:
            champion = self.get_champion(model_name)
            if champion:
                meta = champion
            else:
                versions = self.list_versions(model_name)
                if not versions:
                    raise FileNotFoundError(
                        f"No versions found for model '{model_name}'"
                    )
                meta = versions[0]  # newest

        return ModelVersion.from_dict(meta)

    def _enforce_retention(self, model_name: str) -> None:
        """Remove old versions exceeding the retention limit.

        Never removes the current champion.
        """
        versions = self.list_versions(model_name)
        if len(versions) <= self._max_versions:
            return

        champion = self.get_champion(model_name)
        champion_id = champion["version_id"] if champion else None

        to_remove = versions[self._max_versions:]
        for v in to_remove:
            vid = v["version_id"]
            if vid == champion_id:
                continue
            version_dir = self._version_dir(model_name, vid)
            if version_dir.exists():
                shutil.rmtree(version_dir)
                logger.info(
                    "Evicted model '%s' version %s (retention limit)",
                    model_name,
                    vid,
                )
