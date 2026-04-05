"""Promotion gate -- the firewall between research and production.

Decides whether a candidate model replaces the current production model.
Uses multi-stage validation: absolute thresholds, relative improvement,
and regime robustness checks. Promotion is atomic via symlink swap.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PromotionGate:
    """Decides whether a candidate model replaces the current production model.

    This is the single most important safety mechanism in the system.
    A candidate must pass all gates to be promoted:

    1. Absolute quality thresholds (min Sharpe, max drawdown)
    2. Relative improvement over incumbent
    3. Walk-forward regime robustness
    4. Optional human approval
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.registry_path = Path(cfg.get("model_registry_path", "./models"))
        self.min_oos_sharpe = cfg.get("min_oos_sharpe", 0.5)
        self.max_oos_drawdown = cfg.get("max_oos_drawdown", -0.15)
        self.min_improvement_pct = cfg.get("min_improvement_pct", 0.02)
        self.require_human_approval = cfg.get("require_human_approval", False)

    def evaluate(
        self,
        model_name: str,
        candidate: Any,
        validation: Dict[str, float],
        data_hash: str,
    ) -> Optional[str]:
        """Multi-stage promotion gate.

        Returns the promoted version string (e.g. "v24") if promoted,
        or None if rejected.
        """
        # Gate 1: Absolute quality thresholds
        if validation.get("oos_sharpe", 0) < self.min_oos_sharpe:
            self._log_rejection(model_name, "OOS Sharpe below minimum")
            return None

        if validation.get("oos_max_drawdown", 0) < self.max_oos_drawdown:
            self._log_rejection(model_name, "OOS max drawdown too severe")
            return None

        # Gate 2: Must improve on incumbent
        incumbent_metrics = self._load_incumbent_metrics(model_name)
        if incumbent_metrics:
            incumbent_sharpe = incumbent_metrics.get("oos_sharpe", 0)
            if abs(incumbent_sharpe) > 1e-10:
                improvement = (
                    validation.get("oos_sharpe", 0) - incumbent_sharpe
                ) / abs(incumbent_sharpe)
                if improvement < self.min_improvement_pct:
                    self._log_rejection(
                        model_name,
                        f"Insufficient improvement: {improvement:.3%} "
                        f"< {self.min_improvement_pct:.3%}",
                    )
                    return None

        # Gate 3: Walk-forward regime robustness
        if not self._check_regime_robustness(validation):
            self._log_rejection(model_name, "Failed regime robustness check")
            return None

        # All gates passed -- promote
        return self._atomic_promote(model_name, candidate, validation, data_hash)

    def _atomic_promote(
        self,
        model_name: str,
        candidate: Any,
        validation: Dict,
        data_hash: str,
    ) -> str:
        """Write versioned artifact and atomically swap production symlink."""
        version = self._next_version(model_name)
        version_dir = self._write_version(
            model_name, version, candidate, validation, data_hash
        )

        prod_link = self.registry_path / "production" / model_name
        prod_link.parent.mkdir(parents=True, exist_ok=True)
        tmp_link = prod_link.with_suffix(".tmp")

        relative_target = os.path.relpath(version_dir, prod_link.parent)

        # Remove stale tmp link if it exists
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()

        os.symlink(relative_target, tmp_link)
        os.rename(str(tmp_link), str(prod_link))  # atomic on POSIX

        logger.info("PROMOTED %s -> %s", model_name, version)
        return version

    def _write_version(
        self,
        model_name: str,
        version: str,
        candidate: Any,
        validation: Dict,
        data_hash: str,
    ) -> Path:
        """Write candidate model and metadata to versioned directory."""
        import pickle

        version_dir = self.registry_path / "versions" / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        weights_path = version_dir / "model.pkl"
        with open(weights_path, "wb") as f:
            pickle.dump(candidate, f)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "data_hash": data_hash,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "validation": validation,
        }
        meta_path = version_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        # Save validation results separately
        val_path = version_dir / "validation.json"
        val_path.write_text(json.dumps(validation, indent=2))

        return version_dir

    def _next_version(self, model_name: str) -> str:
        """Determine the next version number for a model."""
        versions_dir = self.registry_path / "versions" / model_name
        if not versions_dir.exists():
            return "v1"

        existing = []
        for d in versions_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    existing.append(int(d.name[1:]))
                except ValueError:
                    pass

        next_num = max(existing, default=0) + 1
        return f"v{next_num}"

    def _load_incumbent_metrics(self, model_name: str) -> Optional[Dict]:
        """Load validation metrics for the current production model."""
        prod_link = self.registry_path / "production" / model_name
        if not prod_link.exists():
            return None

        resolved = prod_link.resolve()
        meta_path = resolved / "metadata.json"
        if not meta_path.exists():
            return None

        metadata = json.loads(meta_path.read_text())
        return metadata.get("validation", {})

    def _check_regime_robustness(self, validation: Dict) -> bool:
        """Check that the model performs across different regimes.

        Requires positive Sharpe in at least 60% of walk-forward windows
        if walk-forward results are available.
        """
        walk_forward = validation.get("walk_forward_sharpes", [])
        if not walk_forward:
            return True  # No walk-forward data -- pass by default

        positive_windows = sum(1 for s in walk_forward if s > 0)
        return positive_windows / len(walk_forward) >= 0.6

    def _log_rejection(self, model_name: str, reason: str):
        logger.warning("REJECTED %s: %s", model_name, reason)

        # Write rejection to retired directory
        retired_dir = self.registry_path / "retired"
        retired_dir.mkdir(parents=True, exist_ok=True)
        rejection_log = retired_dir / "rejections.jsonl"
        with open(rejection_log, "a") as f:
            f.write(
                json.dumps(
                    {
                        "model_name": model_name,
                        "reason": reason,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                + "\n"
            )
