#!/usr/bin/env python3
"""Offline model training pipeline.

Runs as a SEPARATE PROCESS from main_run.py -- NEVER called by the live
trading loop. Invoked by cron, Airflow, Prefect, or manual trigger.

Typical schedule: 5:15 PM ET weekdays, after data settlement.

Usage:
    python train_pipeline.py --config config/training.yaml
    python train_pipeline.py --config config/training.yaml --models alpha_momentum
    python train_pipeline.py --config config/training.yaml --force
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quant_fund.infrastructure.data_builder import DataBuilder
from quant_fund.infrastructure.promotion_gate import PromotionGate

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def run_training_pipeline(
    config: dict,
    models_to_retrain: Optional[List[str]] = None,
    force_promote: bool = False,
) -> Dict[str, Any]:
    """Full offline training pipeline.

    Runs after market close. NEVER called by main_run.py.

    Returns dict of results per model.
    """
    results = {}

    # Phase 1: Build training dataset
    logger.info("Phase 1: Building training dataset")
    data_config = config.get("data", {})
    data_builder = DataBuilder(data_config)
    dataset = data_builder.build(
        cutoff_date=config.get("data_cutoff_date"),
        lookback_days=data_config.get("training_lookback_days", 756),
    )
    data_hash = data_builder.compute_hash(dataset)
    logger.info("Dataset built: %d rows, hash=%s", len(dataset), data_hash[:12])

    if dataset.empty:
        logger.warning("Empty dataset -- skipping training")
        return {"status": "skipped", "reason": "empty_dataset"}

    # Phase 2: Train candidate models
    model_configs = config.get("models", {})
    if models_to_retrain:
        model_configs = {
            k: v for k, v in model_configs.items() if k in models_to_retrain
        }

    logger.info("Phase 2: Training %d candidate models", len(model_configs))
    candidates = {}
    for model_name, model_cfg in model_configs.items():
        try:
            candidate = _train_model(model_name, model_cfg, dataset)
            candidates[model_name] = candidate
            logger.info("Trained candidate for %s", model_name)
        except Exception as e:
            logger.error("Training failed for %s: %s", model_name, e)
            results[model_name] = {"status": "training_failed", "error": str(e)}

    # Phase 3: Validate candidates
    logger.info("Phase 3: Validating %d candidates", len(candidates))
    validation_results = {}
    for model_name, candidate in candidates.items():
        try:
            val_results = _validate_model(model_name, candidate, dataset, config)
            validation_results[model_name] = val_results
            logger.info(
                "Validation for %s: sharpe=%.3f, max_dd=%.3f",
                model_name,
                val_results.get("oos_sharpe", 0),
                val_results.get("oos_max_drawdown", 0),
            )
        except Exception as e:
            logger.error("Validation failed for %s: %s", model_name, e)
            results[model_name] = {"status": "validation_failed", "error": str(e)}

    # Phase 4: Promotion gate
    logger.info("Phase 4: Running promotion gate")
    gate_config = config.get("promotion_gate", {})
    gate_config["model_registry_path"] = config.get(
        "model_registry_path", "./models/registry"
    )
    gate = PromotionGate(gate_config)

    for model_name, candidate in candidates.items():
        if model_name not in validation_results:
            continue

        val = validation_results[model_name]
        if force_promote:
            # Skip gate checks, directly promote
            version = gate._atomic_promote(model_name, candidate, val, data_hash)
            results[model_name] = {"status": "force_promoted", "version": version}
            logger.info("FORCE PROMOTED: %s -> %s", model_name, version)
        else:
            version = gate.evaluate(model_name, candidate, val, data_hash)
            if version:
                results[model_name] = {"status": "promoted", "version": version}
                logger.info("PROMOTED: %s -> %s", model_name, version)
            else:
                results[model_name] = {"status": "rejected"}
                logger.warning("REJECTED: %s failed promotion gate", model_name)

    return results


def _train_model(
    model_name: str, model_cfg: dict, dataset
) -> Any:
    """Train a single model. Extend for your framework."""
    model_type = model_cfg.get("type", "gradient_boost")
    hyperparams = model_cfg.get("hyperparams", {})

    if model_type == "gradient_boost":
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(**hyperparams)
        # Training requires features and target -- placeholder for actual impl
        logger.info("Training %s with %s", model_name, model_type)
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _validate_model(
    model_name: str, candidate: Any, dataset, config: dict
) -> Dict[str, float]:
    """Validate a candidate model. Returns metrics dict."""
    # Placeholder -- real implementation runs backtest + walk-forward
    return {
        "oos_sharpe": 0.0,
        "oos_max_drawdown": 0.0,
        "prediction_mean": 0.0,
        "prediction_std": 1.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline model training pipeline (runs OUTSIDE market hours)"
    )
    parser.add_argument(
        "--config", required=True, help="Path to training config YAML"
    )
    parser.add_argument(
        "--models", nargs="+", help="Specific models to retrain"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip promotion gate (force promote all passing candidates)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    results = run_training_pipeline(
        config=config,
        models_to_retrain=args.models,
        force_promote=args.force,
    )

    print("\n" + "=" * 60)
    print("  TRAINING PIPELINE RESULTS")
    print("=" * 60)
    for model_name, result in results.items():
        status = result.get("status", "unknown")
        version = result.get("version", "")
        print(f"  {model_name}: {status} {version}")
    print("=" * 60)


if __name__ == "__main__":
    main()
