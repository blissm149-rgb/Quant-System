"""Configuration validator — schema and bounds checking.

Validates all config values at system startup. Raises
ConfigurationError if any value is out of bounds, wrong type,
or missing required field. Fail-fast: prevent silent misconfiguration.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


@dataclass
class BoundSpec:
    """Specification for a single config parameter bound."""

    name: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    min_exclusive: bool = True  # True means > min, False means >= min
    max_exclusive: bool = False  # True means < max, False means <= max
    required: bool = False
    expected_type: Optional[type] = None


# All known parameter bounds
RISK_BOUNDS = [
    BoundSpec("drawdown_limit", min_val=0.0, max_val=1.0, min_exclusive=True),
    BoundSpec("max_sector_exposure", min_val=0.0, max_val=1.0, min_exclusive=True),
    BoundSpec("max_single_name_exposure", min_val=0.0, max_val=1.0, min_exclusive=True),
    BoundSpec("daily_var_limit", min_val=0.0, max_val=1.0, min_exclusive=True),
]

TRADING_BOUNDS = [
    BoundSpec("max_leverage", min_val=0.0, max_val=10.0, min_exclusive=True),
    BoundSpec("max_position_size", min_val=0.0, max_val=0.5, min_exclusive=True),
    BoundSpec("max_sector_exposure", min_val=0.0, max_val=1.0, min_exclusive=True),
]

EXECUTION_BOUNDS = [
    BoundSpec("participation_rate", min_val=0.0, max_val=0.5, min_exclusive=True),
    BoundSpec("max_order_size_usd", min_val=0.0, max_val=100_000_000, min_exclusive=True),
    BoundSpec("latency_ms", min_val=0.0, max_val=60_000, min_exclusive=False),
]

SYSTEM_BOUNDS = [
    BoundSpec("initial_nav", min_val=0.0, min_exclusive=True, expected_type=float),
]


def _check_bound(value: Any, spec: BoundSpec) -> Optional[str]:
    """Check a single value against a bound spec.

    Returns error message or None if valid.
    """
    if value is None:
        if spec.required:
            return f"{spec.name}: required but missing"
        return None

    if spec.expected_type is not None:
        if not isinstance(value, (int, float, spec.expected_type)):
            return (
                f"{spec.name}: expected {spec.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    if isinstance(value, (int, float)):
        if spec.min_val is not None:
            if spec.min_exclusive and value <= spec.min_val:
                return f"{spec.name}: {value} must be > {spec.min_val}"
            elif not spec.min_exclusive and value < spec.min_val:
                return f"{spec.name}: {value} must be >= {spec.min_val}"

        if spec.max_val is not None:
            if spec.max_exclusive and value >= spec.max_val:
                return f"{spec.name}: {value} must be < {spec.max_val}"
            elif not spec.max_exclusive and value > spec.max_val:
                return f"{spec.name}: {value} must be <= {spec.max_val}"

    return None


def _find_value(config: dict, name: str) -> Any:
    """Find a value in a possibly nested config dict."""
    if name in config:
        return config[name]
    # Search nested dicts
    for v in config.values():
        if isinstance(v, dict):
            result = _find_value(v, name)
            if result is not None:
                return result
    return None


def validate_config(
    config: dict,
    bounds: List[BoundSpec],
    config_name: str = "config",
) -> List[str]:
    """Validate a config dict against bound specifications.

    Parameters
    ----------
    config : dict
        Configuration dictionary to validate.
    bounds : list of BoundSpec
        Bound specifications to check.
    config_name : str
        Name for error messages.

    Returns
    -------
    list of str
        Error messages (empty if valid).
    """
    errors = []
    for spec in bounds:
        value = _find_value(config, spec.name)
        error = _check_bound(value, spec)
        if error is not None:
            errors.append(f"[{config_name}] {error}")
    return errors


class ConfigValidator:
    """Validates all system configuration at startup.

    Call `validate_all()` from entry points (research_runner,
    paper_trading_runner, live_trading_runner) before initializing
    any components. Raises ConfigurationError on invalid config.
    """

    def __init__(self):
        self._custom_bounds: List[Tuple[str, List[BoundSpec]]] = []

    def add_bounds(
        self, config_name: str, bounds: List[BoundSpec]
    ) -> None:
        """Register additional bound checks."""
        self._custom_bounds.append((config_name, bounds))

    def validate_all(self, configs: Dict[str, dict]) -> None:
        """Validate all configs. Raises ConfigurationError if any invalid.

        Parameters
        ----------
        configs : dict
            Mapping config_name → config_dict. Expected keys:
            "risk", "trading", "execution", "system".
        """
        all_errors = []

        if "risk" in configs:
            all_errors.extend(
                validate_config(configs["risk"], RISK_BOUNDS, "risk_config")
            )
        if "trading" in configs:
            all_errors.extend(
                validate_config(configs["trading"], TRADING_BOUNDS, "trading_config")
            )
        if "execution" in configs:
            all_errors.extend(
                validate_config(configs["execution"], EXECUTION_BOUNDS, "execution_config")
            )
        if "system" in configs:
            all_errors.extend(
                validate_config(configs["system"], SYSTEM_BOUNDS, "system_config")
            )

        for name, bounds in self._custom_bounds:
            if name in configs:
                all_errors.extend(
                    validate_config(configs[name], bounds, name)
                )

        if all_errors:
            msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in all_errors
            )
            logger.error(msg)
            raise ConfigurationError(msg)

        logger.info("Configuration validation passed")

    def validate_single(
        self,
        config: dict,
        config_name: str,
        bounds: Optional[List[BoundSpec]] = None,
    ) -> List[str]:
        """Validate a single config dict. Returns errors (doesn't raise)."""
        bound_map = {
            "risk": RISK_BOUNDS,
            "trading": TRADING_BOUNDS,
            "execution": EXECUTION_BOUNDS,
            "system": SYSTEM_BOUNDS,
        }
        b = bounds or bound_map.get(config_name, [])
        return validate_config(config, b, config_name)
