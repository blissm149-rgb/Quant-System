"""Section 10.2: Config Fuzzer

Generates edge-case config combinations for negative testing.
Boundary values for all numeric parameters.
Invalid type combinations.
"""

from typing import Any, Dict, List, Optional
import itertools
import numpy as np


def make_boundary_config(param_name: str) -> List[Dict[str, Any]]:
    """Generate boundary values for a specific config parameter.

    Returns list of configs with the parameter set to boundary values.
    """
    boundaries = {
        "max_leverage": [0.0, 0.001, 1.0, 2.0, 2.0 - 1e-10, 2.0 + 1e-10, 10.0, 100.0, float("inf")],
        "max_position_size": [0.0, 0.001, 0.02, 0.02 - 1e-10, 0.02 + 1e-10, 0.5, 1.0],
        "max_sector_exposure": [0.0, 0.01, 0.20, 0.20 - 1e-10, 0.20 + 1e-10, 0.5, 1.0],
        "drawdown_limit": [0.0, 0.001, 0.10, 0.15, 0.20, 0.20 - 1e-10, 0.20 + 1e-10, 0.5, 1.0],
        "drawdown_warning": [0.0, 0.05, 0.10, 0.10 - 1e-10, 0.10 + 1e-10, 0.20],
        "drawdown_alert": [0.0, 0.10, 0.15, 0.15 - 1e-10, 0.15 + 1e-10, 0.25],
        "initial_nav": [0.0, 1.0, 100.0, 1_000_000.0, 1e12],
        "min_ic": [0.0, 0.01, 0.03, 0.05, 0.10, 1.0],
        "min_ic_tstat": [0.0, 1.0, 2.0, 3.0, 10.0],
        "min_eval_days": [0, 1, 126, 252, 500, 1000],
        "min_sharpe": [0.0, 0.5, 1.0, 1.5, 2.0, 5.0],
        "max_drawdown": [0.0, 0.10, 0.25, 0.50, 1.0],
        "min_paper_trading_days": [0, 1, 63, 126, 252],
    }

    if param_name not in boundaries:
        return [{}]

    return [{param_name: v} for v in boundaries[param_name]]


def make_invalid_type_configs() -> List[Dict[str, Any]]:
    """Generate configs with invalid types for negative testing.

    Each config has one parameter set to an invalid type.
    Tests should verify these are rejected gracefully.
    """
    invalid_values: List[Dict[str, Any]] = [
        {"max_leverage": "two"},
        {"max_leverage": None},
        {"max_leverage": [2.0]},
        {"max_leverage": -1.0},
        {"max_position_size": "small"},
        {"max_position_size": -0.01},
        {"drawdown_limit": "twenty_percent"},
        {"drawdown_limit": -0.20},
        {"initial_nav": -1_000_000.0},
        {"min_eval_days": -1},
        {"min_eval_days": 3.5},
        {"min_sharpe": "high"},
    ]
    return invalid_values


def make_extreme_configs() -> List[Dict[str, Any]]:
    """Generate configs with extreme but valid values.

    These test that the system handles extreme parameters gracefully
    without crashing (even if results are degenerate).
    """
    return [
        # Extremely tight constraints
        {
            "max_leverage": 0.001,
            "max_position_size": 0.0001,
            "max_sector_exposure": 0.001,
        },
        # Extremely loose constraints
        {
            "max_leverage": 100.0,
            "max_position_size": 1.0,
            "max_sector_exposure": 1.0,
        },
        # Zero risk tolerance
        {
            "drawdown_limit": 0.001,
            "drawdown_warning": 0.0005,
            "drawdown_alert": 0.0008,
        },
        # Huge universe
        {
            "max_leverage": 2.0,
            "max_position_size": 0.001,  # 1/1000
            "max_sector_exposure": 0.10,
        },
        # Single-name portfolio
        {
            "max_leverage": 1.0,
            "max_position_size": 1.0,
            "max_sector_exposure": 1.0,
            "dollar_neutral": False,
        },
    ]


def make_combinatorial_configs(
    param_space: Optional[Dict[str, List[Any]]] = None,
    max_combos: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate random combinations from a parameter space.

    If the full Cartesian product exceeds max_combos, sample randomly.
    """
    if param_space is None:
        param_space = {
            "max_leverage": [1.0, 2.0, 5.0],
            "max_position_size": [0.01, 0.02, 0.05],
            "max_sector_exposure": [0.10, 0.20, 0.40],
            "dollar_neutral": [True, False],
        }

    keys = list(param_space.keys())
    values = list(param_space.values())
    all_combos = list(itertools.product(*values))

    if len(all_combos) <= max_combos:
        return [dict(zip(keys, combo)) for combo in all_combos]

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_combos), size=max_combos, replace=False)
    return [dict(zip(keys, all_combos[i])) for i in indices]
