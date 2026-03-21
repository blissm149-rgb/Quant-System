"""Deterministic seed management for reproducibility.

Provides a central SeedManager that derives deterministic seeds
for each component from a single base seed, ensuring reproducible
results across runs.
"""

import hashlib
from typing import Dict, Optional

import numpy as np


class SeedManager:
    """Manages deterministic seeds for all ML components.

    Given a base seed, derives component-specific seeds using a
    hash-based approach. This ensures:
    - Same base_seed -> same derived seeds -> reproducible results
    - Different components get different seeds (avoiding correlated randomness)
    - Different base_seeds -> different results
    """

    def __init__(self, base_seed: int = 42):
        self._base_seed = base_seed
        self._component_seeds: Dict[str, int] = {}

    def get_seed(self, component: str) -> int:
        """Get a deterministic seed for a named component.

        Args:
            component: Component name (e.g. "gbt_model", "random_forest").

        Returns:
            Integer seed derived from base_seed + component name.
        """
        if component not in self._component_seeds:
            # Hash-based derivation: deterministic and well-distributed
            key = f"{self._base_seed}_{component}"
            hash_val = hashlib.sha256(key.encode()).hexdigest()
            self._component_seeds[component] = int(hash_val[:8], 16) % (2**31)
        return self._component_seeds[component]

    def set_global_seed(self) -> None:
        """Set numpy global random seed from base_seed.

        Useful for ensuring deterministic behavior in code that
        doesn't accept explicit seeds.
        """
        np.random.seed(self._base_seed)

    def get_all_seeds(self) -> Dict[str, int]:
        """Return all derived seeds so far."""
        return dict(self._component_seeds)

    @property
    def base_seed(self) -> int:
        return self._base_seed
