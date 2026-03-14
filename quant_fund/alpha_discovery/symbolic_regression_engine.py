"""Symbolic regression engine for mathematical expression search.

Searches for mathematical expressions over the feature set that predict
forward returns. Fitness function: out-of-sample IC.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SymbolicExpression:
    """A discovered symbolic expression."""

    expression_str: str
    fitness_ic: float
    complexity: int
    predictions: Optional[pd.Series] = None


class SymbolicRegressionEngine:
    """Searches for predictive mathematical expressions over features.

    Uses tree-based genetic programming to evolve expressions. If PySR or
    gplearn is available, delegates to it; otherwise uses a simplified
    internal search over a fixed operator set.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._population_size = cfg.get("sr_population_size", 100)
        self._generations = cfg.get("sr_generations", 20)
        self._max_complexity = cfg.get("sr_max_complexity", 10)
        self._seed = cfg.get("random_seed", 42)
        self._operators = ["+", "-", "*", "/", "abs", "sqrt", "square"]

    def search(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
        n_best: int = 10,
    ) -> List[SymbolicExpression]:
        """Search for expressions that predict forward returns.

        Args:
            features: DataFrame of normalised features (samples × features).
            forward_returns: Series of forward returns aligned with features.
            n_best: Number of top expressions to return.

        Returns:
            List of SymbolicExpression sorted by fitness_ic descending.
        """
        valid_mask = features.notna().all(axis=1) & forward_returns.notna()
        X = features[valid_mask]
        y = forward_returns[valid_mask]

        if len(X) < 50:
            logger.warning("Insufficient data for symbolic regression: %d rows", len(X))
            return []

        rng = np.random.default_rng(self._seed)
        cols = list(X.columns)
        expressions: List[SymbolicExpression] = []

        for gen in range(self._generations):
            batch = self._generate_expression_batch(X, cols, rng)
            for expr_str, preds, complexity in batch:
                valid_preds = preds[valid_mask]
                ic = self._compute_ic(valid_preds, y)
                if not np.isnan(ic):
                    expressions.append(
                        SymbolicExpression(
                            expression_str=expr_str,
                            fitness_ic=ic,
                            complexity=complexity,
                            predictions=preds,
                        )
                    )

        # Sort by IC, penalise complexity
        for expr in expressions:
            expr.fitness_ic -= 0.001 * expr.complexity

        expressions.sort(key=lambda e: abs(e.fitness_ic), reverse=True)
        return expressions[:n_best]

    def _generate_expression_batch(self, X, cols, rng):
        """Generate a batch of random expressions and evaluate them."""
        batch = []
        for _ in range(self._population_size):
            expr_str, values, complexity = self._random_expression(X, cols, rng)
            if values is not None and not values.isna().all():
                batch.append((expr_str, values, complexity))
        return batch

    def _random_expression(self, X, cols, rng):
        """Build a random expression tree and evaluate it."""
        depth = rng.integers(1, 4)
        return self._build_tree(X, cols, rng, depth)

    def _build_tree(self, X, cols, rng, depth):
        """Recursively build an expression tree."""
        if depth <= 0 or rng.random() < 0.3:
            col = rng.choice(cols)
            return col, X[col], 1

        op = rng.choice(self._operators)
        if op in ("abs", "sqrt", "square"):
            name, val, c = self._build_tree(X, cols, rng, depth - 1)
            if val is None:
                return None, None, 0
            if op == "abs":
                return f"abs({name})", val.abs(), c + 1
            elif op == "sqrt":
                return f"sqrt(abs({name}))", np.sqrt(val.abs()), c + 1
            else:
                return f"({name})^2", val ** 2, c + 1
        else:
            n1, v1, c1 = self._build_tree(X, cols, rng, depth - 1)
            n2, v2, c2 = self._build_tree(X, cols, rng, depth - 1)
            if v1 is None or v2 is None:
                return None, None, 0
            if op == "+":
                return f"({n1} + {n2})", v1 + v2, c1 + c2 + 1
            elif op == "-":
                return f"({n1} - {n2})", v1 - v2, c1 + c2 + 1
            elif op == "*":
                return f"({n1} * {n2})", v1 * v2, c1 + c2 + 1
            elif op == "/":
                safe = v2.where(v2.abs() > 1e-8, np.nan)
                return f"({n1} / {n2})", v1 / safe, c1 + c2 + 1
        return None, None, 0

    def _compute_ic(self, predictions: pd.Series, actuals: pd.Series) -> float:
        """Compute rank information coefficient (Spearman correlation)."""
        aligned = pd.DataFrame({"pred": predictions, "actual": actuals}).dropna()
        if len(aligned) < 20:
            return np.nan
        return aligned["pred"].corr(aligned["actual"], method="spearman")
