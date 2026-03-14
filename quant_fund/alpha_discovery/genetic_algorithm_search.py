"""Genetic algorithm search over signal parameter space.

Evolves signal parameters (lookback windows, combination weights,
normalisation methods) toward higher out-of-sample IC, with complexity
penalty to avoid overfitting.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """An individual in the GA population representing a signal configuration."""

    params: Dict[str, float]
    fitness: float = 0.0
    complexity: int = 0


@dataclass
class GAResult:
    """Result of a GA search run."""

    best_params: Dict[str, float]
    best_fitness: float
    generations_run: int
    population_history: List[float] = field(default_factory=list)


class GeneticAlgorithmSearch:
    """GA-based search over signal parameter space.

    The fitness function is out-of-sample IC. Complexity is penalised
    to avoid overfitting.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._population_size = cfg.get("ga_population_size", 50)
        self._generations = cfg.get("ga_generations", 30)
        self._mutation_rate = cfg.get("ga_mutation_rate", 0.1)
        self._crossover_rate = cfg.get("ga_crossover_rate", 0.7)
        self._elite_fraction = cfg.get("ga_elite_fraction", 0.1)
        self._complexity_penalty = cfg.get("ga_complexity_penalty", 0.005)
        self._seed = cfg.get("random_seed", 42)

    def search(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        fitness_fn: Callable[[Dict[str, float]], float],
    ) -> GAResult:
        """Run GA optimisation over the parameter space.

        Args:
            param_ranges: Dict mapping parameter names to (min, max) bounds.
            fitness_fn: Function that takes a parameter dict and returns
                out-of-sample IC (higher is better).

        Returns:
            GAResult with the best parameters found.
        """
        rng = np.random.default_rng(self._seed)
        population = self._initialise_population(param_ranges, rng)
        best_history: List[float] = []

        for gen in range(self._generations):
            for ind in population:
                try:
                    raw_fitness = fitness_fn(ind.params)
                    ind.complexity = self._estimate_complexity(ind.params)
                    ind.fitness = raw_fitness - self._complexity_penalty * ind.complexity
                except Exception as e:
                    logger.debug("Fitness evaluation failed: %s", e)
                    ind.fitness = -999.0

            population.sort(key=lambda x: x.fitness, reverse=True)
            best_history.append(population[0].fitness)
            logger.debug(
                "Generation %d: best fitness=%.4f", gen, population[0].fitness
            )

            n_elite = max(1, int(self._elite_fraction * self._population_size))
            next_gen = population[:n_elite]

            while len(next_gen) < self._population_size:
                p1, p2 = self._tournament_select(population, rng, k=3)
                child = self._crossover(p1, p2, param_ranges, rng)
                child = self._mutate(child, param_ranges, rng)
                next_gen.append(child)

            population = next_gen

        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        return GAResult(
            best_params=best.params,
            best_fitness=best.fitness,
            generations_run=self._generations,
            population_history=best_history,
        )

    def _initialise_population(
        self, param_ranges: Dict[str, Tuple[float, float]], rng
    ) -> List[Individual]:
        """Create random initial population within parameter bounds."""
        population = []
        for _ in range(self._population_size):
            params = {}
            for name, (lo, hi) in param_ranges.items():
                params[name] = rng.uniform(lo, hi)
            population.append(Individual(params=params))
        return population

    def _tournament_select(
        self, population: List[Individual], rng, k: int = 3
    ) -> Tuple[Individual, Individual]:
        """Select two parents via tournament selection."""
        def _pick():
            candidates = rng.choice(len(population), size=min(k, len(population)), replace=False)
            best_idx = max(candidates, key=lambda i: population[i].fitness)
            return population[best_idx]
        return _pick(), _pick()

    def _crossover(
        self,
        p1: Individual,
        p2: Individual,
        param_ranges: Dict[str, Tuple[float, float]],
        rng,
    ) -> Individual:
        """Uniform crossover between two parents."""
        if rng.random() > self._crossover_rate:
            return Individual(params=dict(p1.params))

        child_params = {}
        for name in param_ranges:
            if rng.random() < 0.5:
                child_params[name] = p1.params[name]
            else:
                child_params[name] = p2.params[name]
        return Individual(params=child_params)

    def _mutate(
        self,
        ind: Individual,
        param_ranges: Dict[str, Tuple[float, float]],
        rng,
    ) -> Individual:
        """Gaussian mutation on parameters."""
        params = dict(ind.params)
        for name, (lo, hi) in param_ranges.items():
            if rng.random() < self._mutation_rate:
                spread = (hi - lo) * 0.1
                params[name] = np.clip(
                    params[name] + rng.normal(0, spread), lo, hi
                )
        return Individual(params=params)

    def _estimate_complexity(self, params: Dict[str, float]) -> int:
        """Estimate complexity as the number of non-zero parameters."""
        return sum(1 for v in params.values() if abs(v) > 1e-6)
