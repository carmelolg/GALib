# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from random import Random
from typing import Any, Generic

from ..operators.crossover import OnePoint
from ..operators.mutation import random_reset
from .individual import G

_ONE_POINT = OnePoint()


class Problem(ABC, Generic[G]):
    """Describes what is being optimised. Subclass it.

    **Required**

    - `random_genome(rng)`: build one random candidate solution.
    - `fitness(genome)`: how good one candidate is (a finite number), **or**
      `fitness_batch(genomes)`: the fitness of many candidates at once, e.g. vectorised with numpy.

    **Optional overrides**

    - `maximize`: set to `False` for minimisation problems.
    - `random_gene(rng)`: one random gene; enables the default mutation.
    - `mutate(...)`: replace the default mutation entirely.
    - `crossover(...)`: replace the default one-point crossover.

    The defaults for `crossover` and `mutate` work on str, bytes, tuple, list and numpy array genomes.
    For permutations or real values use the operators in `carmelolg.galib.operators`; for any other
    genome type override them. Operators passed to `GeneticAlgorithm` take precedence over these methods.
    """

    maximize: bool = True

    @abstractmethod
    def random_genome(self, rng: Random) -> G:
        """Return a new random genome. Draw all randomness from ``rng`` (reproducibility)."""

    def fitness(self, genome: G) -> float:
        """Return how good ``genome`` is. Must be finite and depend only on ``genome``.

        Implement this or ``fitness_batch``. The default delegates to ``fitness_batch``.
        """
        self.check_fitness_implemented()
        return self.fitness_batch([genome])[0]

    def fitness_batch(self, genomes: Sequence[G]) -> Sequence[float]:
        """Return the fitness of every genome, in order.

        Override it to evaluate the whole population at once, e.g. with numpy. The default calls
        ``fitness`` for each genome. When the engine has an ``executor``, it calls ``fitness`` instead.
        """
        self.check_fitness_implemented()
        return [self.fitness(genome) for genome in genomes]

    def check_fitness_implemented(self) -> None:
        """Raise TypeError if neither ``fitness`` nor ``fitness_batch`` is overridden."""
        cls = type(self)
        if cls.fitness is Problem.fitness and cls.fitness_batch is Problem.fitness_batch:
            raise TypeError(f"{cls.__name__} must implement fitness() or fitness_batch()")

    def random_gene(self, rng: Random) -> Any:
        """Return one random gene. Needed only by the default ``mutate``."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement random_gene() or override mutate() "
            "(or pass mutation=... to GeneticAlgorithm)"
        )

    def crossover(self, a: G, b: G, rng: Random) -> tuple[G, G]:
        """Combine two parents into two children. Default: one-point crossover."""
        return _ONE_POINT(a, b, rng)

    def mutate(self, genome: G, rng: Random, rate: float) -> G:
        """Return a mutated copy of ``genome``. Default: each gene reset with probability ``rate``."""
        return random_reset(genome, rng, rate, self.random_gene)
