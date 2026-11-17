# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Built-in parent selection operators. All of them prefer a higher ``Individual.score``."""
from __future__ import annotations

from collections.abc import Sequence
from random import Random
from typing import Any, TypeVar

from ..core.individual import Individual

G = TypeVar("G")


def _score(individual: Individual[Any]) -> float:
    return individual.score


class Tournament:
    """Each parent is the best of ``size`` individuals drawn at random.

    Larger ``size`` means stronger selection pressure (faster convergence, less diversity).
    """

    def __init__(self, size: int = 3) -> None:
        if size < 1:
            raise ValueError("tournament size must be >= 1")
        self.size = size

    def __call__(self, population: Sequence[Individual[G]], count: int, rng: Random, /) -> list[Individual[G]]:
        k = min(self.size, len(population))
        return [max(rng.sample(population, k), key=_score) for _ in range(count)]

    def __repr__(self) -> str:
        return f"Tournament(size={self.size})"


class Roulette:
    """Fitness-proportionate selection.

    Scores are shifted so that the worst individual weighs 0 ("windowing"), which also makes
    negative scores work. If every individual has the same score the choice is uniform.
    """

    def __call__(self, population: Sequence[Individual[G]], count: int, rng: Random, /) -> list[Individual[G]]:
        low = min(i.score for i in population)
        weights = [i.score - low for i in population]
        if sum(weights) == 0:
            return rng.choices(population, k=count)
        return rng.choices(population, weights=weights, k=count)

    def __repr__(self) -> str:
        return "Roulette()"


class Rank:
    """Rank-based selection: the worst individual weighs 1, the best weighs N.

    Unlike Roulette, the probability does not depend on how large the fitness gaps are.
    """

    def __call__(self, population: Sequence[Individual[G]], count: int, rng: Random, /) -> list[Individual[G]]:
        ordered = sorted(population, key=_score)
        return rng.choices(ordered, weights=range(1, len(ordered) + 1), k=count)

    def __repr__(self) -> str:
        return "Rank()"
