# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""0/1 knapsack: the same engine on a different genome (a list of bits), using all three customisation levels.

1. parameters        population_size, elite_size, seed
2. operators         a built-in selection and crossover, plus a custom stop condition
3. subclassing       a (mu + lambda) replacement strategy instead of the default generational one
"""
import random

from carmelolg.galib import GeneticAlgorithm, PrintProgress, Problem, Stagnation, StopCondition
from carmelolg.galib.operators import Rank, Uniform

_items_rng = random.Random(7)
ITEMS = [(_items_rng.randint(1, 30), _items_rng.randint(1, 60)) for _ in range(40)]  # (weight, value)
CAPACITY = 250


class Knapsack(Problem[list]):
    def random_genome(self, rng):
        return [rng.randint(0, 1) for _ in ITEMS]

    def random_gene(self, rng):
        return rng.randint(0, 1)

    def fitness(self, genome):
        weight = sum(w for (w, _), take in zip(ITEMS, genome, strict=True) if take)
        value = sum(v for (_, v), take in zip(ITEMS, genome, strict=True) if take)
        return value if weight <= CAPACITY else 0  # overweight bags are worthless


class Converged(StopCondition):
    """Custom stop condition: stop when the population has (almost) no diversity left."""

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def should_stop(self, stats, engine):
        return stats.diversity is not None and stats.diversity < self.threshold


class MuPlusLambda(GeneticAlgorithm):
    """Custom replacement: parents and children compete, the best population_size survive."""

    def replace(self, population, offspring):
        both = sorted([*population, *offspring], key=lambda i: i.score, reverse=True)
        return both[: self.population_size]


if __name__ == "__main__":
    result = MuPlusLambda(
        Knapsack(),
        population_size=150,
        elite_size=0,              # every parent already competes in replace()
        selection=Rank(),
        crossover=Uniform(),
        stop=[Stagnation(80), Converged()],
        callbacks=[PrintProgress(every=20)],
        seed=3,
    ).run()
    print(f"\nbest value: {result.best.fitness} after {result.generations} generations ({result.stopped_by})")
