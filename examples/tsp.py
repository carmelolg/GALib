# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Travelling salesman: a permutation genome, minimisation, permutation operators.

The genome is the order in which the cities are visited. Generic crossovers would duplicate cities, so the
example uses OrderCrossover and InversionMutation, which always produce valid tours.
"""
import math
import random

from carmelolg.galib import GeneticAlgorithm, PrintProgress, Problem, Stagnation
from carmelolg.galib.operators import InversionMutation, OrderCrossover

_cities_rng = random.Random(11)
CITIES = [(_cities_rng.uniform(0, 100), _cities_rng.uniform(0, 100)) for _ in range(30)]


class Tour(Problem[list]):
    maximize = False  # shorter is better

    def random_genome(self, rng):
        tour = list(range(len(CITIES)))
        rng.shuffle(tour)
        return tour

    def fitness(self, tour):
        # closed tour: from the last city back to the first one
        return sum(math.dist(CITIES[tour[k - 1]], CITIES[tour[k]]) for k in range(len(tour)))


if __name__ == "__main__":
    result = GeneticAlgorithm(
        Tour(),
        population_size=200,
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        stop=[Stagnation(150)],
        max_generations=3000,
        callbacks=[PrintProgress(every=50, width=0)],
        seed=1,
    ).run()
    print(f"\nshortest tour: {result.best.fitness:.1f} after {result.generations} generations ({result.stopped_by})")
    print("order:", result.best.genome)
