# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Travelling salesman with an incremental fitness: ``fitness_from_parent``.

Same problem as ``tsp.py``, but with many cities. InversionMutation reverses one segment of the tour, which changes
only two edges, so the length of a child follows from its parent's with four distances instead of one per city
(finding the reversed segment is still a cheap scan of the two tours). The engine calls ``fitness_from_parent`` for
children that were only mutated; children of OrderCrossover (and the initial population) still go through
``fitness``. Both give the same value, so the run is identical either way.
"""
import math
import random

from carmelolg.galib import GeneticAlgorithm, PrintProgress, Problem, Stagnation
from carmelolg.galib.operators import InversionMutation, OrderCrossover

_cities_rng = random.Random(11)
CITIES = [(_cities_rng.uniform(0, 100), _cities_rng.uniform(0, 100)) for _ in range(500)]


def edge(a, b):
    return math.dist(CITIES[a], CITIES[b])


class Tour(Problem[list]):
    maximize = False  # shorter is better

    def random_genome(self, rng):
        tour = list(range(len(CITIES)))
        rng.shuffle(tour)
        return tour

    def fitness(self, tour):
        # closed tour: from the last city back to the first one
        return sum(edge(tour[k - 1], tour[k]) for k in range(len(tour)))


class IncrementalTour(Tour):
    def fitness_from_parent(self, tour, parent):
        old = parent.genome
        changed = [k for k in range(len(tour)) if tour[k] != old[k]]
        if not changed:
            return parent.fitness
        i, j = changed[0], changed[-1]
        n = len(tour)
        if j - i + 1 == n or tour[i : j + 1] != old[i : j + 1][::-1]:  # the whole tour reversed, or not one inversion
            return self.fitness(tour)
        before, after = old[i - 1], old[(j + 1) % n]  # the cities next to the reversed segment
        first, last = old[i], old[j]
        removed = edge(before, first) + edge(last, after)
        added = edge(before, last) + edge(first, after)
        return parent.fitness - removed + added


if __name__ == "__main__":
    result = GeneticAlgorithm(
        IncrementalTour(),
        population_size=100,
        crossover_rate=0.2,
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        mutation_rate=1.0,
        stop=[Stagnation(300)],
        max_generations=3000,
        callbacks=[PrintProgress(every=100, width=0)],
        seed=1,
    ).run()
    print(f"\nshortest tour: {result.best.fitness:.1f} after {result.generations} generations ({result.stopped_by})")
