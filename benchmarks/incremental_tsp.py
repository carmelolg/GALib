# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Time per generation of a travelling salesman run, with and without ``fitness_from_parent``.

Run with:  python benchmarks/incremental_tsp.py [cities] [population] [generations]

Both runs use the same seed, operators and settings, so they must give the same tours; the script checks it and
reports the average time per generation. The gain here is moderate: the child is still compared with its parent to
find the reversed segment, and crossover children are evaluated in full. It grows with the cost of the fitness that
is skipped. ``IncrementalTour`` is the one of ``examples/tsp_delta.py``: keep the two in step.
"""
import math
import random
import sys
import time

from carmelolg.galib import GeneticAlgorithm, Problem
from carmelolg.galib.operators import InversionMutation, OrderCrossover

CITIES_N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
POPULATION = int(sys.argv[2]) if len(sys.argv) > 2 else 100
GENERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 100

_rng = random.Random(11)
CITIES = [(_rng.uniform(0, 100), _rng.uniform(0, 100)) for _ in range(CITIES_N)]


def edge(a, b):
    return math.dist(CITIES[a], CITIES[b])


class Tour(Problem[list]):
    maximize = False

    def random_genome(self, rng):
        tour = list(range(CITIES_N))
        rng.shuffle(tour)
        return tour

    def fitness(self, tour):
        return sum(edge(tour[k - 1], tour[k]) for k in range(len(tour)))


class IncrementalTour(Tour):
    def fitness_from_parent(self, tour, parent):
        old = parent.genome
        changed = [k for k in range(len(tour)) if tour[k] != old[k]]
        if not changed:
            return parent.fitness
        i, j = changed[0], changed[-1]
        if j - i + 1 == len(tour) or tour[i : j + 1] != old[i : j + 1][::-1]:
            return self.fitness(tour)
        before, after = old[i - 1], old[(j + 1) % len(tour)]
        first, last = old[i], old[j]
        return parent.fitness - edge(before, first) - edge(last, after) + edge(before, last) + edge(first, after)


def run(problem):
    engine = GeneticAlgorithm(problem, population_size=POPULATION, max_generations=GENERATIONS, crossover_rate=0.2,
                              crossover=OrderCrossover(), mutation=InversionMutation(), mutation_rate=1.0,
                              seed=1, diversity=None)
    start = time.perf_counter()
    result = engine.run()
    return result, (time.perf_counter() - start) / GENERATIONS


if __name__ == "__main__":
    plain, t_plain = run(Tour())
    incremental, t_incremental = run(IncrementalTour())
    assert math.isclose(plain.best.fitness, incremental.best.fitness, rel_tol=1e-9), "the runs diverged"
    print(f"{CITIES_N} cities, population {POPULATION}: best {plain.best.fitness:.1f} in both runs")
    print(f"fitness only:        {t_plain * 1e3:8.2f} ms/generation")
    print(f"fitness_from_parent: {t_incremental * 1e3:8.2f} ms/generation  ({t_plain / t_incremental:.1f}x)")
