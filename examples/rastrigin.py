# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Minimise the Rastrigin function: real-valued numpy genomes, vectorised fitness, real-valued operators.

Rastrigin has a single global minimum, 0 at the origin, surrounded by many local minima: a classic test for
how well an algorithm keeps exploring. In 10 dimensions a plain genetic algorithm usually ends close to the
origin (best values between about 0.1 and 2 depending on the seed) rather than exactly on it; try other
settings and seeds to see how exploration changes the result.

Requires numpy:  pip install "io.github.carmelolg.galib[numpy]"
"""
import numpy as np

from carmelolg.galib import GeneticAlgorithm, PrintProgress, Problem, Stagnation, TargetFitness
from carmelolg.galib.metrics import gene_spread
from carmelolg.galib.operators import BlendCrossover, GaussianMutation, Tournament

DIMENSIONS = 10
LOW, HIGH = -5.12, 5.12


class Rastrigin(Problem[np.ndarray]):
    maximize = False

    def random_genome(self, rng):
        return np.array([rng.uniform(LOW, HIGH) for _ in range(DIMENSIONS)])

    def fitness_batch(self, genomes):
        x = np.stack(genomes)  # (population, DIMENSIONS): the whole population in one expression
        return 10 * DIMENSIONS + (x**2 - 10 * np.cos(2 * np.pi * x)).sum(axis=1)


if __name__ == "__main__":
    result = GeneticAlgorithm(
        Rastrigin(),
        population_size=300,
        selection=Tournament(2),  # mild selection pressure: keeps exploring longer
        crossover=BlendCrossover(alpha=0.3, low=LOW, high=HIGH),
        mutation=GaussianMutation(sigma=0.5, low=LOW, high=HIGH),
        mutation_rate=0.1,
        diversity=gene_spread,  # standard deviation of the genes: equality-based diversity is meaningless here
        stop=[TargetFitness(1e-3), Stagnation(200)],
        max_generations=5000,
        callbacks=[PrintProgress(every=100, width=0)],
        seed=3,
    ).run()
    print(f"\nbest value: {result.best.fitness:.4f} after {result.generations} generations ({result.stopped_by})")
    print("best point:", np.round(result.best.genome, 3))
