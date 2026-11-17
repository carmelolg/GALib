# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Evolve random strings into a target phrase: the original genetic-algorithm-example, rebuilt on GALib.

Level 1 of customisation: only a Problem and a few parameters, everything else is default.
"""
import string
import time

from carmelolg.galib import Callback, GeneticAlgorithm, PrintProgress, Problem, TargetFitness

TARGET = "Happy Birthday! from a genetic algorithm to you :-)"
ALPHABET = string.ascii_letters + " '!?*&.:;,-()"


class PhraseProblem(Problem[str]):
    def random_genome(self, rng):
        return "".join(rng.choice(ALPHABET) for _ in TARGET)

    def random_gene(self, rng):
        return rng.choice(ALPHABET)

    def fitness(self, genome):
        return sum(a == b for a, b in zip(genome, TARGET, strict=True))  # matching characters


class SlowDown(Callback):
    """Demo only: pause after every generation so the progress is visible (the real run takes ~0.3 s)."""

    def __init__(self, seconds=0.03):
        self.seconds = seconds

    def on_generation(self, stats, engine):
        time.sleep(self.seconds)


if __name__ == "__main__":
    result = GeneticAlgorithm(
        PhraseProblem(),
        population_size=300,
        stop=[TargetFitness(len(TARGET))],
        callbacks=[PrintProgress(every=1, in_place=True), SlowDown(0.03)],
        seed=42,
    ).run()
    print(f"best: {result.best.genome!r} after {result.generations} generations ({result.stopped_by})")
