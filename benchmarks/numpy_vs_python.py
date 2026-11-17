# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Compare a pure-Python problem with the same problem written with numpy.

Run with:  python benchmarks/numpy_vs_python.py [genome_length] [population_size] [generations]

Both versions evolve random strings towards a target of ``genome_length`` characters, with the same
engine settings, and the script reports the average time per generation.
"""
import string
import sys
import time

import numpy as np

from carmelolg.galib import GeneticAlgorithm, Problem

LENGTH = int(sys.argv[1]) if len(sys.argv) > 1 else 500
POPULATION = int(sys.argv[2]) if len(sys.argv) > 2 else 500
GENERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 30

ALPHABET = string.ascii_lowercase + " "
TARGET = "".join(ALPHABET[k % len(ALPHABET)] for k in range(LENGTH))


class PythonPhrase(Problem[str]):
    def random_genome(self, rng):
        return "".join(rng.choices(ALPHABET, k=LENGTH))

    def random_gene(self, rng):
        return rng.choice(ALPHABET)

    def fitness(self, genome):
        return sum(a == b for a, b in zip(genome, TARGET, strict=True))


NP_ALPHABET = np.frombuffer(ALPHABET.encode(), dtype=np.uint8)
NP_TARGET = np.frombuffer(TARGET.encode(), dtype=np.uint8)
CODES = NP_ALPHABET.tolist()


class NumpyPhrase(Problem[np.ndarray]):
    def random_genome(self, rng):
        return np.array(rng.choices(CODES, k=LENGTH), dtype=np.uint8)

    def random_gene(self, rng):
        return rng.choice(CODES)

    def fitness_batch(self, genomes):
        return (np.stack(genomes) == NP_TARGET).sum(axis=1)  # whole population at once


def per_generation(problem):
    engine = GeneticAlgorithm(problem, population_size=POPULATION, max_generations=GENERATIONS, seed=1)
    start = time.perf_counter()
    result = engine.run()
    elapsed = time.perf_counter() - start
    return elapsed / GENERATIONS, result.best.fitness


if __name__ == "__main__":
    print(f"genome length {LENGTH}, population {POPULATION}, {GENERATIONS} generations")
    python_time, python_best = per_generation(PythonPhrase())
    numpy_time, numpy_best = per_generation(NumpyPhrase())
    print(f"pure Python : {python_time * 1000:8.1f} ms/generation   best fitness {python_best:g}")
    print(f"numpy       : {numpy_time * 1000:8.1f} ms/generation   best fitness {numpy_best:g}")
    print(f"speed-up    : {python_time / numpy_time:8.1f}x")
