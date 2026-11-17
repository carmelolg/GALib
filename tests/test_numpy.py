# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
import random

import pytest

from carmelolg.galib import GeneticAlgorithm, Problem, TargetFitness
from carmelolg.galib.metrics import gene_spread, positional_diversity
from carmelolg.galib.operators import (
    BlendCrossover,
    GaussianMutation,
    InversionMutation,
    OnePoint,
    OrderCrossover,
    PartiallyMappedCrossover,
    SwapMutation,
    TwoPoint,
    Uniform,
    random_reset,
)

np = pytest.importorskip("numpy")


@pytest.fixture
def rng():
    return random.Random(0)


@pytest.mark.parametrize("crossover", [OnePoint(), TwoPoint(), Uniform()], ids=repr)
def test_sequence_crossovers_on_arrays(crossover, rng):
    a, b = np.zeros(8, dtype=np.int8), np.ones(8, dtype=np.int8)
    for _ in range(50):
        c1, c2 = crossover(a, b, rng)
        assert isinstance(c1, np.ndarray) and c1.dtype == np.int8 and c1.shape == (8,)
        assert np.all(c1 + c2 == 1)  # complementary
    assert np.all(a == 0) and np.all(b == 1)  # parents untouched


@pytest.mark.parametrize("crossover", [OrderCrossover(), PartiallyMappedCrossover()], ids=repr)
def test_permutation_crossovers_on_arrays(crossover, rng):
    a, b = np.arange(10), np.arange(10)[::-1].copy()
    c1, c2 = crossover(a, b, rng)
    assert isinstance(c1, np.ndarray) and sorted(c1.tolist()) == list(range(10))
    assert sorted(c2.tolist()) == list(range(10))


@pytest.mark.parametrize("mutation", [SwapMutation(), InversionMutation()], ids=repr)
def test_permutation_mutations_on_arrays(mutation, rng):
    child = mutation(np.arange(10), rng, 0.5)
    assert isinstance(child, np.ndarray) and sorted(child.tolist()) == list(range(10))


def test_random_reset_on_arrays(rng):
    genome = np.zeros(1000, dtype=np.int64)
    child = random_reset(genome, rng, 0.1, lambda r: 7)
    assert child.dtype == np.int64 and np.all(genome == 0)
    assert 50 < int((child == 7).sum()) < 150
    assert np.array_equal(random_reset(genome, rng, 0.0, lambda r: 7), genome)


def test_real_operators_on_arrays(rng):
    a, b = np.zeros(5), np.ones(5)
    c1, c2 = BlendCrossover(alpha=0.0)(a, b, rng)
    assert np.all((c1 >= 0) & (c1 <= 1)) and np.all((c2 >= 0) & (c2 <= 1))
    mutated = GaussianMutation(10.0, low=-1.0, high=1.0)(a, rng, 1.0)
    assert np.all(mutated >= -1.0) and np.all(mutated <= 1.0) and not np.array_equal(mutated, a)
    assert np.array_equal(GaussianMutation(1.0)(a, rng, 0.0), a)


def test_array_operators_are_reproducible():
    def run(seed):
        rng = random.Random(seed)
        return Uniform()(np.zeros(20), np.ones(20), rng)[0]

    assert np.array_equal(run(3), run(3))
    assert not np.array_equal(run(3), run(4))


def test_metrics_on_arrays():
    same = [np.array([1, 2, 3])] * 4
    assert positional_diversity(same) == 0.0
    assert positional_diversity([np.array([0, 0]), np.array([1, 1])]) == pytest.approx(0.5)
    assert positional_diversity([np.array([0, 0]), np.array([1, 1, 1])]) is None
    assert gene_spread([np.array([0.0, 0.0]), np.array([2.0, 4.0])]) == pytest.approx(1.5)
    # array metrics agree with list metrics
    genomes = [[random.Random(k).randint(0, 3) for _ in range(6)] for k in range(30)]
    assert positional_diversity([np.array(g) for g in genomes]) == pytest.approx(positional_diversity(genomes))


def test_vectorised_most_common_counts_match_counter():
    from collections import Counter

    from carmelolg.galib.metrics import _most_common_counts

    generator = np.random.default_rng(0)
    for _ in range(200):
        rows, columns, values = generator.integers(1, 30), generator.integers(1, 12), generator.integers(1, 6)
        matrix = generator.integers(0, values, size=(rows, columns))
        expected = [Counter(column.tolist()).most_common(1)[0][1] for column in matrix.T]
        assert _most_common_counts(matrix).tolist() == expected


def test_gaussian_mutation_of_integer_array_returns_floats(rng):
    child = GaussianMutation(0.5)(np.zeros(4, dtype=np.int64), rng, 1.0)
    assert child.dtype.kind == "f" and np.any(child != 0)


def test_numpy_generator_is_reused_per_rng():
    from carmelolg.galib.operators._genome import numpy_rng

    first, second = random.Random(1), random.Random(1)
    assert numpy_rng(first) is numpy_rng(first)
    assert numpy_rng(first) is not numpy_rng(second)
    assert numpy_rng(random.Random(1)).random() == numpy_rng(random.Random(1)).random()


TARGET = np.frombuffer(b"numpy makes it fast", dtype=np.uint8)
ALPHABET = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz ", dtype=np.uint8)


class VectorisedPhrase(Problem[np.ndarray]):
    """Fitness of the whole population in one numpy expression."""

    def random_genome(self, rng):
        return np.array([rng.choice(ALPHABET.tolist()) for _ in TARGET], dtype=np.uint8)

    def random_gene(self, rng):
        return rng.choice(ALPHABET.tolist())

    def fitness_batch(self, genomes):
        return (np.stack(genomes) == TARGET).sum(axis=1)


def test_vectorised_problem_converges_and_is_reproducible():
    def run():
        return GeneticAlgorithm(VectorisedPhrase(), population_size=200, seed=5,
                                stop=[TargetFitness(len(TARGET))], max_generations=500).run()

    first, second = run(), run()
    assert bytes(first.best.genome) == b"numpy makes it fast"
    assert first.generations == second.generations
    assert isinstance(first.best.fitness, float)
