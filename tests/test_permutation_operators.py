# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
import random

import pytest

from carmelolg.galib import GeneticAlgorithm, Problem, TargetFitness
from carmelolg.galib.operators import InversionMutation, OrderCrossover, PartiallyMappedCrossover, SwapMutation

PERMUTATION_CROSSOVERS = [OrderCrossover(), PartiallyMappedCrossover()]


@pytest.fixture
def rng():
    return random.Random(0)


@pytest.mark.parametrize("crossover", PERMUTATION_CROSSOVERS, ids=repr)
@pytest.mark.parametrize("kind", [list, tuple])
def test_children_are_valid_permutations(crossover, kind, rng):
    for _ in range(300):
        a = list(range(10))
        b = list(range(10))
        rng.shuffle(a)
        rng.shuffle(b)
        c1, c2 = crossover(kind(a), kind(b), rng)
        assert type(c1) is kind and type(c2) is kind
        assert sorted(c1) == sorted(c2) == list(range(10))


@pytest.mark.parametrize("crossover", PERMUTATION_CROSSOVERS, ids=repr)
def test_string_permutations(crossover, rng):
    c1, c2 = crossover("abcdefg", "gfedcba", rng)
    assert sorted(c1) == sorted(c2) == list("abcdefg")


@pytest.mark.parametrize("crossover", PERMUTATION_CROSSOVERS, ids=repr)
def test_parents_are_not_modified(crossover, rng):
    a, b = [0, 1, 2, 3, 4], [4, 3, 2, 1, 0]
    crossover(a, b, rng)
    assert a == [0, 1, 2, 3, 4] and b == [4, 3, 2, 1, 0]


@pytest.mark.parametrize("crossover", PERMUTATION_CROSSOVERS, ids=repr)
def test_rejects_non_permutations(crossover, rng):
    with pytest.raises(ValueError, match="permutations"):
        crossover([0, 0, 1], [0, 1, 2], rng)
    with pytest.raises(ValueError, match="permutations"):
        crossover([0, 1, 2], [0, 1, 3], rng)


@pytest.mark.parametrize("crossover", PERMUTATION_CROSSOVERS, ids=repr)
def test_identical_parents_give_identical_children(crossover, rng):
    assert crossover([2, 0, 1], [2, 0, 1], rng) == ([2, 0, 1], [2, 0, 1])


def test_order_crossover_known_result():
    # segment [3, 7) of a (D E F G) is kept; b read from position 7, wrapping, without D E F G gives
    # the donors A I H C B, placed at positions 7, 8, 0, 1, 2
    child = OrderCrossover._child(list("ABCDEFGHI"), list("HGFEDCBAI"), 3, 7)
    assert child == list("HCBDEFGAI")


def test_pmx_known_result():
    # classic example: segment [3, 7) of p1 = 4 5 6 7
    p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    p2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]
    child = PartiallyMappedCrossover._child(p1, p2, 3, 7)
    assert child == [9, 3, 2, 4, 5, 6, 7, 1, 8]


def test_pmx_keeps_positions_of_the_other_parent_when_possible(rng):
    p1, p2 = list(range(8)), [7, 6, 5, 4, 3, 2, 1, 0]
    child = PartiallyMappedCrossover._child(p1, p2, 2, 5)
    assert child[2:5] == [2, 3, 4]
    assert sorted(child) == list(range(8))


@pytest.mark.parametrize("mutation", [SwapMutation(), InversionMutation()], ids=repr)
def test_mutations_keep_permutations(mutation, rng):
    for _ in range(300):
        genome = list(range(12))
        rng.shuffle(genome)
        child = mutation(genome, rng, 0.3)
        assert sorted(child) == list(range(12))


@pytest.mark.parametrize("mutation", [SwapMutation(), InversionMutation()], ids=repr)
def test_mutation_rate_zero_changes_nothing(mutation, rng):
    assert mutation((3, 1, 2, 0), rng, 0.0) == (3, 1, 2, 0)


def test_inversion_reverses_a_segment(rng):
    child = InversionMutation()(list(range(10)), rng, 1.0)
    changed = [i for i, gene in enumerate(child) if gene != i]
    if changed:  # the segment between the first and last changed position is reversed
        start, end = changed[0], changed[-1] + 1
        assert child[start:end] == list(range(start, end))[::-1]


def test_tsp_converges_on_a_circle():
    import math

    cities = [(math.cos(2 * math.pi * k / 12), math.sin(2 * math.pi * k / 12)) for k in range(12)]
    optimum = 12 * math.dist(cities[0], cities[1])

    class Tour(Problem[list]):
        maximize = False

        def random_genome(self, rng):
            tour = list(range(len(cities)))
            rng.shuffle(tour)
            return tour

        def fitness(self, tour):
            return sum(math.dist(cities[tour[k]], cities[tour[k - 1]]) for k in range(len(tour)))

    result = GeneticAlgorithm(
        Tour(), population_size=150, crossover=OrderCrossover(), mutation=InversionMutation(),
        stop=[TargetFitness(optimum + 1e-9)], max_generations=500, seed=1,
    ).run()
    assert result.best.fitness == pytest.approx(optimum)
