# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
import random
from collections import Counter

import pytest

from carmelolg.galib import Individual
from carmelolg.galib.operators import (
    OnePoint,
    RandomReset,
    Rank,
    Roulette,
    Tournament,
    TwoPoint,
    Uniform,
    random_reset,
)


def population(*scores: float) -> list[Individual[str]]:
    return [Individual(genome=f"g{s}", fitness=s, score=s) for s in scores]


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0)


# ---------------------------------------------------------------- crossover

CROSSOVERS = [OnePoint(), TwoPoint(), Uniform()]
PARENTS = [
    ("aaaaaaaa", "bbbbbbbb"),
    (list("aaaaaaaa"), list("bbbbbbbb")),
    (tuple("aaaaaaaa"), tuple("bbbbbbbb")),
]


@pytest.mark.parametrize("crossover", CROSSOVERS, ids=repr)
@pytest.mark.parametrize(("a", "b"), PARENTS, ids=["str", "list", "tuple"])
def test_crossover_children_are_complementary_and_keep_type(crossover, a, b, rng):
    for _ in range(50):
        c1, c2 = crossover(a, b, rng)
        assert type(c1) is type(a) and type(c2) is type(a)
        assert len(c1) == len(c2) == 8
        for g1, g2 in zip(c1, c2, strict=True):
            assert {g1, g2} == {"a", "b"}


@pytest.mark.parametrize("crossover", CROSSOVERS, ids=repr)
def test_crossover_works_on_bytes(crossover, rng):
    c1, c2 = crossover(b"aaaa", b"bbbb", rng)
    assert isinstance(c1, bytes) and set(c1 + c2) <= set(b"ab")


@pytest.mark.parametrize("crossover", CROSSOVERS, ids=repr)
def test_crossover_does_not_modify_parents(crossover, rng):
    a, b = list("aaaa"), list("bbbb")
    crossover(a, b, rng)
    assert a == list("aaaa") and b == list("bbbb")


@pytest.mark.parametrize("crossover", CROSSOVERS, ids=repr)
def test_crossover_rejects_parents_of_different_length(crossover, rng):
    with pytest.raises(ValueError, match="same length"):
        crossover("aaaa", "bbb", rng)


@pytest.mark.parametrize("crossover", CROSSOVERS, ids=repr)
def test_crossover_rejects_unsupported_genome_types(crossover, rng):
    with pytest.raises(TypeError, match="override"):
        crossover(range(6), range(6), rng)


def test_one_point_cut_is_random(rng):
    cuts = {OnePoint()("aaaaaa", "bbbbbb", rng)[0].count("a") for _ in range(300)}
    assert cuts == {1, 2, 3, 4, 5}


def test_short_genomes(rng):
    assert OnePoint()("a", "b", rng) == ("a", "b")
    assert len(TwoPoint()("ab", "cd", rng)[0]) == 2


def test_uniform_probability_extremes(rng):
    assert Uniform(0.0)("aaaa", "bbbb", rng) == ("aaaa", "bbbb")
    assert Uniform(1.0)("aaaa", "bbbb", rng) == ("bbbb", "aaaa")
    with pytest.raises(ValueError):
        Uniform(1.5)


# ----------------------------------------------------------------- mutation

def test_random_reset_rate_zero_and_one(rng):
    assert random_reset("abcdef", rng, 0.0, lambda r: "Z") == "abcdef"
    assert random_reset("abcdef", rng, 1.0, lambda r: "Z") == "ZZZZZZ"
    assert random_reset([1, 2, 3], rng, 1.0, lambda r: 0) == [0, 0, 0]
    assert random_reset((1, 2), rng, 1.0, lambda r: 0) == (0, 0)


def test_random_reset_does_not_modify_input(rng):
    genome = [1, 2, 3]
    child = random_reset(genome, rng, 1.0, lambda r: 0)
    assert genome == [1, 2, 3] and child is not genome


def test_random_reset_mutates_about_rate_times_length(rng):
    changed = sum(
        sum(g == "Z" for g in random_reset("a" * 100, rng, 0.1, lambda r: "Z")) for _ in range(100)
    )
    assert 800 < changed < 1200  # expected 1000


def test_random_reset_operator(rng):
    assert RandomReset(lambda r: 1)([0, 0], rng, 1.0) == [1, 1]


def test_bernoulli_positions_match_independent_trials():
    from carmelolg.galib.operators._genome import bernoulli_positions

    rng = random.Random(1)
    hits = Counter()
    for _ in range(20000):
        positions = list(bernoulli_positions(10, 0.2, rng))
        assert positions == sorted(set(positions)) and all(0 <= p < 10 for p in positions)
        hits.update(positions)
    assert all(abs(hits[p] / 20000 - 0.2) < 0.015 for p in range(10))
    assert list(bernoulli_positions(5, 0.0, rng)) == []
    assert list(bernoulli_positions(5, 1.0, rng)) == [0, 1, 2, 3, 4]
    assert list(bernoulli_positions(0, 0.5, rng)) == []


# ---------------------------------------------------------------- selection

def test_selection_returns_requested_count(rng):
    pop = population(1, 2, 3)
    for select in (Tournament(2), Roulette(), Rank()):
        assert len(select(pop, 7, rng)) == 7


def test_tournament_over_whole_population_returns_best(rng):
    pop = population(1, 5, 3, 9, 2)
    assert {i.score for i in Tournament(size=5)(pop, 10, rng)} == {9}


def test_tournament_of_one_is_uniform(rng):
    counts = Counter(i.score for i in Tournament(size=1)(population(1, 2, 3), 3000, rng))
    assert all(900 < c < 1100 for c in counts.values())


def test_tournament_rejects_invalid_size():
    with pytest.raises(ValueError):
        Tournament(0)


@pytest.mark.parametrize("select", [Roulette(), Rank()], ids=repr)
def test_proportional_selection_prefers_better_and_handles_negative_scores(select, rng):
    counts = Counter(i.score for i in select(population(-10, -5, 0, 5, 10), 3000, rng))
    assert counts[10] > counts[5] > counts[-5]
    assert set(counts) <= {-10, -5, 0, 5, 10}


def test_roulette_with_equal_scores_is_uniform(rng):
    pop = [Individual(genome=name, fitness=4, score=4) for name in "xyz"]
    counts = Counter(i.genome for i in Roulette()(pop, 3000, rng))
    assert len(counts) == 3 and all(900 < c < 1100 for c in counts.values())


def test_rank_ignores_fitness_gaps(rng):
    # same ranks, very different gaps: same distribution
    a = Counter(i.score for i in Rank()(population(1, 2, 3), 3000, random.Random(1)))
    b = Counter(i.score for i in Rank()(population(1, 2, 1000), 3000, random.Random(1)))
    assert [a[1], a[2], a[3]] == [b[1], b[2], b[1000]]
