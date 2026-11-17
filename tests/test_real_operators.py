# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
import random
import statistics

import pytest

from carmelolg.galib import GeneticAlgorithm, Problem, TargetFitness
from carmelolg.galib.metrics import gene_spread
from carmelolg.galib.operators import BlendCrossover, GaussianMutation


@pytest.fixture
def rng():
    return random.Random(0)


def test_blend_alpha_zero_stays_between_parents(rng):
    for _ in range(200):
        c1, c2 = BlendCrossover(alpha=0.0)([0.0, 10.0], [1.0, 20.0], rng)
        for child in (c1, c2):
            assert 0.0 <= child[0] <= 1.0 and 10.0 <= child[1] <= 20.0


def test_blend_explores_outside_parents_and_respects_bounds(rng):
    values = [BlendCrossover(alpha=0.5, low=-0.2, high=1.1)((0.0,), (1.0,), rng)[0][0] for _ in range(2000)]
    assert min(values) < 0.0 and max(values) > 1.0  # explores beyond the parents
    assert min(values) >= -0.2 and max(values) <= 1.1  # but within the bounds


def test_blend_keeps_genome_type_and_validates(rng):
    c1, c2 = BlendCrossover()((1.0, 2.0), (3.0, 4.0), rng)
    assert isinstance(c1, tuple) and isinstance(c2, tuple)
    with pytest.raises(ValueError):
        BlendCrossover(alpha=-1)
    with pytest.raises(ValueError):
        BlendCrossover(low=1, high=0)
    with pytest.raises(ValueError):
        BlendCrossover(low=[0, 0], high=[1, 1, 1])


def test_gaussian_rate_zero_and_one(rng):
    assert GaussianMutation(1.0)([1.0, 2.0], rng, 0.0) == [1.0, 2.0]
    child = GaussianMutation(1.0)([1.0, 2.0], rng, 1.0)
    assert child[0] != 1.0 and child[1] != 2.0


def test_gaussian_noise_has_requested_sigma(rng):
    noise = [GaussianMutation(0.5)([0.0], rng, 1.0)[0] for _ in range(4000)]
    assert abs(statistics.fmean(noise)) < 0.05
    assert statistics.pstdev(noise) == pytest.approx(0.5, rel=0.1)


def test_gaussian_per_gene_sigma_and_bounds(rng):
    mutation = GaussianMutation([0.001, 100.0], low=[-1.0, 0.0], high=[1.0, 1.0])
    for _ in range(200):
        a, b = mutation([0.0, 0.5], rng, 1.0)
        assert -1.0 <= a <= 1.0 and abs(a) < 0.01  # tiny sigma
        assert 0.0 <= b <= 1.0  # huge sigma, clipped


def test_gaussian_validation():
    with pytest.raises(ValueError):
        GaussianMutation(0)
    with pytest.raises(ValueError):
        GaussianMutation([1.0, -1.0])


def test_gene_spread():
    assert gene_spread([[1.0, 5.0], [1.0, 5.0]]) == 0.0
    assert gene_spread([[0.0, 0.0], [2.0, 4.0]]) == pytest.approx((1.0 + 2.0) / 2)
    assert gene_spread(["ab", "cd"]) is None
    assert gene_spread([[1.0], [1.0, 2.0]]) is None


def test_sphere_minimisation_converges():
    class Sphere(Problem[list]):
        maximize = False

        def random_genome(self, rng):
            return [rng.uniform(-5, 5) for _ in range(5)]

        def fitness(self, genome):
            return sum(x * x for x in genome)

    result = GeneticAlgorithm(
        Sphere(), population_size=80, crossover=BlendCrossover(low=-5, high=5),
        mutation=GaussianMutation(0.3, low=-5, high=5), diversity=gene_spread,
        stop=[TargetFitness(1e-3)], max_generations=1000, seed=1,
    ).run()
    assert result.best.fitness <= 1e-3
    assert result.history[0].diversity > result.history[-1].diversity
