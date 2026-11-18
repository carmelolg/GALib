# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor

import pytest

from carmelolg.galib import GeneticAlgorithm, Problem

TARGET = "parallel"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class Phrase(Problem[str]):  # module level: worker processes must be able to pickle it
    batch_sizes: list[int]

    def random_genome(self, rng):
        return "".join(rng.choice(ALPHABET) for _ in TARGET)

    def random_gene(self, rng):
        return rng.choice(ALPHABET)

    def fitness(self, genome):
        return sum(a == b for a, b in zip(genome, TARGET, strict=True))


class Recording(Phrase):
    def __init__(self):
        self.batch_sizes = []

    def fitness_batch(self, genomes):
        self.batch_sizes.append(len(genomes))
        return [self.fitness(g) for g in genomes]


def best(result):
    return [s.best_genome for s in result.history]


def run(**kwargs):
    return GeneticAlgorithm(Phrase(), population_size=20, max_generations=8, seed=3, **kwargs).run()


@pytest.mark.parametrize("parallel", ["thread", "process"])
def test_workers_give_same_result_as_serial(parallel):
    assert best(run(workers=2, parallel=parallel)) == best(run())


def test_chunks_are_split_and_bounded_by_chunk_size():
    problem = Recording()
    with ThreadPoolExecutor(max_workers=2) as pool:
        # threads share the problem object, so the recorded batch sizes are visible
        GeneticAlgorithm(problem, population_size=10, elite_size=0, max_generations=1, seed=1,
                         executor=pool, chunk_size=3).run()
    assert sorted(problem.batch_sizes[:4]) == [1, 3, 3, 3]  # generation 0: 10 genomes in chunks of 3


def test_default_chunks_one_per_worker():
    problem = Recording()
    GeneticAlgorithm(problem, population_size=10, max_generations=0, seed=1, workers=2, parallel="thread").run()
    assert sorted(problem.batch_sizes) == [5, 5]


def test_engine_pool_is_closed_after_run_and_context_manager():
    engine = GeneticAlgorithm(Phrase(), population_size=10, max_generations=2, seed=1, workers=2, parallel="thread")
    engine.run()
    assert engine._pool is None
    with GeneticAlgorithm(Phrase(), population_size=10, seed=1, workers=2, parallel="thread") as manual:
        manual.initialize()
        manual.step()
        assert manual._pool is not None
    assert manual._pool is None
    manual.close()  # twice is fine


def test_user_executor_is_not_closed():
    with ThreadPoolExecutor(max_workers=2) as pool:
        run_with = GeneticAlgorithm(Phrase(), population_size=10, max_generations=1, seed=1, executor=pool)
        run_with.run()
        assert pool.submit(int, "7").result() == 7  # still usable


@pytest.mark.parametrize(
    "kwargs",
    [{"workers": 0}, {"chunk_size": 0}, {"parallel": "gpu"}, {"workers": 2, "executor": ThreadPoolExecutor(1)}],
)
def test_invalid_parallel_arguments(kwargs):
    with pytest.raises(ValueError):
        GeneticAlgorithm(Phrase(), **kwargs)
