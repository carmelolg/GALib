# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
import pytest

from carmelolg.galib import GeneticAlgorithm, Individual, Problem

TARGET = "incremental"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class Phrase(Problem[str]):  # module level: worker processes must be able to pickle it
    def random_genome(self, rng):
        return "".join(rng.choice(ALPHABET) for _ in TARGET)

    def random_gene(self, rng):
        return rng.choice(ALPHABET)

    def fitness(self, genome):
        return sum(a == b for a, b in zip(genome, TARGET, strict=True))


class Incremental(Phrase):
    """Counts matches by looking only at the positions that differ from the parent."""

    def fitness_from_parent(self, genome, parent):
        value = parent.fitness
        for k, (new, old) in enumerate(zip(genome, parent.genome, strict=True)):
            if new != old:
                value += (new == TARGET[k]) - (old == TARGET[k])
        return value


class Recording(Phrase):
    def __init__(self):
        self.parent_calls = []
        self.batches = []

    def fitness_batch(self, genomes):
        self.batches.append(len(genomes))
        return [self.fitness(g) for g in genomes]

    def fitness_from_parent(self, genome, parent):
        self.parent_calls.append((genome, parent))
        return self.fitness(genome)


def history(problem, **kwargs):
    ga = GeneticAlgorithm(problem, population_size=20, max_generations=15, seed=4, mutation_rate=0.2, **kwargs)
    return [s.best_genome for s in ga.run().history]


def test_a_problem_without_the_hook_does_not_use_it():
    assert not Phrase().uses_parent_fitness()
    assert Incremental().uses_parent_fitness()
    assert Phrase().fitness_from_parent("a" * len(TARGET), Individual("b" * len(TARGET), 0.0, 0.0)) == Phrase().fitness(
        "a" * len(TARGET))


def test_the_hook_is_called_only_for_mutated_children_with_their_parent():
    problem = Recording()
    ga = GeneticAlgorithm(problem, population_size=20, max_generations=0, seed=1)
    ga.initialize()
    assert problem.parent_calls == []  # the initial population has no parents
    ga.crossover_rate = 0.0
    population = list(ga.population)
    problem.parent_calls.clear()
    children = ga.reproduce(population)
    assert len(problem.parent_calls) == len(children) == ga.population_size - ga.elite_size
    assert all(parent in population for _, parent in problem.parent_calls)
    assert all(child.genome in [g for g, _ in problem.parent_calls] for child in children)


def test_crossover_children_use_fitness_batch_not_the_hook():
    problem = Recording()
    ga = GeneticAlgorithm(problem, population_size=20, max_generations=0, seed=1, crossover_rate=1.0)
    ga.initialize()
    problem.parent_calls.clear()
    problem.batches.clear()
    ga.reproduce(ga.population)
    assert problem.parent_calls == []
    assert problem.batches == [ga.population_size - ga.elite_size]  # one vectorisable batch, not one call per genome


def test_mixed_batches_keep_the_order_of_the_values():
    problem = Incremental()
    ga = GeneticAlgorithm(problem, population_size=30, max_generations=0, seed=2, crossover_rate=0.5)
    ga.initialize()
    for individual in ga.reproduce(ga.population):
        assert individual.fitness == problem.fitness(individual.genome)


def test_incremental_fitness_gives_the_same_run_as_the_plain_one():
    assert history(Incremental()) == history(Phrase())


@pytest.mark.parametrize("parallel", ["thread", "process"])
def test_workers_give_the_same_run_with_the_hook(parallel):
    assert history(Incremental(), workers=2, parallel=parallel) == history(Incremental())


def test_an_engine_overriding_evaluate_with_the_old_signature_still_works_without_the_hook():
    class Legacy(GeneticAlgorithm):
        def evaluate(self, genomes):
            return super().evaluate(genomes)

    ga = Legacy(Phrase(), population_size=10, max_generations=3, seed=1)
    assert ga.run().generations == 3
