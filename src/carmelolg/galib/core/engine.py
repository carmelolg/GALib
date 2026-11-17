# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence, Sized
from concurrent.futures import Executor
from random import Random
from typing import TYPE_CHECKING, Any, Generic

from ..metrics import DiversityMeasure, positional_diversity
from ..operators.protocols import CrossoverOperator, MutationOperator, SelectionOperator
from ..operators.selection import Tournament
from .individual import G, Individual
from .problem import Problem
from .stats import GenerationStats, Result

if TYPE_CHECKING:
    from ..callbacks import Callback
    from ..termination import StopCondition


def _score(individual: Individual[Any]) -> float:
    return individual.score


class GeneticAlgorithm(Generic[G]):
    """The evolutionary loop.

    Three ways to customise it, from light to deep:

    1. Parameters: ``population_size``, ``elite_size``, rates, ``stop``, ``callbacks``, ``seed``, ``executor``.
    2. Operators: pass ``selection=``, ``crossover=``, ``mutation=`` (see ``carmelolg.galib.operators``), or
       override ``crossover`` / ``mutate`` on your ``Problem``. Explicit arguments win over the Problem.
    3. Subclassing: ``step`` is built from ``reproduce`` and ``replace``, which use ``select``,
       ``crossover``, ``mutate`` and ``evaluate``. Override any of them, plus ``initial_population``
       and ``measure_diversity``. The engine never relies on the population being sorted.

    Default strategy: tournament selection, crossover with probability ``crossover_rate``,
    per-gene mutation, generational replacement keeping the ``elite_size`` best parents.
    """

    def __init__(
        self,
        problem: Problem[G],
        *,
        population_size: int = 100,
        max_generations: int = 1000,
        elite_size: int = 2,
        crossover_rate: float = 0.9,
        mutation_rate: float | None = None,
        selection: SelectionOperator[G] | None = None,
        crossover: CrossoverOperator[G] | None = None,
        mutation: MutationOperator[G] | None = None,
        stop: Iterable[StopCondition] = (),
        callbacks: Iterable[Callback] = (),
        seed: int | None = None,
        executor: Executor | None = None,
        diversity: DiversityMeasure | None = positional_diversity,
    ) -> None:
        """
        :param mutation_rate: per-gene mutation probability; ``None`` means ``1 / genome length``.
        :param stop: extra stop conditions; the run also stops after ``max_generations``.
        :param seed: seed of the run's own ``random.Random``; same seed, same run.
        :param executor: evaluate fitness with ``executor.map(problem.fitness, ...)`` (with a process
            pool the Problem must be picklable). Without it the engine calls ``problem.fitness_batch``.
        :param diversity: how to measure diversity (see ``carmelolg.galib.metrics``); ``None`` disables
            the measure, which saves time on large populations.
        """
        problem.check_fitness_implemented()
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if max_generations < 0:
            raise ValueError("max_generations must be >= 0")
        if not 0 <= elite_size < population_size:
            raise ValueError("elite_size must be in [0, population_size)")
        if not 0 <= crossover_rate <= 1:
            raise ValueError("crossover_rate must be in [0, 1]")
        if mutation_rate is not None and not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1]")

        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.stop: list[StopCondition] = list(stop)
        self.callbacks: list[Callback] = list(callbacks)
        self.seed = seed
        self.executor = executor
        self.diversity = diversity

        self._selection: SelectionOperator[G] = selection if selection is not None else Tournament(3)
        self._crossover: CrossoverOperator[G] = crossover if crossover is not None else problem.crossover
        self._mutation: MutationOperator[G] = mutation if mutation is not None else problem.mutate

        # Run state, (re)set by initialize().
        self.rng = Random(seed)
        self.population: list[Individual[G]] = []
        self.generation = 0
        self.evaluations = 0
        self.history: list[GenerationStats[G]] = []
        self._best: Individual[G] | None = None
        self._effective_mutation_rate: float | None = None

    # ---------------------------------------------------------------- state

    @property
    def best(self) -> Individual[G]:
        """Best individual seen so far in the current run."""
        if self._best is None:
            raise RuntimeError("the algorithm is not initialized: call run() or initialize() first")
        return self._best

    @property
    def effective_mutation_rate(self) -> float:
        """The per-gene mutation rate actually used (``mutation_rate`` or ``1 / genome length``)."""
        if self._effective_mutation_rate is None:
            raise RuntimeError("the algorithm is not initialized: call run() or initialize() first")
        return self._effective_mutation_rate

    # ------------------------------------------------------------- the run

    def run(self) -> Result[G]:
        """Evolve until a stop condition fires or ``max_generations`` generations have been evolved.

        Calling ``run`` again restarts from scratch (with the same seed, the same run).
        """
        self.initialize()
        for callback in self.callbacks:
            callback.on_start(self)

        stopped_by = self._record_generation()
        while stopped_by is None and self.generation < self.max_generations:
            self.step()
            stopped_by = self._record_generation()

        result = Result(
            best=self.best,
            generations=self.generation,
            stopped_by=stopped_by or "max_generations",
            evaluations=self.evaluations,
            history=list(self.history),
            population=list(self.population),
        )
        for callback in self.callbacks:
            callback.on_end(result, self)
        return result

    def initialize(self) -> None:
        """Reset the run state and build the initial population (generation 0).

        ``run`` calls it; call it yourself only to drive the loop manually with ``step``.
        """
        self.rng = Random(self.seed)
        self.generation = 0
        self.evaluations = 0
        self.history = []
        for condition in self.stop:
            condition.reset()
        population = self.initial_population()
        if not population:
            raise RuntimeError("initial_population() returned an empty population")
        self.population = population
        self._best = max(population, key=_score)
        self._effective_mutation_rate = self._resolve_mutation_rate()

    def step(self) -> None:
        """Evolve one generation. Override the building blocks below rather than this method."""
        best = self.best  # raises if not initialized
        offspring = self.reproduce(self.population)
        population = self.replace(self.population, offspring)
        if not population:
            raise RuntimeError("replace() returned an empty population")
        self.population = population
        self.generation += 1
        candidate = max(population, key=_score)
        if candidate.score > best.score:
            self._best = candidate

    # ------------------------------------------------- overridable building blocks

    def initial_population(self) -> list[Individual[G]]:
        """Build and evaluate ``population_size`` random individuals."""
        return self.evaluate([self.problem.random_genome(self.rng) for _ in range(self.population_size)])

    def evaluate(self, genomes: Sequence[G]) -> list[Individual[G]]:
        """Compute the fitness of new genomes: ``problem.fitness_batch``, or ``problem.fitness`` through
        the ``executor`` when one was given."""
        if not genomes:
            return []
        if self.executor is None:
            values = list(self.problem.fitness_batch(genomes))
        else:
            values = list(self.executor.map(self.problem.fitness, genomes))
        if len(values) != len(genomes):
            raise ValueError(f"fitness_batch returned {len(values)} values for {len(genomes)} genomes")
        self.evaluations += len(genomes)
        sign = 1.0 if self.problem.maximize else -1.0
        individuals = []
        for genome, value in zip(genomes, values, strict=True):
            fitness = float(value)
            if not math.isfinite(fitness):
                raise ValueError(f"fitness must be a finite number, got {value!r} for genome {genome!r}")
            individuals.append(Individual(genome, fitness, sign * fitness))
        return individuals

    def select(self, population: Sequence[Individual[G]], count: int) -> list[Individual[G]]:
        """Pick ``count`` parents."""
        parents = self._selection(population, count, self.rng)
        if len(parents) != count:
            raise ValueError(f"selection operator returned {len(parents)} parents, expected {count}")
        return parents

    def crossover(self, a: G, b: G) -> tuple[G, G]:
        return self._crossover(a, b, self.rng)

    def mutate(self, genome: G) -> G:
        return self._mutation(genome, self.rng, self.effective_mutation_rate)

    def reproduce(self, population: Sequence[Individual[G]]) -> list[Individual[G]]:
        """Build ``population_size - elite_size`` evaluated children."""
        wanted = self.population_size - self.elite_size
        parents = self.select(population, 2 * math.ceil(wanted / 2))
        genomes: list[G] = []
        for first, second in zip(parents[::2], parents[1::2], strict=True):
            a, b = first.genome, second.genome
            if self.rng.random() < self.crossover_rate:
                a, b = self.crossover(a, b)
            genomes.append(self.mutate(a))
            genomes.append(self.mutate(b))
        return self.evaluate(genomes[:wanted])

    def replace(
        self, population: Sequence[Individual[G]], offspring: Sequence[Individual[G]]
    ) -> list[Individual[G]]:
        """Build the next generation: the ``elite_size`` best parents plus all the offspring.

        The result is sorted best first for convenience; the engine does not depend on it.
        """
        elites = sorted(population, key=_score, reverse=True)[: self.elite_size]
        return sorted([*elites, *offspring], key=_score, reverse=True)

    def measure_diversity(self, population: Sequence[Individual[G]]) -> float | None:
        """Diversity of the population, with the ``diversity`` measure given to the constructor
        (default: :func:`carmelolg.galib.metrics.positional_diversity`). ``None`` if disabled or not measurable."""
        if self.diversity is None:
            return None
        return self.diversity([i.genome for i in population])

    def collect_stats(self) -> GenerationStats[G]:
        best = max(self.population, key=_score)
        worst = min(self.population, key=_score)
        return GenerationStats(
            generation=self.generation,
            best_fitness=best.fitness,
            mean_fitness=sum(i.fitness for i in self.population) / len(self.population),
            worst_fitness=worst.fitness,
            diversity=self.measure_diversity(self.population),
            best_genome=best.genome,
            evaluations=self.evaluations,
        )

    # ------------------------------------------------------------- internals

    def _record_generation(self) -> str | None:
        """Store the stats of the current generation, notify callbacks, check stop conditions."""
        stats = self.collect_stats()
        self.history.append(stats)
        for callback in self.callbacks:
            callback.on_generation(stats, self)
        for condition in self.stop:
            if condition.should_stop(stats, self):
                return type(condition).__name__
        return None

    def _resolve_mutation_rate(self) -> float:
        if self.mutation_rate is not None:
            return self.mutation_rate
        genome = self.population[0].genome
        if isinstance(genome, Sized) and len(genome) > 0:
            return 1 / len(genome)
        return 0.01
