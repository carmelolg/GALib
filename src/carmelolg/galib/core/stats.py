# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic

from .individual import G, Individual


@dataclass(frozen=True)
class GenerationStats(Generic[G]):
    """Snapshot of the population after a generation (generation 0 = initial population)."""

    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float | None  # see GeneticAlgorithm.measure_diversity; None if not measurable
    best_genome: G | None  # None in ``result.history`` for generations skipped by ``history_genome_every``
    evaluations: int  # fitness evaluations so far


@dataclass(frozen=True)
class Result(Generic[G]):
    """Outcome of ``GeneticAlgorithm.run``."""

    best: Individual[G]  # best individual ever seen, not only in the last generation
    generations: int  # evolved generations (len(history) == generations + 1)
    stopped_by: str  # "max_generations" or the class name of the StopCondition that fired
    evaluations: int
    history: list[GenerationStats[G]] = field(default_factory=list)
    population: list[Individual[G]] = field(default_factory=list)
