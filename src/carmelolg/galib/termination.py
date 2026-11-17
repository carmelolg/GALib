# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Stop conditions. A run stops as soon as any of them returns True (or after ``max_generations``)."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.engine import GeneticAlgorithm
    from .core.stats import GenerationStats


class StopCondition:
    """Base class. Override ``should_stop``; override ``reset`` too if the condition keeps state.

    ``should_stop`` is called once per generation, generation 0 (the initial population) included.
    """

    def reset(self) -> None:
        """Called at the start of every run."""

    def should_stop(self, stats: GenerationStats[Any], engine: GeneticAlgorithm[Any]) -> bool:
        raise NotImplementedError


class TargetFitness(StopCondition):
    """Stop when the best fitness reaches ``target`` (``>=`` when maximising, ``<=`` when minimising)."""

    def __init__(self, target: float) -> None:
        self.target = target

    def should_stop(self, stats: GenerationStats[Any], engine: GeneticAlgorithm[Any]) -> bool:
        best = engine.best.fitness
        return best >= self.target if engine.problem.maximize else best <= self.target


class TimeLimit(StopCondition):
    """Stop once ``seconds`` have passed since the run started (initial population included).

    Checked between generations, so a run can exceed the limit by up to one generation.
    """

    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        self.seconds = seconds
        self._started = time.monotonic()

    def reset(self) -> None:
        self._started = time.monotonic()

    def should_stop(self, stats: GenerationStats[Any], engine: GeneticAlgorithm[Any]) -> bool:
        return time.monotonic() - self._started >= self.seconds


class Stagnation(StopCondition):
    """Stop after ``generations`` consecutive generations in which the best individual did not
    improve by more than ``min_delta``."""

    def __init__(self, generations: int, min_delta: float = 0.0) -> None:
        if generations < 1:
            raise ValueError("generations must be >= 1")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        self.generations = generations
        self.min_delta = min_delta
        self._best_score: float | None = None
        self._stale = 0

    def reset(self) -> None:
        self._best_score = None
        self._stale = 0

    def should_stop(self, stats: GenerationStats[Any], engine: GeneticAlgorithm[Any]) -> bool:
        score = engine.best.score
        if self._best_score is None or score > self._best_score + self.min_delta:
            self._best_score = score
            self._stale = 0
        else:
            self._stale += 1
        return self._stale >= self.generations
