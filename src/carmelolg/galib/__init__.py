# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""carmelolg.galib: a small, extensible library to build genetic algorithms.

Minimal use::

    from carmelolg.galib import GeneticAlgorithm, Problem

    class MyProblem(Problem[str]):
        def random_genome(self, rng): ...
        def fitness(self, genome): ...

    result = GeneticAlgorithm(MyProblem()).run()

Operators live in ``carmelolg.galib.operators``, diversity measures in ``carmelolg.galib.metrics``,
plotting in ``carmelolg.galib.plot``.
"""
from .callbacks import Callback, PrintProgress
from .core import GenerationStats, GeneticAlgorithm, Individual, Problem, Result
from .termination import Stagnation, StopCondition, TargetFitness, TimeLimit

__version__ = "0.4.0"  # x-release-please-version

__all__ = [
    "Callback",
    "GenerationStats",
    "GeneticAlgorithm",
    "Individual",
    "PrintProgress",
    "Problem",
    "Result",
    "Stagnation",
    "StopCondition",
    "TargetFitness",
    "TimeLimit",
]
