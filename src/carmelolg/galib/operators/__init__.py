# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Genetic operators: protocols and built-in implementations.

| kind      | sequences                         | permutations                                  | real values         |
|-----------|-----------------------------------|-----------------------------------------------|---------------------|
| selection | Tournament, Roulette, Rank (any genome)                                                                 |
| crossover | OnePoint, TwoPoint, Uniform       | OrderCrossover, PartiallyMappedCrossover      | BlendCrossover      |
| mutation  | RandomReset                       | SwapMutation, InversionMutation               | GaussianMutation    |
"""
from .crossover import BlendCrossover, OnePoint, OrderCrossover, PartiallyMappedCrossover, TwoPoint, Uniform
from .mutation import GaussianMutation, GeneFactory, InversionMutation, RandomReset, SwapMutation, random_reset
from .protocols import CrossoverOperator, MutationOperator, SelectionOperator
from .selection import Rank, Roulette, Tournament

__all__ = [
    "BlendCrossover",
    "CrossoverOperator",
    "GaussianMutation",
    "GeneFactory",
    "InversionMutation",
    "MutationOperator",
    "OnePoint",
    "OrderCrossover",
    "PartiallyMappedCrossover",
    "RandomReset",
    "Rank",
    "Roulette",
    "SelectionOperator",
    "SwapMutation",
    "Tournament",
    "TwoPoint",
    "Uniform",
    "random_reset",
]
