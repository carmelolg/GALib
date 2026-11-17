# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Core abstractions: the problem, the individual, the engine and its results."""
from .engine import GeneticAlgorithm
from .individual import Individual
from .problem import Problem
from .stats import GenerationStats, Result

__all__ = ["GenerationStats", "GeneticAlgorithm", "Individual", "Problem", "Result"]
