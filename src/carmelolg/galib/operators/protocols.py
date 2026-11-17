# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Signatures every operator must follow.

Any callable with the right signature works: a plain function, a lambda, or an instance of a
class with ``__call__`` (the built-ins are classes, so they can be subclassed). Parameters are
positional-only, so your own functions may name them as they like.

Contract: never modify the genomes or individuals you receive, return new ones.
"""
from __future__ import annotations

from collections.abc import Sequence
from random import Random
from typing import Protocol, TypeVar

from ..core.individual import Individual

G = TypeVar("G")


class SelectionOperator(Protocol[G]):
    """Pick ``count`` parents from ``population`` (repetitions allowed). Prefer a higher ``score``."""

    def __call__(self, population: Sequence[Individual[G]], count: int, rng: Random, /) -> list[Individual[G]]:
        ...


class CrossoverOperator(Protocol[G]):
    """Combine two parent genomes into two child genomes."""

    def __call__(self, a: G, b: G, rng: Random, /) -> tuple[G, G]:
        ...


class MutationOperator(Protocol[G]):
    """Return a mutated copy of ``genome``. ``rate`` is the engine's per-gene mutation rate."""

    def __call__(self, genome: G, rng: Random, rate: float, /) -> G:
        ...
