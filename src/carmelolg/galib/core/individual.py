# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

G = TypeVar("G")
"""Type of a genome: whatever the user's Problem uses (str, list[int], tuple[float, ...], numpy array, ...)."""


@dataclass(frozen=True, eq=False)
class Individual(Generic[G]):
    """A genome together with its evaluation.

    ``fitness`` is the value returned by the Problem.
    ``score`` is the same value oriented so that higher is always better (``-fitness`` when
    ``Problem.maximize`` is False). Operators compare ``score``, so they never need to know
    whether the problem maximises or minimises.

    Individuals are immutable; treat ``genome`` as read-only even when it is a list or an array.
    They compare by identity (``eq=False``): genomes such as numpy arrays have no single truth value
    for ``==``. Compare ``genome`` explicitly when you need value equality.
    """

    genome: G
    fitness: float
    score: float
