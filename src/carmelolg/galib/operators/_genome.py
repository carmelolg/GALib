# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the built-in operators.

Built-in operators work on sequence genomes: str, bytes, tuple, list and, when numpy is installed,
one-dimensional ``numpy.ndarray`` (with vectorised fast paths).
"""
from __future__ import annotations

import math
import weakref
from collections.abc import Iterable, Iterator, Sequence
from random import Random
from typing import Any

np: Any
try:
    import numpy

    np = numpy
except ImportError:  # pragma: no cover - numpy is an optional dependency
    np = None

SUPPORTED = "str, bytes, tuple, list and numpy.ndarray"

Bound = float | Sequence[float] | None
"""A bound for real-valued genes: one value for every gene, one value per gene, or no bound."""


def is_array(value: Any) -> bool:
    """True if ``value`` is a numpy array (always False when numpy is not installed)."""
    return np is not None and isinstance(value, np.ndarray)


_numpy_generators: weakref.WeakKeyDictionary[Random, Any] = weakref.WeakKeyDictionary()


def numpy_rng(rng: Random) -> Any:
    """The numpy ``Generator`` paired with ``rng``.

    Created on first use, seeded from ``rng``, then reused for as long as ``rng`` lives (the engine creates
    a new ``Random`` for every run). Vectorised operators therefore stay reproducible with the run's seed
    without paying the cost of a new generator at every call.
    """
    generator = _numpy_generators.get(rng)
    if generator is None:
        generator = np.random.default_rng(rng.getrandbits(64))
        _numpy_generators[rng] = generator
    return generator


def bernoulli_positions(length: int, rate: float, rng: Random) -> Iterator[int]:
    """Positions in ``range(length)`` selected independently with probability ``rate``.

    Equivalent to testing ``rng.random() < rate`` for each position, but it jumps from one selected
    position to the next with geometrically distributed gaps, so it costs one random draw per selected
    position instead of one per position: with the usual rate ``1 / length`` that is about one draw
    instead of ``length``.
    """
    if rate <= 0 or length <= 0:
        return
    if rate >= 1:
        yield from range(length)
        return
    log_q = math.log1p(-rate)
    position = -1
    while True:
        # number of failures before the next success: floor(log(U) / log(1 - rate)), U in (0, 1]
        position += 1 + int(math.log(1.0 - rng.random()) / log_q)
        if position >= length:
            return
        yield position


def rebuild(template: Any, items: Iterable[Any]) -> Any:
    """Build a genome of the same kind as ``template`` from ``items``.

    Raises TypeError instead of silently changing the genome type.
    """
    if isinstance(template, str):
        return "".join(items)
    if isinstance(template, bytes):
        return bytes(items)
    if isinstance(template, tuple):
        return tuple(items)
    if isinstance(template, list):
        return list(items)
    if is_array(template):
        return np.asarray(list(items), dtype=template.dtype)
    raise TypeError(
        f"built-in operators support {SUPPORTED} genomes, got {type(template).__name__}: "
        "override crossover()/mutate() in your Problem or pass custom operators"
    )


def concat(template: Any, *parts: Any) -> Any:
    """Concatenate slices of genomes into a genome of the same kind as ``template``."""
    if is_array(template):
        return np.concatenate(parts)
    return rebuild(template, [gene for part in parts for gene in part])


def same_length(a: Any, b: Any) -> int:
    """Return the common length of two parents, or raise if they differ."""
    if len(a) != len(b):
        raise ValueError(f"parents must have the same length, got {len(a)} and {len(b)}")
    return len(a)


def as_list(genome: Any) -> list[Any]:
    """Genes as a plain list (numpy arrays are converted with ``tolist`` to get Python scalars)."""
    if is_array(genome):
        return list(genome.tolist())
    return list(genome)


def check_permutations(a: list[Any], b: list[Any]) -> None:
    """Raise if ``a`` and ``b`` are not permutations of the same distinct genes."""
    genes = set(a)
    if len(genes) != len(a) or genes != set(b):
        raise ValueError("permutation operators need parents that are permutations of the same distinct genes")


def at(value: Any, index: int) -> Any:
    """``value[index]`` for a per-gene sequence, ``value`` itself for a scalar or None."""
    if value is None or isinstance(value, (int, float)):
        return value
    return value[index]


def clip(value: float, low: float | None, high: float | None) -> float:
    if low is not None and value < low:
        return low
    if high is not None and value > high:
        return high
    return value


def clip_array(values: Any, low: Bound, high: Bound) -> Any:
    if low is None and high is None:
        return values
    return np.clip(values, -np.inf if low is None else low, np.inf if high is None else high)


def validate_bounds(low: Bound, high: Bound) -> None:
    if low is None or high is None:
        return
    lows = [low] if isinstance(low, (int, float)) else list(low)
    highs = [high] if isinstance(high, (int, float)) else list(high)
    if len(lows) == 1:
        lows = lows * len(highs)
    if len(highs) == 1:
        highs = highs * len(lows)
    if len(lows) != len(highs):
        raise ValueError("per-gene low and high bounds must have the same length")
    if any(lo > hi for lo, hi in zip(lows, highs, strict=True)):
        raise ValueError("low bound must not be greater than high bound")
