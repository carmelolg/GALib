# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Built-in crossover operators.

* Sequences (str, bytes, tuple, list, numpy array) of equal length: ``OnePoint``, ``TwoPoint``, ``Uniform``.
* Permutations (every gene appears exactly once, e.g. a TSP tour): ``OrderCrossover``, ``PartiallyMappedCrossover``.
* Real-valued genes: ``BlendCrossover``.

Every operator returns two new children and never modifies the parents.
"""
from __future__ import annotations

from random import Random
from typing import Any, TypeVar, cast

from ._genome import (
    Bound,
    as_list,
    at,
    check_permutations,
    clip,
    clip_array,
    concat,
    is_array,
    np,
    numpy_rng,
    rebuild,
    same_length,
    validate_bounds,
)

G = TypeVar("G")


class OnePoint:
    """Swap the tails of the two parents after a random cut point."""

    def __call__(self, a: G, b: G, rng: Random, /) -> tuple[G, G]:
        x, y = cast(Any, a), cast(Any, b)
        length = same_length(x, y)
        if length < 2:
            return a, b
        cut = rng.randint(1, length - 1)
        return concat(x, x[:cut], y[cut:]), concat(y, y[:cut], x[cut:])

    def __repr__(self) -> str:
        return "OnePoint()"


class TwoPoint:
    """Swap the segment between two random cut points. Falls back to OnePoint below 3 genes."""

    def __call__(self, a: G, b: G, rng: Random, /) -> tuple[G, G]:
        x, y = cast(Any, a), cast(Any, b)
        length = same_length(x, y)
        if length < 3:
            return OnePoint()(a, b, rng)
        i, j = sorted(rng.sample(range(1, length), 2))
        return concat(x, x[:i], y[i:j], x[j:]), concat(y, y[:i], x[i:j], y[j:])

    def __repr__(self) -> str:
        return "TwoPoint()"


class Uniform:
    """Swap each gene between the two children with probability ``swap_probability``."""

    def __init__(self, swap_probability: float = 0.5) -> None:
        if not 0 <= swap_probability <= 1:
            raise ValueError("swap_probability must be in [0, 1]")
        self.swap_probability = swap_probability

    def __call__(self, a: G, b: G, rng: Random, /) -> tuple[G, G]:
        x, y = cast(Any, a), cast(Any, b)
        length = same_length(x, y)
        if is_array(x):
            mask = numpy_rng(rng).random(length) < self.swap_probability
            return np.where(mask, y, x), np.where(mask, x, y)
        child_x, child_y = [], []
        for gene_x, gene_y in zip(x, y, strict=True):
            if rng.random() < self.swap_probability:
                gene_x, gene_y = gene_y, gene_x
            child_x.append(gene_x)
            child_y.append(gene_y)
        return rebuild(x, child_x), rebuild(y, child_y)

    def __repr__(self) -> str:
        return f"Uniform(swap_probability={self.swap_probability})"


# ------------------------------------------------------------------ permutations

def _cut_points(length: int, rng: Random) -> tuple[int, int]:
    i, j = sorted(rng.sample(range(length + 1), 2))
    return i, j


class OrderCrossover:
    """Order crossover (OX1) for permutations.

    Each child keeps a random segment of one parent in place, and fills the remaining positions with the
    missing genes in the order they appear in the other parent (starting after the segment, wrapping around).
    Children are always valid permutations.
    """

    def __call__(self, a: G, b: G, rng: Random, /) -> tuple[G, G]:
        x, y = as_list(a), as_list(b)
        length = same_length(x, y)
        check_permutations(x, y)
        if length < 2:
            return rebuild(a, x), rebuild(b, y)
        i, j = _cut_points(length, rng)
        return rebuild(a, self._child(x, y, i, j)), rebuild(b, self._child(y, x, i, j))

    @staticmethod
    def _child(keep: list[Any], fill: list[Any], i: int, j: int) -> list[Any]:
        length = len(keep)
        child = list(keep)  # positions in [i, j) stay; the others are overwritten below
        kept = set(keep[i:j])
        donors = [gene for gene in (fill[(j + k) % length] for k in range(length)) if gene not in kept]
        positions = [(j + k) % length for k in range(length - (j - i))]
        for position, gene in zip(positions, donors, strict=True):
            child[position] = gene
        return child

    def __repr__(self) -> str:
        return "OrderCrossover()"


class PartiallyMappedCrossover:
    """Partially mapped crossover (PMX) for permutations.

    Each child copies a random segment of one parent; the genes of the other parent are placed at the
    same positions when possible, otherwise through the mapping defined by the segment. Children are
    always valid permutations, and keep more absolute positions than OrderCrossover.
    """

    def __call__(self, a: G, b: G, rng: Random, /) -> tuple[G, G]:
        x, y = as_list(a), as_list(b)
        length = same_length(x, y)
        check_permutations(x, y)
        if length < 2:
            return rebuild(a, x), rebuild(b, y)
        i, j = _cut_points(length, rng)
        return rebuild(a, self._child(x, y, i, j)), rebuild(b, self._child(y, x, i, j))

    @staticmethod
    def _child(segment_parent: list[Any], other: list[Any], i: int, j: int) -> list[Any]:
        empty = object()
        child: list[Any] = [empty] * len(other)
        child[i:j] = segment_parent[i:j]
        in_segment = set(segment_parent[i:j])
        position_in_other = {gene: index for index, gene in enumerate(other)}
        for k in range(i, j):
            gene = other[k]
            if gene in in_segment:
                continue
            position = k
            while i <= position < j:  # follow the mapping until we land outside the segment
                position = position_in_other[segment_parent[position]]
            child[position] = gene
        return [other[k] if gene is empty else gene for k, gene in enumerate(child)]

    def __repr__(self) -> str:
        return "PartiallyMappedCrossover()"


# ------------------------------------------------------------------- real values

class BlendCrossover:
    """Blend crossover (BLX-alpha) for real-valued genes.

    For each gene, with parents ``p`` and ``q`` and ``d = |p - q|``, each child gets a value drawn uniformly
    from ``[min(p, q) - alpha * d, max(p, q) + alpha * d]``, then clipped to ``[low, high]``.
    ``alpha = 0`` stays between the parents; larger values explore further. Bounds may be a single value
    or one value per gene.
    """

    def __init__(self, alpha: float = 0.5, low: Bound = None, high: Bound = None) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        validate_bounds(low, high)
        self.alpha = alpha
        self.low = low
        self.high = high

    def __call__(self, a: G, b: G, rng: Random, /) -> tuple[G, G]:
        x, y = cast(Any, a), cast(Any, b)
        same_length(x, y)
        if is_array(x):
            generator = numpy_rng(rng)
            lower, upper = np.minimum(x, y), np.maximum(x, y)
            spread = self.alpha * (upper - lower)
            lower, upper = lower - spread, upper + spread
            return (
                clip_array(generator.uniform(lower, upper), self.low, self.high),
                clip_array(generator.uniform(lower, upper), self.low, self.high),
            )
        children: tuple[list[float], list[float]] = ([], [])
        for index, (p, q) in enumerate(zip(x, y, strict=True)):
            spread = self.alpha * abs(p - q)
            lower, upper = min(p, q) - spread, max(p, q) + spread
            for child in children:
                child.append(clip(rng.uniform(lower, upper), at(self.low, index), at(self.high, index)))
        return rebuild(x, children[0]), rebuild(y, children[1])

    def __repr__(self) -> str:
        return f"BlendCrossover(alpha={self.alpha}, low={self.low!r}, high={self.high!r})"
