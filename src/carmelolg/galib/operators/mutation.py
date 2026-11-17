# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Built-in mutation operators.

All of them use the engine's per-gene ``rate``, so that the expected number of mutation events per child is
``rate * len(genome)`` (one on average with the default rate ``1 / len(genome)``). Mutated positions are drawn
with geometric jumps, so the cost grows with the number of mutations, not with the genome length.

* Any sequence genome: ``RandomReset`` (also the default ``Problem.mutate``).
* Permutations: ``SwapMutation``, ``InversionMutation`` (children stay valid permutations).
* Real-valued genes: ``GaussianMutation``.

Every operator returns a new genome and never modifies its input.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from random import Random
from typing import Any, TypeVar, cast

from ._genome import Bound, as_list, at, bernoulli_positions, clip, is_array, rebuild, validate_bounds

G = TypeVar("G")

GeneFactory = Callable[[Random], Any]
"""Returns one random gene, e.g. ``lambda rng: rng.choice("abc")``."""


def _copy(genome: Any) -> Any:
    """A mutable copy of the genes: the array itself for numpy, a list otherwise."""
    return genome.copy() if is_array(genome) else as_list(genome)


def _finish(template: Any, genes: Any) -> Any:
    return genes if is_array(template) else rebuild(template, genes)


def random_reset(genome: G, rng: Random, rate: float, random_gene: GeneFactory) -> G:
    """Replace each gene independently, with probability ``rate``, by ``random_gene(rng)``.

    A child can therefore get 0, 1 or more mutations; on average ``len(genome) * rate``.
    """
    value = cast(Any, genome)
    genes = _copy(value)
    for index in bernoulli_positions(len(genes), rate, rng):
        genes[index] = random_gene(rng)
    return cast(G, _finish(value, genes))


class RandomReset:
    """Operator form of :func:`random_reset`, for use as ``mutation=RandomReset(...)``."""

    def __init__(self, random_gene: GeneFactory) -> None:
        self.random_gene = random_gene

    def __call__(self, genome: G, rng: Random, rate: float, /) -> G:
        return random_reset(genome, rng, rate, self.random_gene)

    def __repr__(self) -> str:
        return f"RandomReset({self.random_gene!r})"


# ------------------------------------------------------------------ permutations

class SwapMutation:
    """For each position, with probability ``rate``, swap its gene with the gene at a random position."""

    def __call__(self, genome: G, rng: Random, rate: float, /) -> G:
        value = cast(Any, genome)
        genes = _copy(value)
        length = len(genes)
        if length >= 2:
            for index in bernoulli_positions(length, rate, rng):
                other = rng.randrange(length)
                genes[index], genes[other] = genes[other], genes[index]
        return cast(G, _finish(value, genes))

    def __repr__(self) -> str:
        return "SwapMutation()"


class InversionMutation:
    """With probability ``min(1, rate * len(genome))``, reverse the order of a random segment.

    Well suited to routing problems: reversing a segment of a tour changes only two edges.
    """

    def __call__(self, genome: G, rng: Random, rate: float, /) -> G:
        value = cast(Any, genome)
        genes = _copy(value)
        length = len(genes)
        if length >= 2 and rng.random() < min(1.0, rate * length):
            i, j = sorted(rng.sample(range(length + 1), 2))
            genes[i:j] = genes[i:j][::-1]
        return cast(G, _finish(value, genes))

    def __repr__(self) -> str:
        return "InversionMutation()"


# ------------------------------------------------------------------- real values

class GaussianMutation:
    """For each gene, with probability ``rate``, add noise drawn from a normal distribution N(0, sigma),
    then clip to ``[low, high]``.

    ``sigma``, ``low`` and ``high`` may be a single value or one value per gene.
    """

    def __init__(self, sigma: float | Sequence[float], low: Bound = None, high: Bound = None) -> None:
        sigmas = [sigma] if isinstance(sigma, (int, float)) else list(sigma)
        if any(s <= 0 for s in sigmas):
            raise ValueError("sigma must be > 0")
        validate_bounds(low, high)
        self.sigma = sigma
        self.low = low
        self.high = high

    def __call__(self, genome: G, rng: Random, rate: float, /) -> G:
        value = cast(Any, genome)
        genes = _copy(value)
        if is_array(genes) and genes.dtype.kind != "f":
            genes = genes.astype(float)  # adding noise to an integer array would truncate it
        for index in bernoulli_positions(len(genes), rate, rng):
            noisy = float(genes[index]) + rng.gauss(0.0, at(self.sigma, index))
            genes[index] = clip(noisy, at(self.low, index), at(self.high, index))
        return cast(G, _finish(value, genes))

    def __repr__(self) -> str:
        return f"GaussianMutation(sigma={self.sigma!r}, low={self.low!r}, high={self.high!r})"
