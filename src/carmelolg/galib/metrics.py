# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Population diversity measures, for ``GeneticAlgorithm(diversity=...)``.

A measure receives the genomes of the current population and returns a number, or ``None`` when it cannot
be computed for those genomes. Write your own with the same signature for other genome types.
"""
from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from .operators._genome import is_array, np

DiversityMeasure = Callable[[Sequence[Any]], float | None]


def _matrix(genomes: Sequence[Any]) -> Any:
    """Stack equal-length 1-D numpy genomes into an (N, L) matrix, or return None."""
    try:
        matrix = np.stack(genomes)
    except ValueError:  # different shapes
        return None
    return matrix if matrix.ndim == 2 else None


def _most_common_counts(matrix: Any) -> Any:
    """For each column of an (N, L) matrix, how many times its most common value occurs. Fully vectorised:
    sort each column, number the runs of equal values, count the run lengths with one bincount."""
    rows, columns = matrix.shape
    ordered = np.sort(matrix, axis=0)
    starts = np.ones((rows, columns), dtype=bool)
    starts[1:] = ordered[1:] != ordered[:-1]
    run_ids = np.cumsum(starts, axis=0) - 1  # 0..rows-1 within each column
    offsets = run_ids + np.arange(columns) * rows  # make run ids unique across columns
    run_lengths = np.bincount(offsets.ravel(order="F"), minlength=rows * columns).reshape(columns, rows)
    return run_lengths.max(axis=1)


def positional_diversity(genomes: Sequence[Any]) -> float | None:
    """Diversity in [0, 1] for sequence genomes of equal length (str, list, tuple, bytes, numpy array).

    For each gene position, the share of individuals that do not carry the most common gene, averaged over
    positions. 0 = every individual is identical (premature convergence).
    Genes are compared by equality: for real-valued genes use :func:`gene_spread`.
    Returns ``None`` for other genomes, unequal lengths or unhashable genes.
    """
    if not genomes:
        return None
    count = len(genomes)
    if all(is_array(g) for g in genomes):
        matrix = _matrix(genomes)
        if matrix is None:
            return None
        if matrix.shape[1] == 0:
            return 0.0
        return float(1 - _most_common_counts(matrix).mean() / count)
    if not all(isinstance(g, Sequence) for g in genomes):
        return None
    length = len(genomes[0])
    if any(len(g) != length for g in genomes):
        return None
    if length == 0:
        return 0.0
    try:
        tops = [Counter(column).most_common(1)[0][1] for column in zip(*genomes, strict=True)]
    except TypeError:  # unhashable genes
        return None
    return sum(1 - top / count for top in tops) / length


def gene_spread(genomes: Sequence[Any]) -> float | None:
    """Mean, over gene positions, of the population standard deviation of each gene.

    Meant for real-valued genomes (list, tuple or numpy array of numbers). It is in the same unit as the
    genes, so compare it with the size of the search space. Returns ``None`` for non-numeric genomes or
    unequal lengths.
    """
    if not genomes:
        return None
    if all(is_array(g) for g in genomes):
        matrix = _matrix(genomes)
        if matrix is None or matrix.shape[1] == 0 or not np.issubdtype(matrix.dtype, np.number):
            return None
        return float(matrix.astype(float).std(axis=0).mean())
    if not all(isinstance(g, (list, tuple)) for g in genomes):
        return None
    length = len(genomes[0])
    if length == 0 or any(len(g) != length for g in genomes):
        return None
    try:
        spreads = [statistics.pstdev(column) for column in zip(*genomes, strict=True)]
    except TypeError:  # non-numeric genes
        return None
    return float(sum(spreads) / length)
