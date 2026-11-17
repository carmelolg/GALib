# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Hooks into a run."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from .core.engine import GeneticAlgorithm
    from .core.stats import GenerationStats, Result


class Callback:
    """Override only the methods you need; all of them do nothing by default."""

    def on_start(self, engine: GeneticAlgorithm[Any]) -> None:
        """Called once, after the initial population is built."""

    def on_generation(self, stats: GenerationStats[Any], engine: GeneticAlgorithm[Any]) -> None:
        """Called for every generation, generation 0 (the initial population) included."""

    def on_end(self, result: Result[Any], engine: GeneticAlgorithm[Any]) -> None:
        """Called once, when the run is over."""


class PrintProgress(Callback):
    """Print one line every ``every`` generations.

    :param every: print only generations that are multiples of ``every`` (generation 0 included).
    :param stream: where to write; ``None`` means ``sys.stdout`` at print time.
    :param width: maximum number of characters of the best genome to show.
    :param flush: passed to ``print``; ``True`` forces the line out immediately (useful when
        output is buffered, e.g. piped to a file or shown in an IDE console).
    :param end: passed to ``print`` (ignored when ``in_place`` is True).
    :param in_place: keep a single line that is rewritten at every generation. The carriage return is
        written before the text (``"\\r" + line``), as tqdm does: consoles such as PyCharm's erase a line
        that ends with ``"\\r"``, so ``end="\\r"`` may show nothing. Always flushes. A newline is
        printed when the run ends.

    To change the line format, subclass and override ``format_line``.
    """

    def __init__(
        self,
        every: int = 1,
        stream: TextIO | None = None,
        width: int = 60,
        *,
        flush: bool = False,
        end: str = "\n",
        in_place: bool = False,
    ) -> None:
        if every < 1:
            raise ValueError("every must be >= 1")
        if width < 0:
            raise ValueError("width must be >= 0")
        self.every = every
        self.stream = stream
        self.width = width
        self.flush = flush
        self.end = end
        self.in_place = in_place
        self._last_length = 0

    def format_line(self, stats: GenerationStats[Any]) -> str:
        """Return the text printed for one generation."""
        diversity = "  -  " if stats.diversity is None else f"{stats.diversity:.3f}"
        return (
            f"gen {stats.generation:5d} | best {stats.best_fitness:10.4g} | "
            f"mean {stats.mean_fitness:10.4g} | div {diversity} | {str(stats.best_genome)[: self.width]}"
        )

    def on_start(self, engine: GeneticAlgorithm[Any]) -> None:
        self._last_length = 0

    def on_generation(self, stats: GenerationStats[Any], engine: GeneticAlgorithm[Any]) -> None:
        if stats.generation % self.every:
            return
        line = self.format_line(stats)
        stream = self.stream or sys.stdout
        if self.in_place:
            # pad with spaces so a shorter line fully covers the previous one
            padding = " " * max(0, self._last_length - len(line))
            self._last_length = len(line)
            print("\r" + line + padding, file=stream, end="", flush=True)
        else:
            print(line, file=stream, end=self.end, flush=self.flush)

    def on_end(self, result: Result[Any], engine: GeneticAlgorithm[Any]) -> None:
        # leave the cursor on a fresh line after an in-place display
        if self.in_place or not self.end.endswith("\n"):
            print(file=self.stream or sys.stdout, flush=self.flush or self.in_place)
