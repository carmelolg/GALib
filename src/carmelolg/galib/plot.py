# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Optional plotting. Requires matplotlib: ``pip install "io.github.carmelolg.galib[plot]"``."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.stats import Result


def plot_history(result: Result[Any], show: bool = True) -> Any:
    """Plot best/mean fitness and diversity of a finished run. Returns the matplotlib figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("plot_history needs matplotlib: pip install matplotlib") from error

    history = result.history
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
    generations = [s.generation for s in history]
    top.plot(generations, [s.best_fitness for s in history], label="best")
    top.plot(generations, [s.mean_fitness for s in history], label="mean")
    top.set_ylabel("fitness")
    top.legend()
    measured = [s for s in history if s.diversity is not None]  # skip unmeasurable points, don't fake 0
    bottom.plot([s.generation for s in measured], [s.diversity for s in measured])
    bottom.set_ylabel("diversity")
    bottom.set_xlabel("generation")
    if show:
        plt.show()
    return fig
