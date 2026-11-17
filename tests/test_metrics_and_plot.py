# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
import pytest

from carmelolg.galib import GeneticAlgorithm, Problem
from carmelolg.galib.metrics import gene_spread, positional_diversity


def test_positional_diversity_edge_cases():
    assert positional_diversity([]) is None
    assert positional_diversity([1, 2]) is None  # not sequences
    assert positional_diversity(["ab", "abc"]) is None  # unequal lengths
    assert positional_diversity(["", ""]) == 0.0
    assert positional_diversity([[[1]], [[2]]]) is None  # unhashable genes
    assert positional_diversity(["ab", "ab", "cd", "cd"]) == pytest.approx(0.5)


def test_gene_spread_edge_cases():
    assert gene_spread([]) is None
    assert gene_spread([[], []]) is None
    assert gene_spread([[1.0, "x"], [2.0, "y"]]) is None


def test_metrics_on_numpy_edge_cases():
    np = pytest.importorskip("numpy")
    assert positional_diversity([np.zeros((2, 2)), np.zeros((2, 2))]) is None  # not 1-D
    assert positional_diversity([np.zeros(0), np.zeros(0)]) == 0.0
    assert gene_spread([np.array(["a"]), np.array(["b"])]) is None  # non-numeric
    assert gene_spread([np.zeros(2), np.zeros(3)]) is None


class Ones(Problem[list]):
    def random_genome(self, rng):
        return [rng.randint(0, 1) for _ in range(8)]

    def random_gene(self, rng):
        return rng.randint(0, 1)

    def fitness(self, genome):
        return sum(genome)


def test_plot_history():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")  # no window
    from carmelolg.galib.plot import plot_history

    result = GeneticAlgorithm(Ones(), population_size=20, max_generations=5, seed=1).run()
    fig = plot_history(result, show=False)
    top, bottom = fig.axes
    assert len(top.lines) == 2 and len(bottom.lines) == 1
    assert list(top.lines[0].get_xdata()) == list(range(6))


def test_plot_history_skips_unmeasurable_diversity():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from carmelolg.galib.plot import plot_history

    result = GeneticAlgorithm(Ones(), population_size=20, max_generations=3, seed=1, diversity=None).run()
    assert len(plot_history(result, show=False).axes[1].lines[0].get_xdata()) == 0
