# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
import pytest

from carmelolg.galib import Callback, GeneticAlgorithm, Problem

TARGET = "history"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class Phrase(Problem[str]):
    def random_genome(self, rng):
        return "".join(rng.choice(ALPHABET) for _ in TARGET)

    def random_gene(self, rng):
        return rng.choice(ALPHABET)

    def fitness(self, genome):
        return sum(a == b for a, b in zip(genome, TARGET, strict=True))


def run(**kwargs):
    return GeneticAlgorithm(Phrase(), population_size=12, max_generations=10, seed=4, **kwargs).run()


def test_default_keeps_every_genome():
    assert all(s.best_genome is not None for s in run().history)


@pytest.mark.parametrize(("every", "kept"), [(1, range(11)), (3, [0, 3, 6, 9]), (5, [0, 5, 10]), (0, [])])
def test_history_genome_every(every, kept):
    result = run(history_genome_every=every)
    assert [s.generation for s in result.history if s.best_genome is not None] == list(kept)
    assert len(result.history) == 11  # the statistics are always there


def test_thinning_changes_nothing_else():
    full, thin = run(), run(history_genome_every=0)
    assert [s.best_fitness for s in full.history] == [s.best_fitness for s in thin.history]
    assert full.best.genome == thin.best.genome and full.evaluations == thin.evaluations


def test_callbacks_still_receive_the_genome():
    seen = []

    class Collect(Callback):
        def on_generation(self, stats, engine):
            seen.append(stats.best_genome)

    run(history_genome_every=0, callbacks=[Collect()])
    assert len(seen) == 11 and all(genome is not None for genome in seen)


def test_negative_value_is_rejected():
    with pytest.raises(ValueError, match="history_genome_every"):
        GeneticAlgorithm(Phrase(), history_genome_every=-1)
