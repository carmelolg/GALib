# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
import io
import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from carmelolg.galib import (
    Callback,
    GeneticAlgorithm,
    PrintProgress,
    Problem,
    Stagnation,
    StopCondition,
    TargetFitness,
    TimeLimit,
)
from carmelolg.galib.operators import Rank, Roulette, TwoPoint, Uniform

TARGET = "hello world"
ALPHABET = "abcdefghijklmnopqrstuvwxyz "


class Phrase(Problem[str]):
    def random_genome(self, rng):
        return "".join(rng.choice(ALPHABET) for _ in TARGET)

    def random_gene(self, rng):
        return rng.choice(ALPHABET)

    def fitness(self, genome):
        return sum(a == b for a, b in zip(genome, TARGET, strict=True))


class SumTo37(Problem[list]):
    """Minimisation: ten digits whose sum is as close as possible to 37."""

    maximize = False

    def random_genome(self, rng):
        return [rng.randint(0, 9) for _ in range(10)]

    def random_gene(self, rng):
        return rng.randint(0, 9)

    def fitness(self, genome):
        return abs(sum(genome) - 37)


class Constant(Problem[str]):
    def random_genome(self, rng):
        return "abc"

    def random_gene(self, rng):
        return "x"

    def fitness(self, genome):
        return 0


def ga(problem=None, **kwargs):
    kwargs.setdefault("population_size", 100)
    kwargs.setdefault("max_generations", 500)
    kwargs.setdefault("seed", 1)
    return GeneticAlgorithm(problem or Phrase(), **kwargs)


def best_genomes(result):
    return [s.best_genome for s in result.history]


# ------------------------------------------------------------------ basics

def test_converges_on_target():
    result = ga(stop=[TargetFitness(len(TARGET))]).run()
    assert result.best.genome == TARGET
    assert result.stopped_by == "TargetFitness"
    assert len(result.history) == result.generations + 1  # generation 0 included


@pytest.mark.parametrize("selection", [Roulette(), Rank()], ids=repr)
@pytest.mark.parametrize("crossover", [TwoPoint(), Uniform()], ids=repr)
def test_converges_with_every_builtin_operator(selection, crossover):
    result = ga(selection=selection, crossover=crossover, stop=[TargetFitness(len(TARGET))]).run()
    assert result.best.genome == TARGET


def test_minimisation_problem():
    result = ga(SumTo37(), stop=[TargetFitness(0)]).run()
    assert result.best.fitness == 0
    assert sum(result.best.genome) == 37


def test_same_seed_same_run_and_run_is_repeatable():
    engine = ga(max_generations=20)
    first, second = engine.run(), engine.run()
    assert best_genomes(first) == best_genomes(second) == best_genomes(ga(max_generations=20).run())


def test_different_seed_differs():
    first = best_genomes(ga(max_generations=5, seed=1).run())
    assert first != best_genomes(ga(max_generations=5, seed=2).run())


def test_elitism_never_loses_the_best():
    best = [s.best_fitness for s in ga(elite_size=2, max_generations=60).run().history]
    assert best == sorted(best)  # maximisation: never decreases


def test_population_size_is_constant():
    assert len(ga(population_size=41, elite_size=3, max_generations=10).run().population) == 41


def test_stops_at_max_generations():
    result = ga(max_generations=7).run()
    assert result.generations == 7 and result.stopped_by == "max_generations"


def test_zero_generations_only_evaluates_initial_population():
    result = ga(max_generations=0, population_size=30).run()
    assert result.generations == 0 and result.evaluations == 30 and len(result.history) == 1


def test_evaluations_do_not_reevaluate_elites():
    assert ga(population_size=50, elite_size=2, max_generations=4).run().evaluations == 50 + 4 * 48


def test_default_mutation_rate_is_one_over_length():
    engine = ga(max_generations=0)
    engine.run()
    assert engine.effective_mutation_rate == pytest.approx(1 / len(TARGET))
    explicit = ga(mutation_rate=0.3, max_generations=0)
    explicit.run()
    assert explicit.effective_mutation_rate == 0.3


# --------------------------------------------------------- stop conditions

def test_stagnation_counts_generations_without_improvement():
    result = ga(Constant(), stop=[Stagnation(5)], max_generations=100).run()
    assert result.stopped_by == "Stagnation"
    assert result.generations == 5


def test_target_reached_by_initial_population_stops_immediately():
    result = ga(Constant(), stop=[TargetFitness(0)]).run()
    assert result.generations == 0 and result.stopped_by == "TargetFitness"


def test_custom_stop_condition():
    class AfterThree(StopCondition):
        def should_stop(self, stats, engine):
            return stats.generation >= 3

    result = ga(stop=[AfterThree()]).run()
    assert result.generations == 3 and result.stopped_by == "AfterThree"


def test_stop_conditions_are_reset_between_runs():
    engine = ga(Constant(), stop=[Stagnation(3)])
    assert engine.run().generations == engine.run().generations == 3


# --------------------------------------------------------------- callbacks

def test_callbacks_are_called_in_order():
    events = []

    class Spy(Callback):
        def on_start(self, engine):
            events.append("start")

        def on_generation(self, stats, engine):
            events.append(stats.generation)

        def on_end(self, result, engine):
            events.append("end")

    ga(max_generations=3, callbacks=[Spy()]).run()
    assert events == ["start", 0, 1, 2, 3, "end"]


def test_print_progress():
    out = io.StringIO()
    ga(max_generations=4, callbacks=[PrintProgress(every=2, stream=out)]).run()
    assert [int(line.split("|")[0].split()[1]) for line in out.getvalue().splitlines()] == [0, 2, 4]
    with pytest.raises(ValueError):
        PrintProgress(every=0)
    with pytest.raises(ValueError):
        PrintProgress(width=-1)


def test_print_progress_flush_is_forwarded():
    class SpyStream(io.StringIO):
        flushes = 0

        def flush(self):
            self.flushes += 1
            super().flush()

    quiet, eager = SpyStream(), SpyStream()
    ga(max_generations=3, callbacks=[PrintProgress(stream=quiet)]).run()
    ga(max_generations=3, callbacks=[PrintProgress(stream=eager, flush=True)]).run()
    assert quiet.flushes == 0
    assert eager.flushes == 4  # generations 0..3


def test_print_progress_in_place_line():
    out = io.StringIO()
    ga(max_generations=2, callbacks=[PrintProgress(stream=out, end="\r")]).run()
    text = out.getvalue()
    assert text.count("\r") == 3 and text.endswith("\r\n")


def test_print_progress_in_place_writes_carriage_return_first():
    out = io.StringIO()
    ga(max_generations=2, callbacks=[PrintProgress(stream=out, in_place=True)]).run()
    chunks = out.getvalue().split("\r")
    assert chunks[0] == ""  # output starts with "\r"
    assert len(chunks) == 4  # generations 0, 1, 2
    assert all(c.startswith("gen") for c in chunks[1:])
    assert out.getvalue().endswith("\n")


def test_print_progress_in_place_pads_shorter_lines():
    class Shrinking(PrintProgress):
        def format_line(self, stats):
            return "x" * (10 - stats.generation)

    out = io.StringIO()
    ga(max_generations=2, callbacks=[Shrinking(stream=out, in_place=True)]).run()
    assert out.getvalue() == "\r" + "x" * 10 + "\r" + "x" * 9 + " " + "\r" + "x" * 8 + " " + "\n"


def test_print_progress_custom_format():
    class Short(PrintProgress):
        def format_line(self, stats):
            return f"{stats.generation}:{stats.best_fitness:g}"

    out = io.StringIO()
    ga(max_generations=1, callbacks=[Short(stream=out)]).run()
    assert out.getvalue().splitlines()[0].startswith("0:")


# -------------------------------------------------------- extension points

def test_explicit_operators_are_used():
    calls = {"crossover": 0, "mutation": 0, "selection": 0}

    def my_crossover(a, b, rng):
        calls["crossover"] += 1
        return a, b

    def my_mutation(genome, rng, rate):
        calls["mutation"] += 1
        return genome

    def my_selection(population, count, rng):
        calls["selection"] += 1
        return [population[0]] * count

    ga(crossover=my_crossover, mutation=my_mutation, selection=my_selection, max_generations=2).run()
    assert all(calls.values())


def test_selection_returning_wrong_count_is_reported():
    with pytest.raises(ValueError, match="parents"):
        ga(selection=lambda population, count, rng: [population[0]], max_generations=1).run()


def test_problem_overrides_are_used_and_explicit_arguments_win():
    calls = []

    class Custom(Phrase):
        def crossover(self, a, b, rng):
            calls.append("problem")
            return a, b

    ga(Custom(), max_generations=2).run()
    assert "problem" in calls

    calls.clear()
    ga(Custom(), crossover=Uniform(), max_generations=2).run()
    assert calls == []


def test_problem_mutate_override_replaces_default_mutation():
    class FlipOne(Problem[str]):
        def random_genome(self, rng):
            return "aaaa"

        def fitness(self, genome):
            return genome.count("b")

        def mutate(self, genome, rng, rate):
            i = rng.randrange(len(genome))
            return genome[:i] + "b" + genome[i + 1:]

    assert ga(FlipOne(), stop=[TargetFitness(4)]).run().best.genome == "bbbb"


def test_subclass_hooks_are_called():
    class Counting(GeneticAlgorithm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.replaced = 0

        def replace(self, population, offspring):
            self.replaced += 1
            return super().replace(population, offspring)

    engine = Counting(Phrase(), max_generations=5, seed=1)
    engine.run()
    assert engine.replaced == 5


def test_unsorted_replace_still_gives_correct_stats_and_best():
    class Unsorted(GeneticAlgorithm):
        def replace(self, population, offspring):
            return sorted(super().replace(population, offspring), key=lambda i: i.score)  # worst first

    result = Unsorted(Phrase(), max_generations=30, seed=1).run()
    for stats in result.history:
        assert stats.best_fitness >= stats.mean_fitness >= stats.worst_fitness
    assert result.best.fitness == max(s.best_fitness for s in result.history)


def test_mu_plus_lambda_via_subclass():
    class MuPlusLambda(GeneticAlgorithm):
        def replace(self, population, offspring):
            both = sorted([*population, *offspring], key=lambda i: i.score, reverse=True)
            return both[: self.population_size]

    result = MuPlusLambda(Phrase(), elite_size=0, population_size=60, max_generations=300, seed=1,
                          stop=[TargetFitness(len(TARGET))]).run()
    assert result.best.genome == TARGET


def test_manual_loop_with_initialize_and_step():
    engine = ga()
    with pytest.raises(RuntimeError, match="initialize"):
        engine.step()
    engine.initialize()
    for _ in range(3):
        engine.step()
    assert engine.generation == 3


def test_executor_gives_same_result():
    plain = ga(max_generations=15).run()
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = ga(max_generations=15, executor=pool).run()
    assert best_genomes(plain) == best_genomes(parallel)


# ---------------------------------------------------------- fitness_batch

def test_fitness_batch_only_problem():
    class Batch(Problem[str]):
        calls = 0

        def random_genome(self, rng):
            return "".join(rng.choice(ALPHABET) for _ in TARGET)

        def random_gene(self, rng):
            return rng.choice(ALPHABET)

        def fitness_batch(self, genomes):
            type(self).calls += 1
            return [sum(a == b for a, b in zip(g, TARGET, strict=True)) for g in genomes]

    problem = Batch()
    result = ga(problem, max_generations=3).run()
    assert Batch.calls == 4  # one batch per generation, initial population included
    assert problem.fitness(TARGET) == len(TARGET)  # fitness() falls back on fitness_batch()
    assert result.evaluations == 100 + 3 * 98


def test_problem_without_fitness_is_rejected():
    class Nothing(Problem[str]):
        def random_genome(self, rng):
            return "a"

    with pytest.raises(TypeError, match="fitness"):
        GeneticAlgorithm(Nothing())


def test_fitness_batch_wrong_length_is_reported():
    class Short(Phrase):
        def fitness_batch(self, genomes):
            return [0.0]

    with pytest.raises(ValueError, match="fitness_batch"):
        ga(Short()).run()


# ------------------------------------------------------------ time limit

def test_time_limit():
    import time

    class Slow(Phrase):
        def fitness(self, genome):
            time.sleep(0.0005)
            return super().fitness(genome)

    started = time.monotonic()
    result = ga(Slow(), population_size=20, stop=[TimeLimit(0.2)], max_generations=10_000).run()
    assert result.stopped_by == "TimeLimit"
    assert time.monotonic() - started < 1.0
    with pytest.raises(ValueError):
        TimeLimit(0)


# -------------------------------------------------------------- validation

def test_default_mutation_needs_random_gene():
    class NoGene(Problem[str]):
        def random_genome(self, rng):
            return "abc"

        def fitness(self, genome):
            return 0

    with pytest.raises(NotImplementedError, match="random_gene"):
        ga(NoGene(), max_generations=1).run()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_fitness_is_rejected(bad):
    class Broken(Constant):
        def fitness(self, genome):
            return bad

    with pytest.raises(ValueError, match="finite"):
        ga(Broken()).run()


@pytest.mark.parametrize("kwargs", [
    {"population_size": 1},
    {"max_generations": -1},
    {"elite_size": 100},
    {"elite_size": -1},
    {"crossover_rate": 1.5},
    {"mutation_rate": -0.1},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        GeneticAlgorithm(Phrase(), **kwargs)


def test_stagnation_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        Stagnation(0)
    with pytest.raises(ValueError):
        Stagnation(5, min_delta=-1)


# --------------------------------------------------------------- diversity

def test_diversity():
    engine = ga(max_generations=0)
    stats = engine.run().history[0]
    assert 0 < stats.diversity <= 1
    assert ga(Constant(), max_generations=0).run().history[0].diversity == 0.0


def test_diversity_is_none_when_not_measurable():
    class Scalar(Problem[float]):
        def random_genome(self, rng):
            return rng.random()

        def fitness(self, genome):
            return genome

        def crossover(self, a, b, rng):
            return a, b

        def mutate(self, genome, rng, rate):
            return genome

    result = ga(Scalar(), max_generations=2).run()
    assert all(s.diversity is None for s in result.history)


def test_diversity_can_be_disabled_or_replaced():
    assert ga(max_generations=1, diversity=None).run().history[0].diversity is None
    assert ga(max_generations=1, diversity=lambda genomes: 0.42).run().history[0].diversity == 0.42
