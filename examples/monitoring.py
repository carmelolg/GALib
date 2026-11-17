# SPDX-FileCopyrightText: 2026 CARMELO LA GAMBA
# SPDX-License-Identifier: Apache-2.0
"""Read the results of a run: the Result, the per-generation history, diversity metrics and plots.

Run from the project root: python examples/monitoring.py   (plots need: pip install matplotlib)
"""
from itertools import pairwise

from phrase import TARGET, PhraseProblem

from carmelolg.galib import GeneticAlgorithm, TargetFitness
from carmelolg.galib.metrics import positional_diversity

result = GeneticAlgorithm(
    PhraseProblem(),
    population_size=200,
    stop=[TargetFitness(len(TARGET))],
    seed=42,
).run()

# 1. The Result: what the run found and why it stopped.
print(f"best genome : {result.best.genome!r}")
print(f"fitness     : {result.best.fitness:g} of {len(TARGET)}")
print(f"generations : {result.generations}")
print(f"stopped by  : {result.stopped_by}")
print(f"evaluations : {result.evaluations}")

# 2. The history: one GenerationStats per generation, generation 0 included.
history = result.history
first, last = history[0], history[-1]
print(f"\nbest fitness went from {first.best_fitness:g} to {last.best_fitness:g}")
print(f"mean fitness went from {first.mean_fitness:.1f} to {last.mean_fitness:.1f}")
print(f"diversity went from {first.diversity:.2f} to {last.diversity:.2f}")

# 3. Series for your own analysis: any field of GenerationStats.
best_series = [s.best_fitness for s in history]
half = len(TARGET) / 2
generation_at_half = next(s.generation for s in history if s.best_fitness >= half)
print(f"half of the phrase was right at generation {generation_at_half}")
print(f"evaluations to get there: {history[generation_at_half].evaluations}")
stalled = sum(before == after for before, after in pairwise(best_series))
print(f"generations where the best did not improve: {stalled}")

# 4. Metrics on any population: here the final one (result.population holds Individual objects).
genomes = [individual.genome for individual in result.population]
print(f"\nfinal diversity measured again: {positional_diversity(genomes):.2f}")
print(f"distinct genomes in the final population: {len(set(genomes))} of {len(genomes)}")

# 5. Plots.
try:
    from carmelolg.galib.plot import plot_history

    figure = plot_history(result, show=False)  # show=False: get the figure without opening a window
    figure.savefig("history.png", dpi=150)
    print("\nplot saved to history.png")
except ImportError:
    print("\nmatplotlib is not installed: pip install matplotlib")
