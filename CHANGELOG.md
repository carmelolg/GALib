# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html): while the version is `0.x` the API may still change.

## [Unreleased]

### Added

- `Problem` base class: `random_genome` plus `fitness` or `fitness_batch` (vectorised evaluation), optional
  `random_gene`, `crossover`, `mutate` and `maximize`.
- `GeneticAlgorithm` engine with tournament selection, crossover rate, per-gene mutation rate (default
  `1 / genome length`), elitism, seeded and independent runs, optional `executor` for parallel fitness, and
  overridable building blocks (`initial_population`, `evaluate`, `select`, `crossover`, `mutate`, `reproduce`,
  `replace`, `measure_diversity`, `collect_stats`); `initialize` and `step` to drive the loop manually.
- Selection operators: `Tournament`, `Roulette`, `Rank`.
- Crossover operators: `OnePoint`, `TwoPoint`, `Uniform`; for permutations `OrderCrossover`,
  `PartiallyMappedCrossover`; for real values `BlendCrossover`.
- Mutation operators: `RandomReset`; for permutations `SwapMutation`, `InversionMutation`; for real values
  `GaussianMutation`. Mutated positions are drawn with geometric jumps (cost proportional to the number of
  mutations, not to the genome length).
- Optional numpy support: numpy array genomes in every built-in operator and metric, vectorised fast paths.
- Diversity measures `positional_diversity` (default) and `gene_spread`, selectable with `diversity=`.
- Stop conditions `TargetFitness`, `Stagnation`, `TimeLimit`; callbacks `Callback`, `PrintProgress`
  (`flush`, `end`, `in_place`, overridable `format_line`); `plot_history` (optional matplotlib).
- Examples (phrase, knapsack, TSP, Rastrigin), numpy benchmark, documentation site sources (MkDocs Material),
  CI workflows.
- Apache-2.0 license with NOTICE file.
