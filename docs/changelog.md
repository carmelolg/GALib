# Changelog

Every release is listed here with its changes, grouped by type. Releases are prepared automatically from the
commit messages, so the entries below are always in sync with the code.

## How releases are made

Commits to `main` follow [Conventional Commits](https://www.conventionalcommits.org/). The type decides the next
version number:

| Commit type | Example | Effect while the version is `0.x` |
|---|---|---|
| `fix:` | `fix: keep elites when the population shrinks` | patch (`0.1.0` to `0.1.1`) |
| `feat:` | `feat: add Boltzmann selection` | minor (`0.1.0` to `0.2.0`) |
| `feat!:` or a `BREAKING CHANGE:` footer | `feat!: rename Problem.fitness` | minor (`0.1.0` to `0.2.0`) |
| `docs:`, `test:`, `chore:`, `ci:`, `refactor:` | `docs: explain elitism` | no release by themselves |

[release-please](https://github.com/googleapis/release-please) keeps an open pull request that bumps the version
(in `pyproject.toml` and `carmelolg.galib.__version__`) and adds the new entry to `CHANGELOG.md`. Merging that pull
request creates the git tag (`v0.2.0`) and the GitHub release.

## Releases

--8<-- "CHANGELOG.md"

## What 0.1.0 contains

The first version ships these features.

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
