# GALib

A small, extensible library to build genetic algorithms in Python 3.10+.
Fully typed, no dependencies in the core, optional numpy acceleration.

**Documentation: <https://carmelolg.github.io/GALib/>**

| | |
|---|---|
| Distribution name (`pip install`) | `io.github.carmelolg.galib` |
| Import name | `carmelolg.galib` |

`io.github.carmelolg.galib` cannot be an import name: `io` is the standard library module, already loaded when
Python starts. `carmelolg` is a [PEP 420](https://peps.python.org/pep-0420/) namespace package.

## Quick start

```python
import string
from carmelolg.galib import GeneticAlgorithm, Problem, TargetFitness

TARGET = "hello world"
ALPHABET = string.ascii_lowercase + " "


class Phrase(Problem[str]):
    def random_genome(self, rng):  # one random candidate
        return "".join(rng.choice(ALPHABET) for _ in TARGET)

    def random_gene(self, rng):  # enables the default mutation
        return rng.choice(ALPHABET)

    def fitness(self, genome):  # higher is better (set `maximize = False` to minimise)
        return sum(a == b for a, b in zip(genome, TARGET))


result = GeneticAlgorithm(Phrase(), stop=[TargetFitness(len(TARGET))], seed=1).run()
print(result.best.genome)  # hello world
```

## Features

- **Three levels of customisation**: parameters; pluggable operators (any callable, or methods on your
  `Problem`); subclasses of the engine overriding `select`, `crossover`, `mutate`, `reproduce`, `replace`, ...
- **Operators** for sequences (`OnePoint`, `TwoPoint`, `Uniform`, `RandomReset`), permutations
  (`OrderCrossover`, `PartiallyMappedCrossover`, `SwapMutation`, `InversionMutation`) and real values
  (`BlendCrossover`, `GaussianMutation`); selection by `Tournament`, `Roulette`, `Rank`.
- **Monitoring**: per-generation statistics and diversity, callbacks (`PrintProgress`), stop conditions
  (`TargetFitness`, `Stagnation`, `TimeLimit`), plots.
- **Performance**: vectorised `fitness_batch`, numpy array genomes, parallel evaluation with any executor.
- **Reproducible**: each run has its own seeded random generator.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"

python -m pytest --cov                     # tests with branch coverage
mypy                                       # strict type checking
ruff check src tests examples benchmarks   # lint
mkdocs serve                               # documentation preview on http://127.0.0.1:8000
python examples/tsp.py                     # examples: phrase, knapsack, tsp, rastrigin
```

The documentation sources are in `docs/` (MkDocs Material); `.github/workflows/docs.yml` publishes them to
GitHub Pages, `.github/workflows/ci.yml` runs tests, types, lint and the docs build.

## License

Copyright 2026 CARMELO LA GAMBA. Licensed under the [Apache License, Version 2.0](LICENSE).
Redistributions must keep the copyright notices and include the [NOTICE](NOTICE) file.
