# Getting started

This page builds a complete program step by step: evolving random strings into a target phrase.

## 1. Describe the problem

A genetic algorithm only needs to know three things about your problem, and you give them by
subclassing [`Problem`](reference/core.md#carmelolg.galib.core.problem.Problem):

```python
import string
from carmelolg.galib import Problem

TARGET = "Happy Birthday!"
ALPHABET = string.ascii_letters + " !"


class Phrase(Problem[str]):  # (1)!
    def random_genome(self, rng):  # (2)!
        return "".join(rng.choice(ALPHABET) for _ in TARGET)

    def random_gene(self, rng):  # (3)!
        return rng.choice(ALPHABET)

    def fitness(self, genome):  # (4)!
        return sum(a == b for a, b in zip(genome, TARGET))
```

1. `Problem[str]`: the genome (one candidate solution) is a string.
2. Build one random candidate. Always use the `rng` you receive, never the global `random` module: this is what
   makes runs reproducible.
3. One random gene (here a character). It enables the default mutation, which replaces random genes.
4. A number saying how good a candidate is: here, how many characters are already right. Higher is better;
   set `maximize = False` on the class for minimisation problems.

## 2. Run the evolution

```python
from carmelolg.galib import GeneticAlgorithm, TargetFitness

engine = GeneticAlgorithm(
    Phrase(),
    population_size=300,
    stop=[TargetFitness(len(TARGET))],  # stop as soon as every character is right
    seed=42,  # same seed, same run
)
result = engine.run()

print(result.best.genome)  # Happy Birthday!
print(result.generations, result.stopped_by)  # 21 TargetFitness
```

[`Result`](reference/core.md#carmelolg.galib.core.stats.Result) contains the best individual ever seen, the
number of generations, why the run stopped and the statistics of every generation (`result.history`).

## 3. Watch it evolve

```python
from carmelolg.galib import PrintProgress

GeneticAlgorithm(
    Phrase(),
    population_size=300,
    stop=[TargetFitness(len(TARGET))],
    callbacks=[PrintProgress(every=5)],
    seed=42,
).run()
```

```text
gen     0 | best          3 | mean     0.2733 | div 0.963 | HaQ JtQgiqhgVJj
gen     5 | best          7 | mean       3.17 | div 0.735 | HaqPi IajthdOB!
gen    10 | best          9 | mean      6.283 | div 0.562 | HapdhvBErthdaka
gen    15 | best         12 | mean      8.883 | div 0.398 | Hapry virthdaG!
gen    20 | best         14 | mean      10.59 | div 0.278 | Happy Birthdak!
```

Each line shows the generation, the best and mean fitness, the [diversity](concepts.md#diversity) of the
population and the best genome. Diversity falls from 0.963 to 0.278 over these 20 generations as the population converges on the answer.
Use `PrintProgress(in_place=True)` to keep a single line that updates in place.

## 4. What the defaults are

| Setting            | Default                                   | Change it with                  |
|--------------------|-------------------------------------------|---------------------------------|
| Population         | 100 individuals                           | `population_size=`              |
| Generations        | at most 1000                              | `max_generations=`, `stop=`     |
| Parent selection   | tournament of 3                           | `selection=`                    |
| Crossover          | one-point, applied with probability 0.9   | `crossover=`, `crossover_rate=` |
| Mutation           | each gene with probability `1 / length`   | `mutation=`, `mutation_rate=`   |
| Replacement        | children replace parents, 2 best kept     | `elite_size=`, subclassing      |
| Diversity measure  | `positional_diversity`                    | `diversity=`                    |

## Next steps

- [Concepts](concepts.md): what selection, crossover, mutation, elitism and diversity mean.
- [Defining a problem](guide/problem.md): genome types, minimisation, vectorised fitness.
- [Operators](guide/operators.md): permutations, real values, custom operators.
- [Examples](examples.md): complete programs.
