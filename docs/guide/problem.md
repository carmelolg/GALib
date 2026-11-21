# Defining a problem

A [`Problem`](../reference/core.md#carmelolg.galib.core.problem.Problem) describes *what* is optimised. The engine
never looks inside your genomes: it only asks the problem to create them, evaluate them and, unless you provide
operators, combine and mutate them.

## Required methods

| Method                          | Purpose                                                              |
|---------------------------------|----------------------------------------------------------------------|
| `random_genome(rng)`            | Build one random candidate solution.                                 |
| `fitness(genome)`               | How good one candidate is: a finite number.                          |
| *or* `fitness_batch(genomes)`   | The fitness of many candidates at once, e.g. vectorised with numpy.  |

`fitness` and `fitness_batch` default to each other, so implement whichever is natural. A problem that implements
neither is rejected when the engine is created.

!!! warning "Always use the `rng` you receive"
    Draw all randomness from the `random.Random` passed to your methods. The global `random` module would make
    runs impossible to reproduce with a seed.

## Optional overrides

| Member                       | Default                                  | Override when                                   |
|------------------------------|------------------------------------------|-------------------------------------------------|
| `maximize`                   | `True`                                   | lower fitness is better: set `maximize = False` |
| `random_gene(rng)`           | raises `NotImplementedError`             | you want the default mutation (random reset)    |
| `crossover(a, b, rng)`       | one-point crossover                      | your genome needs a specific crossover          |
| `mutate(genome, rng, rate)`  | reset each gene with probability `rate`  | your genome needs a specific mutation           |
| `fitness_from_parent(genome, parent)` | calls `fitness(genome)`         | a small mutation lets you derive the child's fitness from its parent's ([incremental fitness](performance.md#4-incremental-fitness)) |

Operators passed to the [engine](engine.md) take precedence over these methods; see [Operators](operators.md).

## Genome types

The genome can be anything; the built-in operators handle the common cases:

=== "Strings"

    ```python
    class Phrase(Problem[str]):
        def random_genome(self, rng):
            return "".join(rng.choice(ALPHABET) for _ in TARGET)

        def random_gene(self, rng):
            return rng.choice(ALPHABET)

        def fitness(self, genome):
            return sum(a == b for a, b in zip(genome, TARGET))
    ```

=== "Bit lists"

    ```python
    class Knapsack(Problem[list]):
        def random_genome(self, rng):
            return [rng.randint(0, 1) for _ in ITEMS]

        def random_gene(self, rng):
            return rng.randint(0, 1)

        def fitness(self, genome):
            weight = sum(w for (w, _), take in zip(ITEMS, genome) if take)
            value = sum(v for (_, v), take in zip(ITEMS, genome) if take)
            return value if weight <= CAPACITY else 0
    ```

=== "Permutations"

    ```python
    from carmelolg.galib.operators import InversionMutation, OrderCrossover

    class Tour(Problem[list]):
        maximize = False  # shorter tours are better

        def random_genome(self, rng):
            tour = list(range(len(CITIES)))
            rng.shuffle(tour)
            return tour

        def fitness(self, tour):
            return sum(math.dist(CITIES[tour[k - 1]], CITIES[tour[k]]) for k in range(len(tour)))

    # generic crossovers would duplicate cities: use permutation operators
    GeneticAlgorithm(Tour(), crossover=OrderCrossover(), mutation=InversionMutation())
    ```

=== "Real values (numpy)"

    ```python
    import numpy as np
    from carmelolg.galib.metrics import gene_spread
    from carmelolg.galib.operators import BlendCrossover, GaussianMutation

    class Sphere(Problem[np.ndarray]):
        maximize = False

        def random_genome(self, rng):
            return np.array([rng.uniform(-5, 5) for _ in range(10)])

        def fitness_batch(self, genomes):
            return (np.stack(genomes) ** 2).sum(axis=1)  # the whole population at once

    GeneticAlgorithm(
        Sphere(),
        crossover=BlendCrossover(low=-5, high=5),
        mutation=GaussianMutation(sigma=0.3, low=-5, high=5),
        diversity=gene_spread,
    )
    ```

Built-in operators support `str`, `bytes`, `tuple`, `list` and one-dimensional numpy arrays, and always return the
same type they receive. For any other genome type (a tree, a graph, an object) override `crossover` and `mutate`,
or pass your own operators.

## Fitness rules

- It must return a **finite** number: `NaN` or infinity raise `ValueError` with the offending genome.
- It must depend **only on the genome**, so that elites never need to be evaluated again.
- Penalties are a simple way to handle constraints: in the knapsack above, an overweight bag is worth 0.
