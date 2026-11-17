# Operators

Operators are the moving parts of the evolution: **selection** picks parents, **crossover** combines them,
**mutation** perturbs the children. All of them live in `carmelolg.galib.operators`.

## Built-in operators

| Kind      | Any sequence                              | Permutations                                       | Real values                          |
|-----------|-------------------------------------------|----------------------------------------------------|--------------------------------------|
| Selection | `Tournament(size)`, `Roulette()`, `Rank()` | same (selection works with any genome)            | same                                 |
| Crossover | `OnePoint()`, `TwoPoint()`, `Uniform(p)`  | `OrderCrossover()`, `PartiallyMappedCrossover()`   | `BlendCrossover(alpha, low, high)`   |
| Mutation  | `RandomReset(random_gene)`                | `SwapMutation()`, `InversionMutation()`            | `GaussianMutation(sigma, low, high)` |

### Selection

- **`Tournament(size=3)`**: each parent is the best of `size` random individuals. The default. Larger sizes mean
  stronger pressure towards the best.
- **`Roulette()`**: probability proportional to fitness (shifted so that negative values work).
- **`Rank()`**: probability proportional to the rank, independent of how large the fitness gaps are.

### Crossover

- **`OnePoint()`** (default), **`TwoPoint()`**: swap the tails, or a middle segment, between random cut points.
- **`Uniform(swap_probability=0.5)`**: swap each gene independently.
- **`OrderCrossover()`** (OX1): keeps a segment of one parent and fills the rest in the order of the other parent.
- **`PartiallyMappedCrossover()`** (PMX): keeps a segment and maps the conflicting genes; preserves more absolute
  positions than OX.
- **`BlendCrossover(alpha=0.5)`** (BLX-α): each child gene is drawn around the two parent genes; `alpha` sets how
  far beyond them it may go.

Permutation crossovers always produce valid permutations, and reject parents that are not permutations of the same
genes.

### Mutation

All mutations use the engine's per-gene `mutation_rate`: the expected number of mutation events per child is
`rate × L`.

- **`RandomReset(random_gene)`**: replaces genes with `random_gene(rng)`. It is what `Problem.mutate` does by
  default, using `Problem.random_gene`.
- **`SwapMutation()`**: swaps genes with random positions.
- **`InversionMutation()`**: reverses a random segment, with probability `min(1, rate × L)`. A good fit for routing:
  it changes only two edges of a tour.
- **`GaussianMutation(sigma, low=None, high=None)`**: adds normal noise and clips to the bounds. `sigma` and the
  bounds may be one value or one value per gene.

## Choosing operators

Pass them to the engine; explicit arguments win over the methods of your `Problem`:

```python
from carmelolg.galib.operators import Rank, TwoPoint

GeneticAlgorithm(problem, selection=Rank(), crossover=TwoPoint())
```

Or define them on the problem itself, so that every engine uses them:

```python
class Tour(Problem[list]):
    def crossover(self, a, b, rng):
        return OrderCrossover()(a, b, rng)

    def mutate(self, genome, rng, rate):
        return InversionMutation()(genome, rng, rate)
```

## Writing your own

An operator is **any callable** with the right signature: a function, a lambda, or an object with `__call__`.
Parameters are positional-only, so name them as you like.

| Kind      | Signature                                       |
|-----------|-------------------------------------------------|
| Selection | `(population, count, rng) -> list[Individual]`  |
| Crossover | `(a, b, rng) -> (child_a, child_b)`             |
| Mutation  | `(genome, rng, rate) -> genome`                 |

```python
def scramble(genome, rng, rate):
    """Shuffle a random segment (a permutation-safe mutation)."""
    genes = list(genome)
    if rng.random() < rate * len(genes):
        i, j = sorted(rng.sample(range(len(genes) + 1), 2))
        segment = genes[i:j]
        rng.shuffle(segment)
        genes[i:j] = segment
    return genes


GeneticAlgorithm(problem, mutation=scramble)
```

Rules for custom operators:

1. **Never modify** the genomes or individuals you receive: return new ones.
2. **Use the `rng`** you receive for all randomness.
3. Selection must return exactly `count` individuals and should prefer a higher `Individual.score`. The score is
   always "higher is better" (it is `-fitness` for minimisation problems), so your operator never needs to know
   the direction.

The built-ins are classes, so you can also subclass them:

```python
from carmelolg.galib.operators import Tournament


class LoggingTournament(Tournament):
    def __call__(self, population, count, rng, /):
        parents = super().__call__(population, count, rng)
        print(f"selected {len(parents)} parents")
        return parents
```
