# Performance and numpy

In most real problems the **fitness function** dominates the running time. GALib offers three ways to speed
it up.

## 1. Parallel evaluation

Ask the engine for workers: the new genomes of each generation are split into **chunks**, one per worker,
and each chunk is evaluated by `problem.fitness_batch` (which calls `fitness` for every genome by default).

```python
result = GeneticAlgorithm(problem, workers=8).run()                     # process pool, managed by the engine
result = GeneticAlgorithm(problem, workers=8, parallel="thread").run()  # thread pool
result = GeneticAlgorithm(problem, workers=8, chunk_size=4).run()       # smaller chunks, better balance
```

- `parallel="process"` (default) suits CPU-bound fitness functions (simulations, rendering, heavy computations).
  The `Problem` must be picklable, and on macOS and Windows the script needs the usual
  `if __name__ == "__main__":` guard.
- `parallel="thread"` suits fitness functions that wait on I/O, or numpy code that releases the GIL.
- `chunk_size=None` makes one chunk per worker. Use a smaller value when fitness costs are uneven, a larger one
  when sending genomes to the workers is the bottleneck.
- The engine creates the pool lazily and closes it when `run()` ends. After a manual `initialize()` / `step()`
  loop, call `engine.close()` or use the engine in a `with` block.
- To use your own pool, pass `executor=pool` instead of `workers` (you close it; `workers` and `executor` are
  mutually exclusive). Chunking works the same way.
- Results do not depend on the number of workers: the same seed gives the same run.

## 2. Vectorised fitness

Implement `fitness_batch` instead of `fitness` to evaluate the whole population in one numpy expression:

```python
import numpy as np


class Rastrigin(Problem[np.ndarray]):
    maximize = False

    def random_genome(self, rng):
        return np.array([rng.uniform(-5.12, 5.12) for _ in range(10)])

    def fitness_batch(self, genomes):
        x = np.stack(genomes)  # shape (population, 10)
        return 10 * x.shape[1] + (x**2 - 10 * np.cos(2 * np.pi * x)).sum(axis=1)
```

## 3. numpy genomes

Every built-in operator and diversity measure accepts one-dimensional numpy arrays and keeps their `dtype`.
Vectorised paths are used where they help (uniform and blend crossover, diversity), and every numpy random draw
comes from a generator derived from the run's seed, so runs stay reproducible.

Install the extra: `pip install "carmelolg-galib[numpy]"`.

## What to expect

`benchmarks/numpy_vs_python.py` runs the same string-matching problem written in pure Python and with numpy
(numpy genomes plus `fitness_batch`). Measured on one machine (Apple silicon, Python 3.13):

| Genome length × population | Pure Python     | numpy           | Speed-up |
|----------------------------|-----------------|-----------------|----------|
| 100 × 200                  | 1.7 ms/gen      | 0.9 ms/gen      | 1.9×     |
| 500 × 500                  | 14.8 ms/gen     | 5.2 ms/gen      | 2.9×     |
| 2000 × 500                 | 59.4 ms/gen     | 24.4 ms/gen     | 2.4×     |

The gain grows with the cost of the fitness function: here fitness is a cheap comparison, so the rest of the loop
weighs as much as the evaluation. Run the benchmark on your machine:

```bash
python benchmarks/numpy_vs_python.py 500 500 30   # genome length, population, generations
```

## Other tips

- **Mutation is cheap by design**: mutated positions are drawn with geometric jumps, so the cost depends on the
  number of mutations (about one per child by default), not on the genome length.
- **Long runs with big genomes**: `result.history` keeps the best genome of every generation. Set
  `history_genome_every=1000` (or `0`) so the memory does not grow with the number of generations.
- **Skip the diversity measure** with `diversity=None` if you do not need it: it scans the whole population at
  every generation.
- **Elites are never evaluated twice**: `result.evaluations` tells how many fitness calls a run made.
