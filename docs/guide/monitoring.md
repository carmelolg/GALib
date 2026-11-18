# Monitoring and stopping

## Statistics

After every generation, generation 0 included, the engine records a
[`GenerationStats`](../reference/core.md#carmelolg.galib.core.stats.GenerationStats):

| Field           | Meaning                                         |
|-----------------|-------------------------------------------------|
| `generation`    | generation number                               |
| `best_fitness`  | best fitness in the current population          |
| `mean_fitness`  | average fitness of the population               |
| `worst_fitness` | worst fitness of the population                 |
| `diversity`     | the diversity measure, or `None`                |
| `best_genome`   | genome of the best individual (see below)       |
| `evaluations`   | fitness evaluations so far                      |

They are available in `result.history`, and passed to callbacks and stop conditions as the run goes.

By default `result.history` keeps the best genome of every generation. With big genomes (images, large arrays) that
adds up: a run of tens of thousands of generations can fill the memory. `history_genome_every=N` keeps the genome
only every N generations (`0` never) and sets `best_genome=None` in the other entries. Statistics are never dropped,
callbacks and stop conditions always receive the full stats, and `result.best` is not affected.

### Reading a run

`run()` returns a [`Result`](../reference/core.md#carmelolg.galib.core.stats.Result): the best individual ever
seen, why the run stopped, and the whole history.

```python
result = GeneticAlgorithm(problem, seed=42).run()

result.best.genome     # the best genome found
result.best.fitness    # its fitness
result.generations     # generations evolved
result.stopped_by      # "max_generations" or the name of the stop condition that fired
result.evaluations     # fitness evaluations spent
result.population      # the final population (Individual objects)
```

`result.history` has one `GenerationStats` per generation, generation 0 included, so any series is a list
comprehension:

```python
history = result.history

best = [s.best_fitness for s in history]
mean = [s.mean_fitness for s in history]
diversity = [s.diversity for s in history]       # may contain None, see "Diversity measures"

# first generation that reached a given fitness, and what it cost
generation = next(s.generation for s in history if s.best_fitness >= 25)
print(generation, history[generation].evaluations)
```

[`examples/monitoring.py`](../examples.md#monitoring) runs all of this on the phrase problem.

## Stop conditions

The run stops after `max_generations`, or as soon as **any** of the `stop` conditions fires:

| Condition                                | Stops when                                                                       |
|------------------------------------------|----------------------------------------------------------------------------------|
| `TargetFitness(target)`                  | the best fitness reaches `target` (`>=`, or `<=` when minimising)                |
| `Stagnation(generations, min_delta=0)`   | the best does not improve by more than `min_delta` for `generations` generations |
| `TimeLimit(seconds)`                     | `seconds` have passed since the run started                                      |

```python
from carmelolg.galib import Stagnation, TargetFitness, TimeLimit

GeneticAlgorithm(problem, stop=[TargetFitness(100), Stagnation(50), TimeLimit(30)])
```

Write your own by subclassing
[`StopCondition`](../reference/termination-callbacks.md#carmelolg.galib.termination.StopCondition);
override `reset()` too if it keeps state between generations:

```python
from carmelolg.galib import StopCondition


class Converged(StopCondition):
    """Stop when the population has almost no diversity left."""

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def should_stop(self, stats, engine):
        return stats.diversity is not None and stats.diversity < self.threshold
```

## Callbacks

A [`Callback`](../reference/termination-callbacks.md#carmelolg.galib.callbacks.Callback) is notified with
`on_start(engine)`, `on_generation(stats, engine)` and `on_end(result, engine)`; override only what you need.

### PrintProgress

```python
PrintProgress(every=1, stream=None, width=60, *, flush=False, end="\n", in_place=False)
```

| Parameter  | Effect                                                                        |
|------------|-------------------------------------------------------------------------------|
| `every`    | print one generation out of `every`                                           |
| `stream`   | where to write (default: standard output)                                     |
| `width`    | characters of the best genome to show (0 hides it)                            |
| `flush`    | flush after each line, useful when output is buffered (pipes, IDE consoles)   |
| `end`      | passed to `print`                                                             |
| `in_place` | rewrite a single line at each generation                                      |

!!! tip "Live progress in an IDE console"
    `in_place=True` writes `"\r" + line`, the technique used by tqdm. Prefer it to `end="\r"`: some consoles,
    PyCharm's among them, erase a line that *ends* with a carriage return, so nothing would be visible.

To change the text, override `format_line(stats)`:

```python
class Compact(PrintProgress):
    def format_line(self, stats):
        return f"{stats.generation}: {stats.best_fitness:g}"
```

### Writing a callback

```python
from carmelolg.galib import Callback


class SaveImprovements(Callback):
    def __init__(self):
        self.improvements = []

    def on_generation(self, stats, engine):
        if not self.improvements or stats.best_fitness > self.improvements[-1][1]:
            self.improvements.append((stats.generation, stats.best_fitness))
```

## Diversity measures

| Measure                  | For                                     | Value                                          |
|--------------------------|-----------------------------------------|------------------------------------------------|
| `positional_diversity`   | strings, bits, permutations (default)   | 0 (all identical) to 1                         |
| `gene_spread`            | real-valued genes                       | mean standard deviation of the genes           |
| `None`                   | anything                                | not measured: saves time on large populations  |

You can also call a measure on any list of genomes, for instance the final population of a run:

```python
from carmelolg.galib.metrics import positional_diversity

genomes = [individual.genome for individual in result.population]
positional_diversity(genomes)    # 0.04: almost every position carries the same gene
len(set(genomes)) / len(genomes) # 0.84: but most genomes still differ by a few genes
```

Low diversity with a flat best fitness is the sign of premature convergence: raise the mutation rate or lower the
selection pressure (a smaller `Tournament`).

A measure is any function `(genomes) -> float | None`:

```python
from carmelolg.galib.metrics import gene_spread

GeneticAlgorithm(problem, diversity=gene_spread)
GeneticAlgorithm(problem, diversity=lambda genomes: len(set(genomes)) / len(genomes))
```

## Plotting

`plot_history` draws the best and mean fitness (top) and the diversity (bottom) of a finished run. It needs
matplotlib: `pip install "carmelolg-galib[plot]"`.

```python
from carmelolg.galib.plot import plot_history

result = GeneticAlgorithm(problem).run()
plot_history(result)
```

It works with any problem, since it only reads `result.history`. By default it opens a window and waits until you
close it. To keep the figure instead, pass `show=False`: the function returns the matplotlib `Figure`.

```python
figure = plot_history(result, show=False)
figure.savefig("history.png", dpi=150)
```

??? note "No window appears"
    `plot_history` needs an interactive matplotlib backend. Check `matplotlib.get_backend()`: `agg` cannot open
    windows (servers, some IDE modes). Save the figure with `show=False`, or pick a backend before plotting,
    for example `matplotlib.use("MacOSX")` on macOS.

    Also check that matplotlib is installed in the **same Python** that runs your script (`sys.executable`).

### Your own plots

For anything else, plot the history yourself. This compares the best fitness of several seeds:

```python
import matplotlib.pyplot as plt

for seed in range(5):
    result = GeneticAlgorithm(problem, seed=seed).run()
    plt.plot([s.best_fitness for s in result.history], label=f"seed {seed}")

plt.xlabel("generation")
plt.ylabel("best fitness")
plt.legend()
plt.show()
```

Use `s.evaluations` on the x axis instead of the generation to compare settings with different population sizes
fairly.
