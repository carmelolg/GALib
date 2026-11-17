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
| `best_genome`   | genome of the best individual                   |
| `evaluations`   | fitness evaluations so far                      |

They are available in `result.history`, and passed to callbacks and stop conditions as the run goes.

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

A measure is any function `(genomes) -> float | None`:

```python
from carmelolg.galib.metrics import gene_spread

GeneticAlgorithm(problem, diversity=gene_spread)
GeneticAlgorithm(problem, diversity=lambda genomes: len(set(genomes)) / len(genomes))
```

## Plotting

With matplotlib installed (`pip install "io.github.carmelolg.galib[plot]"`):

```python
from carmelolg.galib.plot import plot_history

result = GeneticAlgorithm(problem).run()
plot_history(result)  # best and mean fitness, diversity
```
