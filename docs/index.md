---
title: Genetic algorithms for Python
hide:
  - navigation
  - toc
  - footer
---

<div class="galib-hero" markdown>
<div class="galib-hero__text" markdown>

<span class="galib-eyebrow">Genetic algorithms for Python</span>

# Describe the problem. <span class="galib-grad">Let it evolve.</span>

<p class="galib-lead">
GALib runs the evolution while you only say <em>what</em> to optimise. Sensible defaults work out of the box,
and every behaviour can be replaced, from one parameter to the whole replacement strategy.
</p>

[Get started](getting-started.md){ .md-button .md-button--primary }
[How it works](concepts.md){ .md-button }

<p class="galib-install"><span>$</span> pip install carmelolg-galib</p>

<div class="galib-links" markdown>
[User guide](guide/problem.md) · [Examples](examples.md) · [API reference](reference/index.md) · [Changelog](changelog.md)
</div>

</div>
<div class="galib-terminal" role="img" aria-label="Terminal showing a genetic algorithm evolving random text into the phrase Happy Birthday!">
<div class="galib-terminal__bar"><i></i><i></i><i></i><b>python phrase.py</b></div>
<pre><code><span class="l dim" style="--i:0">$ python phrase.py</span><span class="l" style="--i:1"><span class="dim">gen     0 | best</span> <span class="num">3</span> <span class="dim">| mean</span> <span class="num">0.27</span> <span class="dim">| div</span> <span class="num">0.963</span> <span class="dim">|</span> <span class="gene">HaQ JtQgiqhgVJj</span></span><span class="l" style="--i:2"><span class="dim">gen     5 | best</span> <span class="num">7</span> <span class="dim">| mean</span> <span class="num">3.17</span> <span class="dim">| div</span> <span class="num">0.735</span> <span class="dim">|</span> <span class="gene">HaqPi IajthdOB!</span></span><span class="l" style="--i:3"><span class="dim">gen    10 | best</span> <span class="num">9</span> <span class="dim">| mean</span> <span class="num">6.28</span> <span class="dim">| div</span> <span class="num">0.562</span> <span class="dim">|</span> <span class="gene">HapdhvBErthdaka</span></span><span class="l" style="--i:4"><span class="dim">gen    15 | best</span> <span class="num">12</span> <span class="dim">| mean</span> <span class="num">8.88</span> <span class="dim">| div</span> <span class="num">0.398</span> <span class="dim">|</span> <span class="gene">Hapry virthdaG!</span></span><span class="l" style="--i:5"><span class="dim">gen    20 | best</span> <span class="num">14</span> <span class="dim">| mean</span> <span class="num">10.59</span> <span class="dim">| div</span> <span class="num">0.278</span> <span class="dim">|</span> <span class="gene">Happy Birthdak!</span></span><span class="l ok" style="--i:6">Happy Birthday!</span><span class="l" style="--i:7"><span class="dim">21 TargetFitness</span></span></code></pre>
</div>
</div>

<div class="galib-section" markdown>

## Three levels of control

<p>Start with the defaults. Reach deeper only when you need to.</p>

</div>

<div class="grid cards galib-levels" markdown>

-   :material-tune-variant: **1 · Parameters**

    ---

    Tune the search without writing any operator.

    ```python
    GeneticAlgorithm(
        problem,
        population_size=300,
        elite_size=5,
        mutation_rate=0.02,
    )
    ```

    [:octicons-arrow-right-24: The engine](guide/engine.md)

-   :material-puzzle: **2 · Operators**

    ---

    Plug in built-in operators, or any function with the right signature.

    ```python
    GeneticAlgorithm(
        problem,
        selection=Rank(),
        crossover=OrderCrossover(),
        mutation=my_mutation,
    )
    ```

    [:octicons-arrow-right-24: Operators](guide/operators.md)

-   :material-source-branch: **3 · Subclassing**

    ---

    Replace whole phases of the loop, such as the replacement strategy.

    ```python
    class MuPlusLambda(GeneticAlgorithm):
        def replace(self, population, offspring):
            pool = [*population, *offspring]
            pool.sort(key=lambda i: -i.score)
            return pool[: self.population_size]
    ```

    [:octicons-arrow-right-24: Subclassing](guide/engine.md#level-3-subclassing)

</div>

<div class="galib-section" markdown>

## One engine, many problems

<p>Strings, bits, tours, real numbers: pick the genome that fits and the matching operators.</p>

</div>

<div class="grid cards galib-four" markdown>

-   :material-format-text: **Text and symbols**

    Evolve a phrase with the default operators. A first program in a few lines.

    [:octicons-arrow-right-24: Phrase](examples.md#phrase)

-   :material-bag-personal: **Selection problems**

    Knapsack as a list of bits, with a custom stop condition and replacement strategy.

    [:octicons-arrow-right-24: Knapsack](examples.md#knapsack)

-   :material-map-marker-path: **Routes and orders**

    Travelling salesman with permutations, order crossover and inversion mutation.

    [:octicons-arrow-right-24: Salesman](examples.md#travelling-salesman)

-   :material-function-variant: **Numeric optimisation**

    Real-valued numpy genomes with a vectorised fitness over the whole population.

    [:octicons-arrow-right-24: Rastrigin](examples.md#rastrigin-function)

</div>

<div class="galib-section" markdown>

## Under the hood

</div>

<div class="grid cards galib-four" markdown>

-   :material-shield-check: **Typed and tested**

    `mypy --strict`, 97% branch coverage, Python 3.10 to 3.14.

-   :material-package-variant-closed: **No dependencies**

    The core needs nothing. numpy and matplotlib are optional extras.

-   :material-restart: **Reproducible**

    Every run has its own seeded generator: the same seed gives the same run.

-   :material-chart-line: **Observable**

    Statistics, diversity, callbacks and stop conditions for every generation.

</div>

## Install

=== "Core"

    ```bash
    pip install carmelolg-galib
    ```

=== "With numpy"

    ```bash
    pip install "carmelolg-galib[numpy]"
    ```

=== "With plots"

    ```bash
    pip install "carmelolg-galib[plot]"
    ```

!!! note "Distribution name and import name"
    You install `carmelolg-galib` ([PyPI](https://pypi.org/project/carmelolg-galib/)) and import `carmelolg.galib`.

## Thirty seconds example

```python
import string
from carmelolg.galib import GeneticAlgorithm, Problem, TargetFitness

TARGET = "hello world"
ALPHABET = string.ascii_lowercase + " "


class Phrase(Problem[str]):
    def random_genome(self, rng):  # one random candidate solution
        return "".join(rng.choice(ALPHABET) for _ in TARGET)

    def random_gene(self, rng):  # enables the default mutation
        return rng.choice(ALPHABET)

    def fitness(self, genome):  # how good a candidate is: higher is better
        return sum(a == b for a, b in zip(genome, TARGET))


result = GeneticAlgorithm(Phrase(), stop=[TargetFitness(len(TARGET))], seed=1).run()
print(result.best.genome)  # hello world
```

Continue with [Getting started](getting-started.md), or read the [Concepts](concepts.md) if genetic
algorithms are new to you.
