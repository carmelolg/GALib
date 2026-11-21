# Examples

Complete programs live in the
[`examples/`](https://github.com/carmelolg/GALib/tree/main/examples) folder of the repository.
Run them from the project root after `pip install -e ".[numpy,plot]"`. `rastrigin.py --plot` draws the history.

| Example          | Genome                | Shows                                                                          |
|------------------|-----------------------|--------------------------------------------------------------------------------|
| `phrase.py`      | string                | the defaults only; live progress on a single line                              |
| `knapsack.py`    | list of bits          | the three customisation levels: parameters, operators, a subclassed engine     |
| `tsp.py`         | permutation           | minimisation, `OrderCrossover`, `InversionMutation`                            |
| `tsp_delta.py`   | permutation           | `fitness_from_parent`: incremental fitness for mutated children                |
| `rastrigin.py`   | numpy array of floats | vectorised `fitness_batch`, `BlendCrossover`, `GaussianMutation`, `gene_spread` |
| `monitoring.py`  | string                | the `Result`, the history, diversity metrics, `plot_history`                   |

## Phrase

```python
--8<-- "examples/phrase.py"
```

## Knapsack

```python
--8<-- "examples/knapsack.py"
```

## Travelling salesman

```python
--8<-- "examples/tsp.py"
```

## Travelling salesman, incremental fitness

```python
--8<-- "examples/tsp_delta.py"
```

## Rastrigin function

```python
--8<-- "examples/rastrigin.py"
```

## Monitoring

```python
--8<-- "examples/monitoring.py"
```

## Built with GALib

[GAPoly](https://carmelolg.github.io/GAPoly/) ([source](https://github.com/carmelolg/GAPoly)) is a larger program built
on the library: its `Problem` turns a genome of polygons, discs, Voronoi cells, tiles, glyphs or brush strokes into an
image, and the fitness is the distance from a photo. It uses numpy genomes, parallel evaluation (`workers=N`) and
callbacks to save the evolution.
