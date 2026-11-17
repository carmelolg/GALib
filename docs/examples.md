# Examples

Complete programs live in the
[`examples/`](https://github.com/carmelolg/GALib/tree/main/examples) folder of the repository.
Run them from the project root after `pip install -e ".[numpy]"`.

| Example          | Genome                | Shows                                                                          |
|------------------|-----------------------|--------------------------------------------------------------------------------|
| `phrase.py`      | string                | the defaults only; live progress on a single line                              |
| `knapsack.py`    | list of bits          | the three customisation levels: parameters, operators, a subclassed engine     |
| `tsp.py`         | permutation           | minimisation, `OrderCrossover`, `InversionMutation`                            |
| `rastrigin.py`   | numpy array of floats | vectorised `fitness_batch`, `BlendCrossover`, `GaussianMutation`, `gene_spread` |

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

## Rastrigin function

```python
--8<-- "examples/rastrigin.py"
```
