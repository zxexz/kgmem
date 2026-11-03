# Compute graph relations with strandedcpu

[Documentation](../index.md) / [How-to guides](index.md)

Install [strandedcpu](https://github.com/zxexz/strandedcpu) in the same environment as kgmem.
Install the public package source:

```sh
python -m pip install "git+https://github.com/zxexz/strandedcpu.git"
```

For a separate source checkout, set `KGMEM_STRANDEDCPU` to its root explicitly:

```sh
export KGMEM_STRANDEDCPU=../strandedcpu
python examples/demo.py --strandedcpu
```

The demo asks about successor, double, square, digit sum, smallest prime factor, and a two-hop composition.
It prints each generated answer with its gold value.

<a id="strandedcpu"></a>

## Load a computed graph

Initialize `mlm` as in the [Python guide](use-memory.md), then load the integer entities and program names:

```python
from kgmem.strandedcpu import strandedcpu_graph

kg, scorer = strandedcpu_graph(range(121), ["square", "successor", "digit sum", "smallest prime factor"])
mlm.load_graph(kg, scorer=scorer)
mlm.generate("Q: What is the square of 7?\nA:")
```

`backend="auto"` uses checked CUDA execution when available and falls back to the interpreter.
A fault or a result outside the graph produces no tail. Use `backend="transformer"` with `run_relation()`
to run the analytical transformer reference path.

`strandedcpu_graph()` needs a program for every requested relation. For a graph mixing stored edges with
computed relations, construct `StrandedCPUScorer` with an `ExactScorer` as its `fallback`.
See the [API reference](../reference/python-api.md#strandedcpu) for signatures and cache behavior, and the
[experiment report](../explanation/strandedcpu.md) for measurements.
