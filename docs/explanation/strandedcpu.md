# Computed relations with strandedcpu

[Documentation](../index.md) / [Explanation](index.md)

A graph relation can compute a value instead of reading stored triples. `StrandedCPUScorer` uses the
same `(head, relation) -> tails` interface as the other scorers: it parses the head name as integer x,
executes that relation's program, and links the resulting y to an entity already in the graph.
`strandedcpu_graph()` constructs and materializes such a graph.

strandedcpu is an independent project built on the analytical looped transformer in Mehmet Kerem
Turkcan's [LOOM paper](https://arxiv.org/abs/2604.08816). The ordinary integration uses checked ISA
execution on the interpreter or a batched CUDA kernel. Those execute instructions directly;
`backend="transformer"` runs the analytical transformer's matrix operations as a reference path.
The paper describes the inherited architecture, not strandedcpu's later checks or this integration.

## Execution boundary

The scorer caches each `(head, relation)` result and batches uncached pairs through
`strandedcpu.api.score_options_batch`. `backend="auto"` selects CUDA when available and the interpreter
otherwise. Checked arithmetic returns no tail on faults such as overflow. A valid computed value
outside the loaded entity set also has no tail. A successful calculation establishes the program's
result; it does not establish that the question selected the right program.

Relations without programs can use an `ExactScorer` fallback, allowing one graph to combine facts and
computations. The library includes successor, predecessor, double, half, square, cube, digit sum,
reverse, number of divisors, smallest prime factor and collatz steps. `run_relation()` selects a
single program; direct calls are available through `strandedcpu.api.run_function`.

Install strandedcpu in the environment or set `KGMEM_STRANDEDCPU` to an explicit source checkout.

## Selecting the program

The adapter must still identify the head entity and relation. Exact execution does not correct a
wrong relation choice, an early halt or a name-decoding failure. The
[experimental appendix](../experiments/appendices/strandedcpu-details.md) records those failures,
the inference rules they motivated and runtime measurements. The main
[architecture explanation](architecture.md) describes retrieval and the copy head.

[Use a computed graph](../how-to/use-strandedcpu.md) or look up
[the scorer API](../reference/python-api.md#strandedcpu).
