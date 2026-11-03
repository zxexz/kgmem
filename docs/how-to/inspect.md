# Trace an answer in the inspector

[Documentation](../index.md) / [How-to guides](index.md)

With the [core dependencies](setup.md) installed:

```sh
python -m kgmem.inspector
```

Open [127.0.0.1:8765](http://127.0.0.1:8765). The server lists the included SmolLM2 v9 adapter. To compare another set:

```sh
python -m kgmem.inspector --adapters 'runs/*.pt'
```

Choose an example and run it. Select a token to inspect its mention attention, entity and relation scores,
candidate tails, and gates. “Step through” follows the current answer through the diagram.

The next-token chart shows four distributions: a memory-off pass, the upper layers after memory injection,
the scratchpad readout added to those logits, and the final copy mixture. The chart uses a log scale.
Entity-link panels show raw cosines separately from weights renormalized over selected candidates.

Edit an edge to rerun the prompt and compare answers. Remember followed by Ask uses separate sessions;
the inspector keeps a persistent scratchpad for each adapter. Its trace shows read/write addresses, write
strength, overlaps with earlier writes, and state after each token.

The server reads the block through forward hooks and its context object.
