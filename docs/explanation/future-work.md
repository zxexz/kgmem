# Open questions and prior art

[Documentation](../index.md) / [Explanation](index.md)

Gemma 4 has no linear-attention layers, so the scratchpad uses a separate learned head.

Graph writes remain unimplemented. The existing linkers could emit `(h, r, t)` triples for
`ExactScorer`, extending persistence beyond the scratchpad. This would need a policy for deciding
when to write and how to replace or retract facts.

The scorer interface also admits other computations: it accepts an addressed lookup and returns a
value plus candidate names. [strandedcpu](strandedcpu.md) uses that interface to execute programs
without training the computation itself.

## Prior art worth reading

KnowBERT (Peters et al., 2019) injects KB entity embeddings after entity linking.
ReifiedKB (Cohen et al., ICLR 2020) supplies differentiable sparse `follow` over a symbolic KB.
KBLaM (Microsoft, ICLR 2025) trains adapters on synthetic KBs to generalize to unseen ones.

The copy path builds on pointer-generator networks (See et al., 2017). Related writable-memory work
includes fast weight programmers (Schlag, Irie, Schmidhuber, 2021), Gated Delta Networks
(Yang, Kautz, Hatamizadeh, ICLR 2025), Neural Turing Machines / DNC (Graves et al.) and
Memory Layers at Scale (Berges et al., 2024).
