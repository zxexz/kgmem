# Architecture

[Documentation](../index.md) / [Explanation](index.md)

kgmem adds a trainable memory block and a copy head to a frozen language model. The graph supplies facts;
the block learns to link a question to entities and relations, traverse the graph and emit an answer.
Each training step loads a fresh graph and supervises queries against its facts.
The scratchpad supplies a separate state that the block can read and write across calls.

The target model is `google/gemma-4-26B-A4B-it`. At its width the adapter has 9.3M trainable parameters.
The default attachment is after global-attention layer 17 of 30, roughly two thirds through the decoder;
the trained Gemma checkpoint uses layer 23 because linking measured better there. Early development and
measurements used SmolLM2-135M on one CPU core. Later runs used an H100.

Integration tests cover attachment, training, persistence and agreement between cached decoding and
a full forward pass. Gemma tests use a tiny random model built from the real configuration; the
[Gemma appendix](../experiments/appendices/gemma4.md) separately records real-weight measurements.

```text
                       frozen layers 0..L
                               │ h
                 ┌─────────────┴────────────────────────────────────────────────┐
                 │ MemoryBlock  (x = RMSNorm(h))                                │
                 │                                                              │
                 │  MentionLinker ── per-token: x_s · K_ent  → top-k entities   │
                 │        │          mention attention: query → which s         │
                 │        ▼                                                     │
                 │  hop 1: relation linker(x) ─► scorer(h_ids, r_ids) ─► tails  │
                 │  hop 2: relation linker(x) ─► scorer(tails,  r_ids) ─► tails │
                 │  halting softmax over hops ─► candidate entities + weights   │
                 │        │                                                     │
                 │        ├─► value = Σ w · V(key_e) ─► W_kg (zero init) ──┐    │
                 │        └─► (candidates kept in ctx for the copy head)   │    │
                 │                                                         │    │
                 │  DeltaScratchpad: S ← S + β k (v − kᵀS)ᵀ ─► W_pad ──────┤    │
                 │                                                         │    │
                 │      h + gate · (...) ◄─────────────────────────────────┘    │
                 └─────────────┬────────────────────────────────────────────────┘
                               │
                       frozen layers L+1..N ─► LM logits
                               │
          copy head: p = (1−λ)·p_LM + λ·p_ptr(next token of each candidate's name)
```

## Keys and graph traversal

The graph's embedding coordinates never enter the LM. The key encoder runs each name through the frozen
model in an `Entity: <name>` context and takes its layer-L state. Entity keys use the name's last token;
relation keys use mean pooling by default, or last-token pooling for the relation-mention path.
The scorer accepts `(head_id, relation_id)` pairs and returns `(tail_ids, probs)`. Loading another graph
replaces its names, keys and scorer without changing the adapter's projections.

`ExactScorer` traverses stored triples. It sorts `(h·R + r)` keys and looks them up with `searchsorted`,
using O(#triples) storage and rebuilding its index lazily after edits. `PyKEENScorer` calls a trained
PyKEEN `ERModel` through `score_t`; it can return plausible links absent from the stored triples.
`kg_from_pykeen()` aligns its ids with a `TriplesFactory`.

## Entity and relation linking

Querying entity keys from the answer position reached 60–67% accuracy on training worlds but only about
13% on held-out worlds. Matching each mention token to the keys generalized better: a name's final token
in a question often has a similar state to that token in the key encoder. On SmolLM2-135M, layer 19,
576 held-out questions gave:

| entity linking variant                                   | held-out accuracy |
|----------------------------------------------------------|-------------------|
| answer-position query, mean-pooled keys (trained)        | 0.13              |
| mention token vs mean-pooled keys, zero-shot             | 0.59              |
| mention token vs last-token keys, **zero-shot, no training** | **0.95**      |
| per-token linker + mention attention, trained 8 epochs   | 0.98              |

`MentionLinker` scores each token against the entity keys. It selects top-k candidates without gradients,
then differentiates through their scores. At the query position, attention selects the top-2 earlier
positions and inherits their candidates. A cache retains the token-level links during generation;
`tests/test_integration.py` compares incremental decoding with a full forward pass.

The initial per-hop relation linkers read the query state directly, reaching about 99% on held-out
questions. `RelationMentionLinker` also matches tokens against relation-name keys. For MQuAKE,
`RelationDecoder` replaces the independent per-hop heads with a recurrent decoder that attends to the
question and conditions each relation on the preceding one.

Each hop passes its highest-weight tails to the next hop as heads. A halting softmax combines the answers
from different hop counts. After key matching, traversal costs
O(positions × k_ent × k_rel × max_tails) per hop; graph size affects the chunked top-k search over keys.

## Copying an answer

A retrieved entity's value is projected into the residual stream. That alone did not make the frozen
upper layers spell unfamiliar names: “Orbek” could decode as “Or Sor Quistan”. The copy head instead
proposes tokens from the candidate names and mixes them with the LM distribution through a learned gate λ.

For each name it finds the longest `j` for which `ids[i−j+1..i] == name[:j]`, proposes `name[j]`, and marks
a full match as complete. Candidate carry-over retains the retrieval from the position where a name began:
at lag j, it keeps names whose first j tokens match the emitted tokens and proposes token j. Otherwise each
position halfway through a name would have to recover the same graph answer from its own hidden state.

Completion suppresses the pointer's next-name proposal. Early adapters then relied on the LM to end the
line; v9 also learns an end-of-answer proposal through `end_gate`. The
[adapter experiments](../experiments/appendices/adapter-development.md#v9) separate these spelling and stopping changes from retrieval.

## Scratchpad state

`DeltaScratchpad` keeps an `n_slots × d_val` state `S`. Causal read and write attention select a variable's
name token, then a shared address projection produces its keys. The block reads before writing and computes
values as `W_v h + W_r · read`, so a position can update a value it has just read. A direct readout maps the
read through the frozen LM head; the upper layers need not decode the stored value.

Write strength `β` starts near 0, padding cannot write, and `α` stays at 1 unless `decay=True`.
With `hard_addressing=True`, keys are straight-through one-hot vectors. A one-hot write with gate=1 and β=1
overwrites exactly one row, as `tests/test_units.py::test_delta_rule_is_exact_ram` checks with hand-set
weights. Addresses, values and write strength are learned. A sigmoid β approaches 1 without reaching
it exactly.

The original adapter uses soft slot addresses (`hard_addressing=False`). v7 normalizes keys to unit length
with `l2_keys`, making a write replace the value along its key even when that key spans several slots.
v8 and v9 use `cont_keys`: normalized continuous keys rather than a softmax over slots. Those variants and
their interference measurements are recorded in [the scratchpad experiments](../experiments/appendices/adapter-development.md#v8).

`persistent_state` is a registered buffer. `remember()` and `generate(persist=True)` read and commit it,
and saving the adapter saves that buffer. `scratchpad_batch` trains with two sessions: a write such as
`Set kel to fox. ... Set kel to rice.`, then `Q: What is kel?` with neither text nor KV cache carried over.
Only the scratchpad state connects them, with gradients propagated through it by truncated BPTT.
In a single shared context the frozen LM can answer from its induction heads, leaving little reason for
the scratchpad to learn storage.
