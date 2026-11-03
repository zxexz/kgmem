# Python API

[Documentation](../index.md) / [Reference](index.md)

Import `KnowledgeGraph`, `ExactScorer`, `PyKEENScorer`, `kg_from_pykeen`, `MemConfig`, `MemoryBlock`,
`MemoryLM`, and `Batch` from `kgmem`. Program-backed graph APIs live in `kgmem.strandedcpu`.

## Graphs and scorers

Source: [`kgmem/graph.py`](../../kgmem/graph.py).

```python
KnowledgeGraph(entities: list[str], relations: list[str],
               triples: list[tuple[int, int, int]] = [])
```

Each graph gets its own `triples` list. Entity and relation IDs are indices into the name lists;
`ent_index` and `rel_index` map names back to those IDs.

| Method | Result |
| --- | --- |
| `add(h: str, r: str, t: str)` | Append an edge between existing names; increment the graph version. |
| `set(h: str, r: str, t: str)` | Replace every edge for `(h, r)` with the specified tail; increment the graph version. |
| `tails(h: str, r: str) -> list[str]` | Return stored tail names for the pair. |

`add()` and `set()` require existing names and increment the graph version. `ExactScorer` checks that
version before a lookup. Mutating `triples` directly bypasses invalidation. For a different vocabulary,
construct a new graph and call `load_graph()` to rebuild the indexes and encoded keys.

### Scorer interface

`Scorer` is defined in `kgmem.graph`:

```python
scorer(h: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

| Value | Shape and meaning |
| --- | --- |
| `h`, `r` | Long tensors `[P]` containing head and relation IDs. |
| Returned tails | Long tensor `[P, M]` of entity IDs. |
| Returned probabilities | Float tensor `[P, M]`; each row sums to at most one. Zero weight marks unused tail slots. |

IDs must refer to the loaded graph's entity and relation order. Built-in scorers return their outputs on `h.device`.

```python
ExactScorer(kg, max_tails=4, device="cpu")
ExactScorer.has_rel() -> torch.Tensor | None

PyKEENScorer(model, max_tails=4, temperature=1.0, chunk=None,
             min_prob=0.0, max_scores=1 << 26)

kg_from_pykeen(tf, entity_labels=None, relation_labels=None) -> KnowledgeGraph
```

`ExactScorer` holds the graph by reference and rebuilds after a version change. It keeps at most
`max_tails` tails per pair and divides mass uniformly among them. Missing pairs have zero probability.
Its device must match the lookup tensors. `has_rel()` returns `[E, R]` booleans, or `None` when
`E * R > 2**28`.

`PyKEENScorer` sets the model to evaluation mode and scores every entity. It returns the top `max_tails`
with probabilities from the full softmax, zeroing those below `min_prob`; the retained mass is not
renormalized. `chunk` overrides the row limit calculated from `max_scores` and `model.num_entities`.

`kg_from_pykeen()` orders names by the factory's integer IDs and copies `mapped_triples`. Optional label dictionaries replace displayed names while preserving that ID order.

## MemoryLM

Source: [`kgmem/memlm.py`](../../kgmem/memlm.py).

```python
MemoryLM(model, tokenizer, cfg=MemConfig(), layer_idx=None,
         name_prefix="Entity:", add_bos=None, device=None)
```

`MemoryLM` freezes the base parameters and wraps one decoder layer with a `MemoryBlock`.
An explicit `layer_idx` is taken modulo the decoder depth. The default selects a layer near two-thirds
depth, preferring a nearby full-attention layer when layer types are available.
The block defaults to that layer's device.

Attributes: `lm`, `tok`, `cfg`, `block`, `ctx`, `device`, and `layer_idx`.
`MemoryLM` does not inherit `nn.Module`; use `block.eval()` and `block.parameters()` for the trainable module.

### Graph and key methods

| Signature | Behavior |
| --- | --- |
| `tokenize_name(name: str) -> list[int]` | Tokenize `" " + name` without special tokens. |
| `encode_names(names, bs=256, pool=None) -> Tensor` | Encode names through the attach layer, with memory disabled; use `cfg.key_pool` when `pool` is omitted. `"first"` and `"last"` select one token; other values take the mean over name tokens. |
| `load_graph(kg, scorer=None, max_name_tokens=12)` | Replace `ctx.graph` with encoded entity/relation keys, copy-name tokens, aliases, and the scorer. Default scorer: `ExactScorer(kg, cfg.max_tails, device)`. |
| `unload_graph()` | Set `ctx.graph` to `None`. |
| `calibrate(texts: list[str], skip=1)` | Set per-dimension input mean and standard deviation from attach-layer states, excluding the first `skip` tokens of each text. Encode graph keys after calibration. |
| `hidden_at_layer(ids, attn=None) -> Tensor` | Return attach-layer hidden states with memory disabled; stop before later layers. |

Copy names and entity aliases are truncated to `max_name_tokens`; key encoding still uses the full names.
Relation aliases use the full tokenized relation. Ambiguous aliases map to `-1`.
Loading or unloading a graph leaves the persistent scratchpad intact.

### Forward calls and state

```python
forward(ids, attn=None, aux_idx=None, state=None, aux_mention=None, keep=None)
block_forward(h, attn=None, aux_idx=None, aux_mention=None, state=None)
write_session(ids, attn=None, state=None)
generate(prompt: str, max_new_tokens=24, stop="\n", persist=False,
         use_memory=True) -> str
remember(text: str)
reset_memory()
```

| Call | State and return value |
| --- | --- |
| `forward()` | Clear per-call caches, use the supplied scratchpad state, and return LM logits. `keep` is passed as Hugging Face `logits_to_keep`. Graph copy probabilities are combined separately during loss and generation. |
| `block_forward()` | Run the block on cached attach-layer states; return modified hidden states. The updated scratchpad is in `ctx.state`. |
| `write_session()` | Run through the attach layer and block, then stop; return scratchpad state without running later layers or the LM head. |
| `generate()` | Greedy decode with a per-call KV cache. Return generated text before the first `stop` substring, if supplied. Scratchpad state carries between decoding steps. |
| `generate(persist=True)` | Start from a clone of `block.persistent_state` and commit the resulting state back after decoding. |
| `remember()` | Update persistent scratchpad state from text using `write_session()`; retain no KV cache. Requires the scratchpad to be enabled. |
| `reset_memory()` | Zero `block.persistent_state`. Requires the scratchpad to be enabled. |

Scratchpad state is `[B, cfg.n_slots, cfg.d_val]`. `forward()`, `block_forward()`, and `write_session()`
start from zero when `state=None` and do not commit to `persistent_state`.
`generate()` reads and commits persistent state only with `persist=True`.
`use_memory=False` disables both the block and copy mixture during generation.

### Training, evaluation, and inspection

| Signature | Result |
| --- | --- |
| `trainable_parameters()` | List of trainable block parameters. |
| `loss(b, aux_weight=1.0, sparse_logits=True, keep_base=0.0, stop_weight=1.0)` | `(loss, logs)`; answer-token NLL plus auxiliary losses, or `plain_loss()` when `b.plain` is true. |
| `stage1_loss(b, h, prefix_h=None, aux_weight=1.0)` | `(loss, logs)` for auxiliary training from cached hidden states. |
| `aux_loss(b, aux_weight=1.0)` | `(loss, logs)` using supervision and the most recent forward context. |
| `answer_logprobs(b, sparse_logits=True, stop_weight=1.0)` | Flat labeled next-token log probabilities and per-token weights. |
| `exact_match(b, stops=False)` | One `(kind, correct)` tuple per answer span; with `stops=True`, append a Boolean for a correct answer followed by a newline. Teacher-forced token evaluation. |
| `read(h, ids, pos=None, k=3, window=1024) -> dict` | Graph retrieval from cached states shaped `[1, T, d]` and tokens `[1, T]`; returns `hops`, `halt`, and `candidates`. Requires a loaded, enabled graph head. |

`Batch` requires `ids`, `labels`, and `attn` shaped `[B, T]`, plus `spans` containing
`(batch_index, start, end, kind)` tuples with exclusive `end`. Label `-100` excludes a position from loss.

Optional fields:

- Graph targets: `aux_idx`, `aux_ent`, `aux_rel`, `aux_hops`, `aux_mention`, `aux_rel_mention`, `aux_rel_span_start`.
- Scratchpad targets: `ptr_w`, `ptr_r`, `beta_tgt`, `addr_pair`, `addr_neg`, `addr_same`.
- `prefix: Batch`: prior session whose scratchpad state carries into this batch. Its KV cache does not carry over.
- `plain=False`: switch to the plain-text preservation loss.

### Persistence

```python
save(path: str, meta: dict | None = None)
load(path: str, strict: bool = True)
```

`save()` writes a PyTorch checkpoint with these keys:

| Key | Contents |
| --- | --- |
| `cfg` | `cfg.__dict__`, including all configuration fields below. |
| `layer_idx` | Selected decoder layer index. |
| `block` | Block state dictionary: parameters, `in_shift`/`in_scale` calibration buffers, and `persistent_state` when the scratchpad is enabled. |
| `meta` | Supplied metadata, or `None`; use plain values for weights-only loading. |

The checkpoint does not include the base model, tokenizer, graph/scorer, encoded keys, KV cache, or
transient `ctx.state`. Construct the matching model and wrapper before calling `load()`.
`load()` checks `layer_idx` and restores block tensors; it does not apply `cfg` or `meta`.

With `strict=False`, unknown keys and shape mismatches are filtered out. Missing `in_shift` and `in_scale`
retain their current buffers. The return value is PyTorch's `load_state_dict()` result.

## MemConfig and MemoryBlock

Source: [`kgmem/modules.py`](../../kgmem/modules.py).

`MemConfig` is a dataclass. `save()` includes all of these fields in `cfg`:

| Fields | Defaults | Use |
| --- | --- | --- |
| `d_q`, `d_val` | `256`, `256` | Linker and memory-value dimensions. |
| `k_ent`, `k_rel`, `k_cand`, `max_tails`, `hops` | `4`, `3`, `8`, `4`, `2` | Entity/relation candidates, retained candidates, scorer tail count, and graph hops. |
| `use_kg`, `use_scratchpad` | `True`, `True` | Construct graph head and scratchpad. |
| `n_slots`, `hard_addressing`, `decay` | `64`, `False`, `False` | Scratchpad address dimension, one-hot addressing, and learned decay. |
| `key_pool`, `rel_key_pool` | `"last"`, `"mean"` | Entity and relation key pooling. |
| `k_mention`, `d_att` | `2`, `64` | Mention candidates and pointer-attention dimension. |
| `read_norm`, `read_gate` | `False`, `4.0` | Scratchpad read normalization and low-norm scaling threshold. |
| `kg_gate_loss` | `1.0` | Graph-gate supervision weight. |
| `rel_mention`, `relm_raw_cos`, `relm_first`, `rel_halt` | `False`, `0.0`, `False`, `False` | Relation mention linking, raw-cosine contribution, first-token keys, and mention-aware halting. |
| `l2_keys`, `cont_keys` | `False`, `False` | Unit-normalized scratchpad keys and continuous addresses. |
| `readout_gate`, `gate_pad_value` | `False`, `False` | Scratchpad logit-readout gate and gating of its hidden-state contribution. |
| `hop_order` | `0.0` | Relation-order heuristic threshold; uses raw relation cosine. |
| `carry_fresh`, `carry_rel`, `carry_share`, `carry_top` | `1.0`, `False`, `0.0`, `0.0` | Copy-name candidate carry behavior; positive `carry_top` takes precedence over `carry_share`. |
| `halt_rel_aux`, `halt_rel_same` | `0.0`, `False` | Named-relation halting supervision and same-relation feature. |
| `carry_commit` | `False` | Keep the copy gate open while carrying a name. |
| `exact_link`, `merge_cands` | `0.0`, `False` | Exact-name alias prior and merging candidates reached by multiple paths. |
| `rel_exists`, `rel_exists_soft` | `False`, `0.0` | Weight graph relations/candidates by edge availability from `has_rel()`. |
| `rel_decoder` | `0` | Positive width enables the relation decoder. |
| `rel_named`, `exclude_head`, `end_gate` | `0.0`, `False`, `False` | Named-relation boost, exclusion of the linked head from later hops, and learned copy line-end gate. |

`MemoryBlock(d_model: int, cfg: MemConfig)` is an `nn.Module`. Its `forward(h, ctx)` returns modified hidden states and updates the context's graph candidates, scratchpad state, and logit-readout direction. `standardize(h)` applies saved per-dimension calibration. The block's output projections start at zero; the separate copy gate is initialized with bias `-2.0`.

## strandedcpu

Source: [`kgmem/strandedcpu.py`](../../kgmem/strandedcpu.py).

```python
strandedcpu_graph(values, relations, programs=None, backend="auto", N=16,
           max_tails=4, checked=True) -> tuple[KnowledgeGraph, StrandedCPUScorer]

StrandedCPUScorer(kg, programs=None, backend="auto", N=16, max_tails=4,
           fallback=None, checked=True)
```

`strandedcpu_graph()` names entities with `str(value)` and requires a program for every relation.
It computes all entity/relation pairs and adds successful in-graph tails to `kg.triples`.
`programs=None` or an empty dictionary selects `PROGRAMS`.

`compute(h, r)` returns a cached result dictionary or `None`. `compute_many(pairs)` fills missing entries;
`cache`, `runs`, and `hits` are available for inspection. Graph edits do not invalidate this cache.
Relations without programs use `fallback` when supplied.

A non-integer head, failed domain check, checked execution fault, or out-of-graph result yields no tail.
Successful tails have probability `1.0` in the first slot.

Checked execution is used for `auto`, `interpreter`, and `cuda` when requested and available.
Other modes use `run_relation()` with the library's domain predicates. `DOMAINS` and `DOMAINS2` describe
the bundled programs at `N=16`; selecting another word size does not change those predicates.
With `checked=False`, `auto` selects the interpreter; unchecked execution supports only `interpreter`
and `transformer` and can wrap or raise on faults. CUDA requires checked execution.

| Signature | Return |
| --- | --- |
| `source(body: str, inputs=("x",)) -> str` | C program with input globals, output global `y`, and local declarations moved to the start of `main`. |
| `coerce(name: str) -> int \| None` | Parse a signed integer entity name after trimming whitespace. |
| `name_of(value: int) -> str` | `str(value)`. |
| `run_program(body, inputs, backend="interpreter", N=16, trace=False)` | Dictionary containing `y`, `steps`, `seconds`, `backend`, and optional `trace`. |
| `run_relation(body, x, backend="interpreter", N=16, trace=False)` | `run_program()` with the single input `x`. |
| `run_checked(body, inputs, N=16, backend="interpreter")` | Dictionary containing `y`, `steps`, `seconds`, `backend`, `fault`, and `detail`; execution faults set `y=None`. |
| `run_checked_batch(jobs, N=16, backend="auto")` | List of checked result dictionaries for `(body, inputs)` pairs. Reported time divides batch wall time equally among jobs. |

`PROGRAMS` and `DOMAINS` cover one-input graph relations; `PROGRAMS2` and `DOMAINS2` cover two-input
programs for direct callers. The loader imports `strandedcpu.api` lazily. Set `KGMEM_STRANDEDCPU` for a
source checkout override, or install the package in the current environment.
