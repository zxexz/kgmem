# Gemma 4 experiments and scaling limits

[Experiments](../index.md) / [Appendices](index.md)

The model's `config.json` makes `AutoModelForCausalLM` load `google/gemma-4-26B-A4B-it` as
`Gemma4ForConditionalGeneration`.
`find_decoder_layers` selects its 30 text layers, excluding the vision tower. Hidden size is 2816.
The pattern repeats five sliding-window layers, window 1024, then one global layer; global indices are
5, 11, 17, 23 and 29. `pick_layer` targets 2/3 depth, index 19, then chooses the nearest global layer,
17. At that attachment backward needs layers 18–29. The MoE has 128 experts, 8 active, beside a dense
MLP. The adapter does not change expert routing, but all 26B weights must remain resident.

## Attachment and training rounds

On one H100 80GB in bf16, zero-shot entity linking with 49 keys gave top-1 / top-4 scores of
0.472 / 0.632 at layer 17, 0.493 / 0.646 at layer 11, and 0.785 / 0.889 at layer 23.
The experiment therefore attached after 23, leaving backward through 24–29. Its first run follows
the v7 recipe with v9 flags, omitting `hop_order` because that threshold was calibrated on SmolLM2.
The [training guide](gemma4-recipes.md#first-run) records the commands.

Stage 1 raises held-out linking from 0.0 to 0.977, or 0.983 on unseen relations. Relation mentions reach
0.99–1.00 and scratchpad address agreement 0.992. Stage 2 occupies 77 of the H100's 80 GB.
After setting `exact_link=0.9`, `python -m experiments.synthetic.eval_adapters runs/gemma4/first.pt --model google/gemma-4-26B-A4B-it --dtype bfloat16`
gives the following results over 3 seeds × 8 worlds.

| Gemma 4 26B-A4B, layer 23 | memory off | first adapter | SmolLM2 v9 |
|---|---|---|---|
| graph 1 / 2 hops (answer ended) | 0.04 / 0.02 | 0.981 / 0.873 | 1.000 / 1.000 |
| unseen relations 1 / 2 hops | 0.02 / 0.00 | 0.917 / 0.862 | 0.989 / 1.000 |
| scratchpad; across sessions, overwritten / untouched | 0.03 | 0.946; 0.975 / 0.975 | 0.987; 1.00 / 1.00 |
| strandedcpu one-hop over every x in 0..199 / two-hop compositions | | 0.993 / 0.50 | 1.000 / 1.00 |
| plain-English harm; 20 assignment-like sentences, mean harm / share hurt | | 0.084; 0.479 / 65% | 0.000; 0.000 / 0% |

Subsequent passes train selected parameters on top of that checkpoint. The same 3 seeds × 8 worlds
produce:

| Gemma 4 26B-A4B | first adapter | + prose negatives (scratchpad only) | + halting (`kg.halt_rel` only) | + `carry_fresh=0.3` | round 6 = `checkpoints/gemma4.pt` |
|---|---|---|---|---|---|
| graph 1 / 2 hops | 0.981 / 0.873 | 0.981 / 0.873 | 0.981 / **0.936** | 0.981 / 0.936 | **0.987 / 0.983** |
| unseen relations 1 / 2 hops | 0.917 / 0.862 | 0.914 / 0.862 | **0.935 / 0.923** | **0.949 / 0.976** | **0.976** / 0.976 |
| scratchpad; plain-English harm | 0.946; 0.084 | **0.966; −0.012** | 0.966; −0.018 | 0.966; −0.018 | **0.974**; 0.012 |
| across sessions: overwritten / untouched | 0.975 / 0.975 | 1.0 / 0.975 | 1.0 / 0.975 | 1.0 / 0.975 | **1.0 / 1.0** |
| 20 assignment-like sentences: mean harm / share hurt | 0.479 / 65% | **0.185 / 35%** | | 0.169 / 30% | **0.073 / 15%** |
| strandedcpu one-hop over every x / two-hop compositions | 0.993 / 0.50 | 0.993 / 0.50 | 0.993 / **0.72** | **1.000 / 0.98** | **1.000 / 1.00** |

The scratchpad's two output paths caused the assignment-like prose damage. None of the 100 texts in
Gemma's generated plain-text pool represented that case. `experiments/synthetic/train.py --plain_assign 100` adds sentences
from `tasks.assignment_prose`, whose templates and words are disjoint from the 20 evaluation sentences.

`experiments/synthetic/calibrate_hop_order.py` finds no usable threshold on these Gemma states: the 10th percentile for
relation words is 0.277, while “ Where” and “ of” reach 0.281. `hop_order` remains off.
On strandedcpu two-hop errors, both lookups were often correct but halting assigned 0.83–0.96 to one
hop, as in the SmolLM2 v8 failure. Training only the named-relation term with
`--halt_rel_aux 1 --train_only kg.halt_rel.` corrects most of these cases.

After that pass, 14 of the remaining 20 two-hop errors retrieve the right answer but spell another
name. “What country is Stomika living in?” retrieves Lolo at 0.98 and emits “ L St el de …”.
After “ L”, carried “olo” has weight 0.499, while a fresh lookup gives “ St”, the first token of
hop-1 entity Stozor, 0.501. Applying v7's `carry_fresh=0.3` without training reduces probe errors from
20 to 6 of 95.

Round 6 trains both halting terms, the copy gate and the end gate together for 1,200 steps with
`carry_fresh=0.3` active. It then trains only the scratchpad for 3,000 steps, using 300 prose negatives
and plain text in 40% of batches. The probe misses 1 of 95. `checkpoints/gemma4.pt` is this 45 MB
checkpoint. Its remaining gaps from SmolLM2 v9 are unseen relations at 0.976, scratchpad at 0.974,
and assignment-like prose harm at about 0.07 nats/token, with 3 of 20 sentences hurt by more than 0.1.

The [round-6 fragment](gemma4-recipes.md#round-6) starts from `runs/gemma4/round5.pt`, round 5 after the prose
and halting passes. Both round-6 outputs receive `exact_link=0.9` before evaluation.
`python -m experiments.synthetic.eval_adapters checkpoints/gemma4.pt --model google/gemma-4-26B-A4B-it --dtype bfloat16`
measures the final column. Intermediate artifacts are required to resume that fragment.

## Check linking before training

`experiments/synthetic/check_linking.py` measures the assumption behind the entity keys without training: does the last token
of a name in context match that name's key? SmolLM2-135M layer 19 gives 0.94–0.97 on 144 held-out mentions,
with about 50 keys per world. The Gemma results above show why it is useful to compare layer 17 with
`--layer 23` or 11 before training.

`experiments/synthetic/check_linking.py --distractors N` adds N unmentioned syllable-generated names. They often share endings with real
mentions, stressing the same kind of confusion that names ending in “… City” or “… FC” can cause.
On SmolLM2-135M layer 19, top-1 and recall within the retained `k_ent=4` candidates are:

| keys in the table | 49 | 1,044 | 10,028 | 30,021 |
|---|---|---|---|---|
| top-1 | 0.94 | 0.71 | 0.48 | 0.42 |
| top-4 recall | 1.00 | 0.85 | 0.65 | 0.55 |

These are zero-shot linker measurements, not end-to-end graph scores. Learned key/query projections,
relations and scorer constraints can reject a wrong head, but the table does not establish that they
recover the loss at 10⁵+ entities. Linking accuracy and lookup cost require separate measurements.

## Model-specific behavior and memory

`MemoryLM` adds `<bos>` for Gemma because its tokenizer does not add it by default;
`add_bos=None` detects this automatically and is tested. The tied vocabulary has 262,144 entries and
a final-logit softcap of 30. The scratchpad readout is mapped through the tied embedding matrix and
added after the cap. These adapters use raw `Q: ... / A:` sequences with the `-it` checkpoint.
Using chat turns would require matching `tokenizer.apply_chat_template` formatting in `tasks.py`
and at generation time.

The bf16 weights occupy 51.6 GB: 25.8B parameters counted on the meta device, including the vision
tower. An 80 GB GPU also holds the fp32 adapter and small-batch activations. The default layer-17
attachment backpropagates through 18–29; the trained layer-23 adapter uses 24–29. Logits and
scratchpad bias move between devices for `device_map="auto"`, but multi-GPU execution was not tested
in these runs.

### Sparse logits

With a 262k vocabulary, computing logits at every position dominates activation storage.
`MemoryLM._logits_at` applies the LM head, softcap and scratchpad bias only to labelled (row, position)
pairs. Models without a separate decoder fall back to HF `logits_to_keep`. Evaluation computes answer
positions; generation and scratchpad write sessions compute only the last position. Tiny Llama and
tiny Gemma 4 tests compare loss with the full-logit path, including Gemma's softcap.

On the synthetic graph batch, bf16 logits shrink from 277 MB for 528 positions to 57 MB for 108 answer
tokens. The scratchpad batch shrinks from 105 MB to 17 MB. Short answers in longer documents permit
larger reductions.

### Quantization in the tested versions

With bitsandbytes 0.50.2 and transformers 5.21, a tiny text-only model built from the real Gemma 4
configuration loads and runs in 4-bit mode, but only `nn.Linear` modules are replaced.
`Gemma4TextExperts` stores each layer's 128 experts in two fused 3-D `nn.Parameter`s,
`gate_up_proj` [128, 1408, 2816] and `down_proj` [128, 2816, 704], which stay bf16.
The tested transformers integration has no path for those parameters.

Experts account for 22.84B of the full model's 25.81B parameters, or 88.5%. Text linear layers account
for 1.66B, embeddings 0.74B and vision/audio 0.57B. Thus `load_in_4bit=True` reduces the estimate from
51.6 GB to roughly 49 GB. Substantial savings would require quantizing experts, for example int8
weight-only with per-row scales and a patched `Gemma4TextExperts.forward` that dequantizes selected
experts. This repository does not implement that path.

## Cached training measurements

The auxiliary losses need only the attach-layer states. `experiments/synthetic/pretrain_linkers.py cache` saves those states
once per batch; the initial layer-17 setup ran only through layer 17. `experiments/synthetic/pretrain_linkers.py train`
constructs the model on the meta device without loading its 52 GB of weights. A test compares cached
auxiliary losses with a full forward pass. End-to-end training then uses `experiments/synthetic/train.py --init runs/synthetic/stage1.pt`
to learn the copy gate, value/output projections and scratchpad readout through the LM head.

An early SmolLM2-135M CPU run cached 330 batches in about 8 minutes. Stage 1 took about 37 s/epoch.
After 4 epochs, held-out scores were 0.99 for entity linking, 0.997 / 1.0 for relations, 1.0 for both
scratchpad pointers and 1.0 for address agreement. After 75 more end-to-end steps:

| | 1-hop | 2-hop | scratchpad |
|---|---|---|---|
| from scratch, 75 steps | 0.82 | 0.42 | 0.02 |
| stage 1, then 75 steps (no read norm) | 0.98 | 0.90 | 0.00 |
| stage 1 (with `read_norm`), then 75 steps | 0.98 | 0.90 | 0.02 |

Without read normalization, stage-1 addresses already produce full-scale reads on the first step.
The zero-initialized readout then overshoots: logit bias reaches 325 versus 39 in a healthy adapter,
and NLL is about 14 nats. `read_norm` applies RMSNorm to the read. It defaults on for new runs;
`--no_read_norm` describes the original shipped SmolLM2 adapter, not later versions.

Normalization reduces scratchpad loss from 7.7 to 2.0 nats over 75 steps, but the scratchpad accuracy
at 75 steps is no better than training from scratch. The addresses are learned; the value/readout
path still needs training. Continuing to 225 steps gives graph scores of 1.00 / 1.00 at step 150 and
0.98 / 0.98 at 225. Scratchpad scores are 0.11 and 0.05, with only 64 questions per evaluation.
These runs support the graph pretraining benefit but do not show a scratchpad accuracy benefit.

Training the readout on cached states is a possible extension. It requires the tied embedding matrix,
262k × 2816 or 1.5 GB in bf16, rather than the full model. A read-to-unembedding loss at the query
position could supervise the gold value token; that extension is not established by the table.

## Larger graphs

Without gradients, the linkers project each key table once per table/weight version and cache it.
One CPU decode step at SmolLM2 width took 3.8 / 22 / 3,421 ms for 10⁴ / 10⁵ / 10⁶ entities before caching,
compared with 0.15 / 0.95 / 7.3 ms afterward. The one-time 10⁶-entity projection took 1.4 s.
Keys stay in the LM dtype and are upcast by chunk: a 10⁶ × 2816 Gemma table needs 5.6 GB in bf16
rather than 11 GB in fp32. Top-k blocks target 256 MB, and each distinct (head, relation) pair reaches
the scorer once per hop regardless of how many positions request it.

Training must still recompute projections with gradients at each step. This is practical for synthetic
worlds of about 50 entities, but the implementation does not solve training against 10⁶-entity tables.
An ANN index such as FAISS could select candidates while retaining gradient computation on their scores;
that path is not implemented. Key encoding also requires a forward per name through the lower layers,
18 layers at the default layer-17 attachment, once per graph and cached to disk. Mention attention
costs O(T²) per sequence; a window or a confidence filter would be needed for much longer contexts.

The [scratchpad CUDA kernels](../../explanation/gpu-kernels.md) replace the torch token loop in supported configurations.
`fla.ops.gated_delta_rule.chunk_gated_delta_rule` is another implementation of the recurrence to
consider for long-context work.

A PyKEEN graph, for example RotatE on Wikidata5M, can be loaded with
`kg_from_pykeen(tf, entity_labels=...)` and queried by `PyKEENScorer` or `ExactScorer`, including one
per hop. Human-readable labels are required for keys. A larger deployment would also need alias
handling, for example several alias keys pointing to one entity id.
`PyKEENScorer` scores all entities for each distinct pair. Its `max_scores` chunks target 256 MB,
allowing 14 pairs per call for Wikidata5M's 4.6M entities. Long prompts mentioning many entities can
therefore be expensive even after deduplication; `ExactScorer` avoids this dense scoring cost.
