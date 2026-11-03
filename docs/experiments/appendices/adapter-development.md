# Adapter development

[Experiments](../index.md) / [Appendices](index.md)

These experiments track the graph and scratchpad paths separately. Checkpoint compatibility is listed in the
[adapter reference](../../reference/adapters.md); commands are in the [training guide](../../how-to/train-adapters.md).
An adapter belongs to its base model. Gemma 4 requires its own trained block.

## Checkpoint comparison

Held-out names, 3 seeds × 8 worlds; harm is extra NLL in nats/token.

| adapter | graph 1-hop / 2-hop | scratchpad | unseen relation names, 1-hop / 2-hop | zero-shot strandedcpu questions | damage, question text / plain English (nats/token) |
|---|---|---|---|---|---|
| `checkpoints/archive/kgmem_adapter.pt` (original) | 0.88 / 0.90 | 0.70 | n/a | n/a | 5.4 / 1.7 |
| `checkpoints/archive/kgmem_adapter_v2.pt` | 1.00 / 1.00 | 0.79 | n/a | n/a | 3.2 / 5.8 |
| `checkpoints/archive/kgmem_adapter_v3.pt` | 1.00 / 1.00 | 0.84 | n/a | 0.07 | 0.07 / 5.3 |
| `checkpoints/archive/kgmem_adapter_v6.pt` | 0.99 / 0.96 | 0.76 | 0.90 / 0.74 | 0.83 one-hop, 0.06 two-hop | 0.000 / 0.009 |
| `checkpoints/archive/kgmem_adapter_v7.pt` | 0.99 / 0.97 | 0.76 | 0.97 / 0.99 | 0.95 one-hop, 0.78 two-hop | 0.000 / 0.10 |
| `checkpoints/archive/kgmem_adapter_v8.pt` | 1.00 / 0.99 | 0.96 | 0.97 / 0.99 | 0.95 one-hop, 0.80 two-hop | 0.000 / 0.000 |
| **`checkpoints/smollm2-v9.pt`** (recommended) | **1.00 / 1.00** | **0.99** | **0.99 / 1.00** | **1.00 one-hop, 1.00 two-hop** | **0.000 / 0.000** |

## Early CPU experiments: SmolLM2-135M, 1.95M trainable parameters

These first tables predate the name split in `tasks.is_heldout`. Training and evaluation drew from one
pool; replaying the stream found about 65% of evaluation entity names and all but one variable name
in training. Facts were fresh, so these scores test unseen-fact lookup without establishing unseen-name
generalization. Later `evaluate()` calls use held-out names.

Exact match uses teacher-forced argmax at every answer token. On 96 held-out graph questions it agreed
with `generate()` on all cases. Early evaluations contained roughly 100–150 graph and 64–96 scratchpad
questions; uncertainty is about ±0.06 for 64 questions.

| setup | 1-hop | 2-hop | scratchpad (cross-session) |
|---|---|---|---|
| base LM, memory off | 0.02 | 0.07 | 0.00 |
| answer-position entity linking (first version), 100 steps | 0.00 | 0.07 | 0.08 |
| per-token linking + mention attention, 300 steps | 0.82 | 0.59 | 0.02 |
| + scratchpad pointer/β supervision + contrastive hash (825 steps total) | 0.73 | 0.68 | 0.70 |
| same adapter + candidate carry-over at inference (no retraining) | **0.93** | **0.86** | **0.70** |

Without address supervision, scratchpad accuracy stayed near 0.03, against chance around 0.04:
read and write addresses came from their current positions and did not agree. Pointer and β labels
made the pointers accurate, but recall stayed near 0.08 because “kel” in a write and a question still
mapped to different slots. Address consistency raised recall to 0.39 within 75 steps and 0.70 by the end,
at about 3–4 s/step on one CPU core.

The 0.70 checkpoint combines four warm starts (`--init`), with losses changed between runs.
Three runs stopped before their planned learning-rate schedule ended. All used SmolLM2-135M, batch 8,
`--p_pad 0.5`, no `read_norm` (not yet implemented), a fresh optimizer and schedule, and data seed 0.

| stage | from | scratchpad losses on | LR schedule | steps run | scratchpad after |
|---|---|---|---|---|---|
| A | scratch | none | 2e-3, planned for 450 steps | 225 | 0.08 |
| B | A | + where read/write addresses point, when to write | 2e-3, planned for 450 steps | 150 | 0.08 |
| C | B | + address consistency | 2e-3, planned for 375 steps | 150 | 0.53 |
| D | C | same | 1e-3, 300 steps | 300 | **0.70** |

### Reproducing with the current code

`experiments/synthetic/train.py` saves arguments, warm-start ancestry and the last evaluation in `torch.load(path)["meta"]`.
The following attempts used the held-out-name evaluation at seed 12345 over 4 worlds, on a 14-core Mac
at 0.3–0.45 s/step.

| run (SmolLM2-135M, batch 8, `--p_pad 0.5`) | 1-hop | 2-hop | scratchpad |
|---|---|---|---|
| one run, 825 steps, every loss on | 0.98 | 1.00 | 0.25 |
| same, `--seed 1` | 0.98 | 0.95 | 0.22 |
| same, `--no_read_norm` | 0.96 | 1.00 | 0.25 |
| one run, 2500 steps (scratchpad 0.19–0.27 from step 750 on) | 1.00 | 1.00 | 0.22 |
| 225 → 150 → 150 → 300 warm-start chain, every loss on throughout | 0.96 | 0.95 | 0.20 |

Graph scores reproduced in one run on unseen names. Scratchpad scores plateaued near 0.2–0.25 with
all losses enabled from step 0, across the tested lengths, seeds, normalization settings and warm
restarts, including 3,000 steps after stage-1 pretraining. The same 825-step run on the earlier code
scored 0.14.

The stage-1 + 3,000-step adapter revealed a collapsed address map. On a held-out batch, every read
selected its variable's write and β at value tokens was 1.0. Read/write value cosine was 0.95.
Decoding the written value gave the correct answer 100% of the time, but decoding the actual read did
so only 21%. Four variables used 1.02 distinct slots per session, and only one of 64 slots across
32 sessions. Each later write overwrote another variable.

The anti-collision loss compared soft addresses and could decrease overlap by flattening them.
At initialization, 97% of variable pairs already shared a slot. Hard addressing made log-overlap losses
spike and collapsed both pointers. The revised [address objectives](../../explanation/training.md) penalize overlap with
other variables and encourage use of multiple slots. Stage-1 `slot_apart` rises from 0.00 to 0.91 with
read/write agreement 0.84; end-to-end recall rises from 0.25 to 0.79.

| run, evaluated on held-out names over 3 eval seeds × 8 worlds (~380 scratchpad questions) | 1-hop | 2-hop | scratchpad | distinct slots / 4 vars |
|---|---|---|---|---|
| shipped `checkpoints/archive/kgmem_adapter.pt` (four-stage chain above; had seen most eval names) | 0.88 | 0.90 | 0.70 | 2.98 |
| stage 1 + 3,000 steps, before the fix | 1.00 | 1.00 | 0.28 | 1.02 |
| one run from scratch, 825 steps, with the fix | 0.99 | 0.97 | 0.63 | 3.14 |
| stage 1 (4 epochs) + 750 steps, with the fix | 0.99 | 1.00 | 0.79 | 3.56 |
| stage 1 (4 epochs) + 1,500 steps, with the fix = **`checkpoints/archive/kgmem_adapter_v2.pt`** | **1.00** | **1.00** | **0.79** | 3.56 |

The v2 stage-1 initialization was not seeded. `--seed` now controls it, so the recorded checkpoint's
exact scores are not expected on a rerun. Extending this run beyond 750 steps did not improve recall.

<a id="v3"></a>

### v3: overwrite consistency and output gates

`checkpoints/archive/kgmem_adapter_v3.pt` adds same-address supervision for a variable's overwrites, graph-gate supervision
and scaled near-empty reads. It uses a fresh cache because overwrite pairs add a batch field.
The comparison evaluates all three adapters with the read gate, including the older checkpoints.

| on held-out names, 3 eval seeds × 8 worlds | original | v2 | **v3** |
|---|---|---|---|
| 1-hop / 2-hop / scratchpad | 0.88 / 0.90 / 0.70 | 1.00 / 1.00 / 0.79 | **1.00 / 1.00 / 0.84** |
| `lm_harm`: extra loss on the question text, nats/token | 5.43 | 3.16 | **0.066** |
| graph injection on question tokens / at answers (×residual) | 0.61 / 0.91 | 0.51 / 0.78 | **0.00** / 1.48 |
| a variable's overwrites in the same slot | 0.88 | 0.89 | **0.95** |
| different variables in different slots | 0.78 | 0.91 | 0.86 |
| readout bias on graph questions, scratchpad started empty (logits) | 3.41 | 0.02 | 0.19 |
| stage 1: read/write slot agreement / slots apart | n/a | 0.84 / 0.91 | 0.97 / 0.86 |

The [v3 recipe](training-recipes.md#v3) improves question-text preservation, but an ordinary
English paragraph still incurs +1.7 nats/token for the original, +5.8 for v2 and +5.3 for v3. Gate
supervision on question templates did not teach the gate to close on general text. v4 adds base-model
samples with KL to the memory-off distribution. v6 adds the same KL on question text after v5's copy
head and name-continuation positions introduce 0.29 nats/token of harm there.

v3's soft writes also leave partial old values. `kel = fox` and `kel = rice` both reach slot 9, but at
weights 0.46 and 0.33. The readout scores rice +12.1 and fox +11.9; the demo answers “fox”. v2's
different slot layout happens to answer that example correctly. Stronger entropy penalties or an
`addr_scale` schedule were proposed here; v7 instead fixes overwrite scaling with unit-length keys.

### v4–v6: relation names and text preservation

These versions add relation-world and plain-text tasks to the v3 recipe. Except for v4's single-seed
row, evaluation uses held-out entity/relation names over 3 seeds × 8 worlds.

| | 1-hop | 2-hop | scratchpad | unseen relations, 1 / 2 hops | harm: question text / plain English |
|---|---|---|---|---|---|
| v4 (`--p_rel 0.25 --p_plain 0.1`; single-seed eval) | 0.98 | 1.00 | 0.77 | 0.57 / 0.11 | 0.02 / 0.008 |
| v5 (+ `--rel_mention`, `--p_rel 0.3`) | 0.99 | 0.98 | 0.77 | 0.84 / 0.67 | 0.29 / 0.002 |
| **v6** (+ contrast pairs, multi-word names, `--keep_base 1`) = `checkpoints/archive/kgmem_adapter_v6.pt` | 0.99 | 0.96 | 0.76 | **0.90 / 0.74** | **0.000 / 0.009** |

The [v6 recipe](training-recipes.md#v6) took about 35 minutes on a 14-core Mac. The shipped
`checkpoints/archive/kgmem_adapter_v6.pt` sets `relm_raw_cos=30` after training and records it in `meta["inference_changes"]`.
Stage-1 held-out relation linking through mention attention is 0.995 / 1.00.
The [strandedcpu experiments](../../explanation/strandedcpu.md) explain the prior's effect on unfamiliar relation names.

<a id="v7"></a>

### v7: two-hop relation selection and unit-length write keys

`--relm_first` matches the first word of relation names and supervises attention over the whole name.
`--rel_halt` adds named-relation features to halting. Training uses the raw-cosine prior, with half of
relation-world questions requiring two hops. `--l2_keys` normalizes scratchpad keys to unit length.

| | 1-hop | 2-hop | scratchpad | overwritten variable recalled | unseen relations, 1 / 2 hops | harm: question text / plain English |
|---|---|---|---|---|---|---|
| v6 | 0.99 | 0.96 | 0.76 | 0.70 | 0.90 / 0.74 | 0.000 / 0.009 |
| **v7** = `checkpoints/archive/kgmem_adapter_v7.pt` | 0.99 | 0.97 | 0.76 | **1.00** | **0.97 / 0.99** | 0.000 / 0.10 |

The overwrite test uses 20 held-out-name trials of “Set A to v1. Set B to v2. Set A to v3.”,
then asks for each variable in a new session. `python -m experiments.synthetic.eval_adapters <adapter> --only strandedcpu`
includes it. A softmax address has length below 1, so a delta-rule write only partly replaces the value
read along that key. A unit-length key replaces that value completely even when spread over slots.
`tests/test_units.py::test_l2_keys_make_soft_overwrites_exact` checks both cases.

The [v7 recipe](training-recipes.md#v7) took about 40 minutes. The shipped
`checkpoints/archive/kgmem_adapter_v7.pt` adds `relm_raw_cos=60, hop_order=0.4, carry_fresh=0.3` for inference, recorded in
`meta["inference_changes"]`; see [the program-backed evaluations](../../explanation/strandedcpu.md).

<a id="v8"></a>

### v8: continuous keys and gated scratchpad readout

v8 retrains only the scratchpad (`--only_pad`). It freezes the graph path and the shared norm so
v7's inference settings remain calibrated. Graph and strandedcpu score differences come from the
added `carry_share` completion rule, described under [ending a name](strandedcpu-details.md#ending-a-retrieved-name).

v7 maps held-out names to about 4.5 effective slots out of 64. Two random names share a slot 25% of the
time: after “Set kel to fox. Set vom to tea. Set kel to rice.”, the final write can also replace vom.
The pointer selects the right token, but names with similar endings collide: “garul” and “adsal” both
use slot 2. Penalizing slot overlap trades address agreement for separation.

`--cont_keys` uses the normalized projection of a name's state directly, as in DeltaNet. Interference
then depends on key cosine. An InfoNCE loss asks a read to identify its variable's final write among
all final writes in the batch, including other sequences. On held-out names, read/write matching is
0.997 and distinct-variable |cos| stays below 0.5 in 0.983 of pairs, compared with 0.95 / 0.82 for slots.

Continuous-key recall initially generated “rice, vermicelli rice rice” without stopping. v7 had ended
some answers because its read pointer moved to “Q”, whose stored value decoded as a newline.
Teacher-forced scoring of answer tokens alone gave 0.95 to the non-stopping run. `train.evaluate`
therefore also reports `+stop`, requiring the next prediction to end the line.

The scratchpad readout previously had no gate. In “the water is high and brown, and children are told
to stay away”, v7 writes at “brown” and keeps reading it, pushing “ green” and “ white”. The loss on
“ told” rises by 4.6 nats; this sentence accounts for its 0.10 mean harm. `--readout_gate` supervises
the output gate open only at scratchpad answer and line-end positions. `--gate_pad_value` gates the
residual injection too, removing the remaining 0.04 harm.

Training the whole block with continuous keys preserves scratchpad accuracy but changes relation
linking: punctuation maps to “located in” at 1.0, and “ double” maps to “digit sum”. `hop_order` then
misfires, reducing graph one-hop accuracy to 0.39. Freezing that path avoids the regression.

| held-out; 3 seeds × 8 worlds unless noted | v7 | v8 = `checkpoints/archive/kgmem_adapter_v8.pt` |
|---|---|---|
| scratchpad / with the answer ended | 0.76 / 0.76 | **0.96 / 0.96** |
| across sessions, 40 trials: overwritten / untouched variable | 1.00 / 0.80 | 0.975 / **0.975** |
| plain-English harm, the 8 eval sentences (nats/token) | 0.104 | **0.000** |
| 64 base-model completions of new prompts: mean harm / share hurt by > 0.1 | 0.041 / 5% | **0.004 / 2%** |
| 20 handwritten sentences that read like assignments: mean harm / share hurt | 0.564 / 45% | **0.010 / 5%** |
| graph 1 / 2 hops, unseen relations 1 / 2 hops, strandedcpu one-hop / two-hop | 0.99 / 0.97, 0.97 / 0.99, 0.95 / 0.78 | **1.00 / 0.99**, 0.97 / 0.99, 0.95 / **0.80** |
| graph 1 / 2 hops with the answer ended | 0.97 / 0.92 | **0.98 / 0.93** |

`python -m experiments.synthetic.eval_adapters checkpoints/archive/kgmem_adapter_v7.pt checkpoints/archive/kgmem_adapter_v8.pt` produces the rows; `--only` selects
groups. The [v8 recipe](training-recipes.md#v8) reuses `runs/synthetic/cache_v7`. Address training takes about
a minute and scratchpad training about 15. The shipped `checkpoints/archive/kgmem_adapter_v8.pt` retains v7's three
inference settings plus `carry_share=0.2`, recorded in `meta["inference_changes"]`.

For continuous keys the inspector displays one variable per row, since state rows are key dimensions
rather than variable slots. It also shows the readout gate in the timeline. The v8 demo answers all
14 graph questions, including two inference-time edits and a fresh held-out-name world, and both
cross-session questions: kel overwritten from fox to rice, and untouched vom = tea. v7 recalls rice
for vom in that example.

<a id="v9"></a>

### v9: completion, halting and longer scratchpad training

v9 retains v8's entity/relation linkers and shared norm, preserving `hop_order` calibration.
Its H100 training stages take about 15 minutes total. The changes below address measured errors in
retrieval interpretation and output; they do not retrain the linkers.

A completed name must outweigh longer retrieved names sharing its prefix. Without this condition,
“10” at hop 1 with weight 0.34 prematurely ends “the square of the predecessor of 11”, whose answer
“100” has weight 0.66. This extends `carry_share=0.2`.

`--end_gate` adds 17 parameters. After a name completes, the copy head proposes a line end through a
gate on the current state. v8 leaves this to the LM, which after “A: Zencor” assigns the line end only
0.01–0.18. Training the graph value path did not fix it; the
[completion experiments](strandedcpu-details.md#ending-a-retrieved-name) give those attempts. The learned
end proposal raises two-hop accuracy with the answer ended from 0.929 to 1.000.

`--halt_rel_aux` trains the named-relation halting term on its own. v8 learned coefficients around
±2.9 logits for one hop but ±0.22 for two, because the hidden-state term already decided trained names.
For unfamiliar strandedcpu names, “the square of the predecessor of 7” then halts 50/50.
`--halt_rel_same` also tells halting whether the hops select the same relation. Without that feature,
“favourite food” and “assistant coach” can look like two relations when attention lands on different
words. Training never repeats a relation within a question, so matching relation ids support one hop
in this task. These changes raise strandedcpu two-hop from 0.82 to 1.00 and unseen-relation one-hop
from 0.965 to 0.984.

`carry_commit` keeps the copy gate at least as open as it was when a carried name began. Previously,
“the predecessor of 16” could emit “1”: the gate is 0.93 when “15” begins but 0.49 after “1”.
“of 163” could repeat “163”. Training the gate on the existing data made predecessor accuracy worse,
164/199, since training does not include answers sharing digits with the question. The inference rule
raises predecessor accuracy over every x from 177/199 to 199/199.

`--numeric_digits balanced` changes synthetic numeric worlds. The old 0..999 draw makes one-digit
numbers only 1% of entities, while the program-backed graph contains mostly short numbers.
For “the double of 2”, the graph value drives the line end to 0.96, compared with 0.003 in the base
model, producing an empty answer. Balanced sampling takes a third each from 0..9, 10..99 and 100..999;
evaluation worlds stay unchanged.

`exact_link=0.9` reserves 0.9 of linking weight for an exact entity or relation name ending at the
current position. Learned states place similar numbers close together: “the digit sum of 31” links
32, and “the double of 76” links 66. “ label” in “record label” can select “assistant coach” at 0.86.
The prior uses no future tokens, so cached and full-forward decoding still agree. Wide strandedcpu
accuracy rises from 0.985 to 1.000 and unseen-relation accuracy from 0.984 / 0.991 to 0.989 / 1.000.

Finally, 3,000 scratchpad-only steps follow v8's 1,200 CPU steps. `--train_only pad.,…` leaves the graph
path fixed. At 0.19 s/step, recall rises from 0.958 to 0.987 and cross-session recall from 0.975 to
1.000 for both variables. None of the 20 assignment-like evaluation sentences exceeds the harm criterion.

| held-out; 3 seeds × 8 worlds unless noted | v8 | v9 = `checkpoints/smollm2-v9.pt` |
|---|---|---|
| graph 1 / 2 hops, with the answer ended | 0.982 / 0.929 | **1.000 / 1.000** |
| unseen relations 1 / 2 hops | 0.965 / 0.991 | **0.992 / 1.000** |
| strandedcpu one-hop / two-hop (10 per relation / composition) | 0.95 / 0.80 | **1.00 / 1.00** |
| strandedcpu one-hop, every x in 0..199 (911 questions; `--only strandedcpuwide`) | 0.951 | **1.000** |
| scratchpad / with the answer ended | 0.958 / 0.956 | **0.987 / 0.979** |
| across sessions (40 trials): overwritten / untouched; strandedcpu overwrite test (20) | 0.975 / 0.975; 0.95 / 0.95 | **1.00 / 1.00; 1.00 / 1.00** |
| plain-English / question-text harm (nats/token) | 0.000 / 0.000 | 0.000 / 0.000 |
| 64 completions of new prompts, mean harm / share hurt / worst: H100's set; CPU's set | 0.006 / 3% / 0.19; 0.004 / 2% / 0.15 | **0.000 / 0% / 0.02**; 0.003 / 2% / 0.13 |
| 20 assignment-like sentences: mean harm / share hurt / worst | 0.010 / 5% / 0.12 | **0.000 / 0% / 0.00** |

The [v9 recipe](training-recipes.md#v9) uses `--train_only` parameter-name prefixes;
other weights retain their `--init` values. `--hop_order`, `--carry_fresh`, `--carry_share` and
`--carry_commit` are active during training-time evaluation. The shipped `checkpoints/smollm2-v9.pt`
also sets `exact_link=0.9` and `carry_rel=True` in `meta["inference_changes"]`.

MetaQA later adds longest-match linking to `exact_link`. On v9 this introduces one graph failure:
a fresh lookup outweighs the name being spelled, producing “Bely Katorcor” for Belyulyul.
`carry_rel` fixes the scale mismatch by expressing `carry_fresh` as a ratio of fresh to carried mass
rather than a multiplier on raw weights. `python -m experiments.synthetic.eval_adapters checkpoints/archive/kgmem_adapter_v8.pt checkpoints/smollm2-v9.pt`
produces the table on CPU or GPU. The table records the H100 run; the 64 sampled texts differ from the
CPU set because CUDA sampling produces different completions.
