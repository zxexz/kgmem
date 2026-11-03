# Training objectives

[Documentation](../index.md) / [Explanation](index.md)

Only `MemoryBlock` trains. Each step loads a fresh synthetic world with 8 countries, 16 cities,
5 currencies, 20 people and 6 relations, all named from invented syllables. Since the facts change between
steps, the useful training signal is how to query the graph. The loss combines answer-token NLL under the
copy mixture with labels supplied by the generator: the mention position, its entity, whether the candidate
set contains the gold head, each hop's relation and the hop count. For real text, string matching entity
names supplies mention positions.

## Learning to store variables

Scratchpad batches supervise both pointers and write strength. At a value token the write pointer should
select the variable name and `β` should be 1. In the later question the read pointer should select that name.
`β` should be 0 elsewhere, including the entire read session. The address loss asks a read to recover the
variable's last write and asks different variables to use different addresses.

Read/write agreement alone did not establish storage. An early run put every variable in one of the
64 slots and recalled only the last value written, while agreement still measured 1.0. The slot-based
objective therefore penalizes a read's overlap with other variables' last writes and maximizes mutual
information between writes and slots: individual writes should be sharp, but the batch should use several
slots. `slot_apart` reports the fraction of variable pairs assigned different slots. The
[adapter experiments](../experiments/appendices/adapter-development.md) show the collapsed runs as well as the corrected ones.

The inspector exposed three further failures. A variable's earlier and final writes could use different
slots, leaving both `kel = fox` and `kel = rice` in memory. `addr_same` now supervises their agreement.
The graph path also injected a vector 2–3× the residual's size at every token, including questions and
scratchpad text. `kg_gate_loss` supervises that gate open at graph answers and closed elsewhere.
Finally, `read_norm` amplified nearly empty reads enough to add +2.7 logits to ` apple`.
`read_gate` scales reads below its threshold in proportion to their norm; this inference rule also applies
to older checkpoints.

`experiments/synthetic/train.py` reports `lm_harm`, the increase in next-token loss on question text, excluding answers,
in nats per token. It measures interference with the base model separately from answer accuracy.

## Relation names and ordinary text

The original generator used six fixed relations. A new graph with a different schema was consequently
read through whichever of those six names seemed closest. `RelWorld` (`--p_rel`) samples six relations
from `tasks.REL_POOL`, about 130 nouns with one in eight held out. It asks generic questions such as
“What is the {r} of {h}?” and “What is the {r2} of the {r1} of {h}?”; half the worlds use numeric entity names.

`--rel_mention` adds `RelationMentionLinker`. It matches each token to last-token relation keys, then each
hop attends to the token naming its relation. A learned gate combines that result with the query-position
linker, which is still needed for questions such as “Who leads X?” that do not spell the relation name.
Without training, a relation word's state is nearest its own key 0.95–0.97 of the time on held-out pool names
and strandedcpu programs. Stage 1 reaches 0.97 / 0.98 on held-out relation names at hops 1 / 2 through this
path, compared with 0.61 / 0.70 without it.

Plain-text batches (`--p_plain`) use completions sampled once from the base model and cached by
`train.plain_pool`. They minimize KL(memory off ‖ memory on) over the next-token distribution and supervise
the graph gate closed. `plain_harm` measures loss changes on separate handwritten sentences.
From v3 to v4, this reduced plain-English harm from +5.3 to +0.008 nats/token. Question-text KL
(`--keep_base`) later addressed interference from the copy head on graph questions.

## Cached pretraining

The auxiliary linker and pointer losses need the attach-layer states but not the upper LM layers.
`experiments/synthetic/pretrain_linkers.py` caches a few thousand synthetic batches, trains those components offline, then
`experiments/synthetic/train.py` trains the remaining paths end to end. The cached linker diagnostic ran each epoch in seconds
on CPU. The [Gemma and cached-training report](../experiments/appendices/gemma4.md) gives timings and explains why pretraining
helped the graph path sooner than the scratchpad readout.
