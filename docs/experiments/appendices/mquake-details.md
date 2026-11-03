# MQuAKE results and evaluation protocol

[Experiments](../index.md) / [Appendices](index.md)

[MQuAKE](https://github.com/princeton-nlp/MQuAKE) (Zhong et al., EMNLP 2023) tests questions whose
answers change after knowledge edits. Each case edits one to four Wikidata facts and asks a 2-, 3-
or 4-hop question in three paraphrases. Editing Fernando Santos's citizenship, for example, changes
the answer to a question about the head of state of his country. A case counts as correct if any
paraphrase yields the new answer or an alias. Edits are applied in batches of k cases before asking
questions. At `k=all`, weight-editing methods ROME, MEMIT and MEND score below 10% in the published
experiments; retrieval methods must also identify the relevant edits among the whole batch.

kgmem writes `(s, r, old)` to `(s, r, new)` in the graph without updating adapter weights. The graph
contains MQuAKE's own pre-edit chains and the unedited facts traversed by edited chains: 10,151 facts,
12,883 entities and 37 relations across 9,218 MQuAKE-CF cases. GMeLLo, RAE and StruEdit likewise have
access to a base knowledge graph. MeLLo, PokeMQA, DeepEdit and GWalk instead retain only edits and
obtain unedited facts from the LM. Their scores belong to a different, more demanding setting and
are excluded from this first table.

The backbone is frozen SmolLM2-135M. Its 4-hop adapter has 3.9M parameters: the MetaQA initialization,
expanded to four hops at 2.6M, plus a 1.3M relation decoder. Training uses 5,567 MQuAKE-CF cases with
no question or chain shared with any test case, the same pool PokeMQA uses.

| Multi-hop accuracy (%) | CF-3k k=1 | k=100 | k=1000 | k=all | MQuAKE-2002 | MQuAKE-hard | MQuAKE-T k=1 | T k=all |
|---|---|---|---|---|---|---|---|---|
| GMeLLo, GPT-J 6B | 76.3 | 53.4 | 49.5 | 49.0 | | | 86.9 | 81.5 |
| GMeLLo, Vicuna 7B | 71.3 | 46.5 | 42.5 | 41.9 | | | 97.1 | 85.1 |
| GMeLLo, GPT-3.5-Turbo | | | | 66.4 | | | | |
| RAE, Llama-2 7B chat | | | | 69.1* | | | | 66.2 |
| StruEdit, GPT-3.5-Turbo-Instruct | | | | 62.7 | 85.3 | 75.5 | | |
| StruEdit, GPT-4o-mini | | | | 66.5 | 86.3 | 77.3 | | |
| **kgmem**, SmolLM2-135M | **98.5** | **83.0** | **69.7** | **67.0** | **98.6** | **98.1** | **98.8** | **98.8** |
| a perfect graph walker | 100 | 84.1 | 70.4 | 67.5 | 100 | 100 | 100 | 100 |

The comparison rows are paper results, not reruns: GMeLLo Table 1 and text, RAE Table 2 with all edits
in one batch, and StruEdit Tables 2 and 3 at full batch. MQuAKE-2002 and MQuAKE-hard are DeepEdit's
conflict-free CF-3k subsets: 2,002 cases and 429 four-hop cases with four edits each. Both use `k=all`.

kgmem's table rows rank the graph head's entities at the answer position. Decoding on 300 cases per
set at `k=all` scores 97.7 on CF-3k-v2, 98.3 on MQuAKE-2002, 98.7 on MQuAKE-hard and 99.0 on MQuAKE-T,
within sampling error of ranked evaluation. Three relation-decoder training runs differ by at most
0.7 points on each set; CF-3k-v2 all-edit scores are 98.3, 98.4 and 98.5. The shipped adapter is a
fourth run with `--p_plain 0.2`, a fifth of steps on Wikipedia prose. It has the best dev score,
0.937 per question versus 0.932 for the first run. It supplies all shipped-adapter rows here and
improves each test set except MQuAKE-hard, where 98.1 versus 98.8 is a difference of three cases.

## Conflicting edits, masked evaluation and question wording

Original CF-3k edits can contradict facts needed by another case: one question traverses the UK's
head of state while another edits it. Walking the gold relation paths through every applied edit
matches labels on 84.1% of cases at k=100, 70.4% at k=1000 and 67.5% at k=all. kgmem stays within
1.5 points of these gold-path reference scores. A model can score differently by ignoring a conflicting
edit, so these figures describe consistent gold-path traversal, not an unconditional accuracy bound.
The authors' September 2024 (`2024/9`) v2 revision fixes the conflicts and is their recommended
multi-edit file.

StruEdit scores 66.5 on the original full batch and 86.3 on the conflict-free 2,002 cases. The latter
contributes about 58 points over all 3,000 cases; roughly 9 points therefore come from conflicting
cases, whose labels need not agree with all edits applied. kgmem scores 98.6 on the same 2,002-case
subset. These results require the conflict policy to be stated alongside accuracy.

RAE's 69.1 also uses a different policy. Its default `main.py` option `correctConflict` restores
original facts when a test case's gold chain requires them. This is per-case masking derived from
gold labels; kgmem scores 98.6 under masked evaluation. RAE also sets search depth from the gold
chain length. Its rerun on kgmem's all-edit protocol appears below.

MQuAKE-T contains real updates, such as mayors and heads of government. GMeLLo's Vicuna model answers
96.9% at k=1 from the LM alone when given the edited fact (its Table 2), but scores 81–86 with
k=100, 500 or 1,868. kgmem's graph-based score is 98.8 at both k=1 and k=all.

MQuAKE-Remastered (Zhong et al., ICLR 2025) reports omitted hops in all three paraphrases for 22% of
CF-3k cases, 31% of MQuAKE-CF and 12.5% of MQuAKE-T. For example, “what religion is Albert Mohler
associated with?” can stand for the religion of his employer. kgmem trains on such questions too,
and some remaining errors have this ambiguity. Remastered rewrites the questions.

The graph itself is another difference: kgmem uses the dataset's facts, while GMeLLo and RAE query
a larger Wikidata graph. Changes since dataset construction can make an unedited hop disagree
with the original label.

| kgmem, multi-hop accuracy (%) | k=1 | k=all | k=masked | no edits (original answers) |
|---|---|---|---|---|
| MQuAKE-CF-3k-v2 (the authors' fix of the conflicts) | 98.8 | 98.7 | 98.7 | 99.2 |
| MQuAKE-CF-3k (original) | 98.5 | 67.0 | 98.6 | 99.2 |
| MQuAKE-Remastered CF-3k (every question rewritten) | 97.6 | 67.2 | 98.9 | 99.4 |
| MQuAKE-Remastered T | 99.5 | 99.5 | 99.5 | 99.9 |

For v2, the updated MQuAKE paper reports MeLLo/GPT-3.5 at 52.1 / 35.3 / 31.8 / 30.9 for
k=1 / 100 / 1000 / 3000, in the edits-only setting. Remastered's `k=masked` applies all edits but
restores each case's required unedited facts. The gold-path score is then 100%, or 99.7% on
Remastered CF-3k because some revised edit and chain entity ids disagree.

GWalk reports 71.5 on Remastered CF-3k with 3,000 edits and Llama-3.1-8B, and 86.3 on T with all edits
and Qwen2.5-7B. Those also use the edits-only setting, so they are not direct comparisons to kgmem.
The unedited column above asks the same questions without edits and checks their original answers.
Adding Remastered training rewrites (`--rewrites add`) helps its rewritten sets but not the original
MQuAKE sets. Before the relation decoder, this gives 99.2 on masked Remastered CF-3k and 99.4 on T.

## Baselines on the same graph and edits

`experiments/qa/mquake_baselines.py` evaluates two alternatives under kgmem's exact graph, edit batches and scoring.
The first prompts an instruction-tuned LLM with all 37 relation names and 12 examples from the training
pool to extract a topic entity and relation chain. This resembles GMeLLo's KBQA module and StruEdit's
chain representation. The second fine-tunes all 135M SmolLM2 weights as a semantic parser on kgmem's
training questions. Both walk their extracted chain through the edited graph. They link named topic
entities against names present in the question, whereas kgmem learns its entity linker.

| Multi-hop accuracy (%), one protocol | CF-3k-v2 | 2002 | hard | CF-3k k=1 | CF-3k k=all | CF-3k masked | T k=1 | T k=all | R-CF-3k masked | R-T |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct, 12-shot chain | 76.5 | 77.2 | 52.0 | 74.2 | 52.1 | 74.2 | 82.3 | 82.4 | 84.4 | 82.2 |
| Gemma 4 26B-A4B-it, 12-shot chain | 80.2 | 81.6 | 59.7 | 79.0 | 54.8 | 79.0 | 93.6 | 93.7 | 96.1 | 85.4 |
| SmolLM2-135M fine-tuned as a parser | 96.3 | 97.4 | 96.0 | 97.4 | 65.9 | 97.3 | 97.8 | 97.7 | 98.3 | 98.2 |
| kgmem before the relation decoder | 96.8 | 96.9 | 95.1 | 96.7 | 65.7 | 96.9 | 98.7 | 98.8 | 97.3 | 98.1 |
| kgmem with the decoder, no prose in training | 98.3 | 98.2 | 98.8 | 98.2 | 66.7 | 98.3 | 98.7 | 98.7 | 98.6 | 99.2 |
| **kgmem**: SmolLM2-135M frozen, 3.9M-parameter adapter | **98.7** | **98.6** | **98.1** | **98.5** | **67.0** | **98.6** | **98.8** | **98.8** | **98.9** | **99.5** |
| a perfect graph walker | 100 | 100 | 100 | 100 | 67.5 | 100 | 100 | 100 | 99.7 | 100 |

Unspecified k means all edits; R- means Remastered. On this protocol, kgmem exceeds the few-shot
extractors by 2.8 to 46 points with a backbone 50 to 190 times smaller. The closer comparison is the
parser: the same 135M base model and training questions, with all weights trained instead of a 3.9M
block. Parser runs use 3, 10 or 20 epochs, plus 10 at a higher learning rate. The 3-epoch checkpoint
has the best dev score and is shown.

The shipped adapter exceeds that parser by 0.6 points on Remastered CF-3k to 2.4 on CF-3k-v2. It also
exceeds the longer parser runs; the closest are 96.7 on MQuAKE-hard and 98.0 on Remastered CF-3k at
10 epochs. Before the relation decoder, kgmem and the parser are similar. The decoder supplies the
improvement in these runs.

## RAE reruns

`experiments/qa/mquake_rae.py` runs [RAE's public code](https://github.com/sycny/RAE) with GPT-J, used for its reported
69.3 on MQuAKE-CF. RAE performs beam search with LLM relation scores, answer-entropy pruning, worked
MQuAKE-CF examples, answer generation and its own answer check.

The port replaces RAE's Wikidata5m-based graph with MQuAKE's graph and all file edits, uses entity-name
lookup in that graph instead of live Wikidata lookup, and disables `correctConflict`. A further change
retains a terminal path when an entity has no outgoing edges. RAE otherwise keeps only paths reaching
the gold chain length + 2 hops; all paths in this sparser graph died before that depth. RAE's pruning
still chooses the final cut, and search depth still uses gold chain length. Samples are seeded and
random because source files are ordered by hop count; kgmem is scored on the same cases.

| Multi-hop accuracy (%), kgmem's protocol, all edits | CF-3k-v2 (1,000 cases) | MQuAKE-T (600) | MQuAKE-hard (429) |
|---|---|---|---|
| RAE, GPT-J, its chain as retrieved | 50.4 | 43.3 | 22.4 |
| RAE, GPT-J, its chain after its pruning | 33.8 | 3.8 | 26.3 |
| **kgmem** | **98.7** | **98.7** | **98.1** |

Pruning often cuts after one fact in MQuAKE's graph. A chain ending “TV Land has its headquarters
located in the city of New York City” yields “new york city” for a head-of-government question.
On MQuAKE-T, GPT-J can then answer from pre-edit knowledge, such as “donald j trump” after the
United States head-of-government edit to Joe Biden. Under its own Wikidata graph and conflict
restoration, RAE reports 69.3 and 63.9.

A separate reproduction follows RAE's README command,
`main.py --model gpt2 --mode beam --dataset MQuAKE-CF-3k`, with all defaults including `correctConflict`;
its `gpt2` option means GPT-2 XL. It uses a seeded random 300 CF-3k cases. RAE's supplied graph files
require a university sign-in, so this run rebuilds them from public Wikidata5m triples with RAE's
`preprocess/edit_KG.py`. The tested APIs rejected default library User-Agents and rate-limited
one-call-per-entity labels. Labels therefore come from Wikidata5m aliases while entity search stays live.

That run scores 37.7, or 35.0 without pruning, compared with the paper's 62.8 on all CF-3k using GPT-2.
The sample and rebuilt graph differ from the paper, so this does not isolate a cause for the gap.
On the same 300 cases, the GPT-2 XL port to kgmem's protocol scores 14.0, or 35.3 without pruning.
kgmem scores 73.7 with all 3,000 edits, against a gold-path walker at 74.0 on that sample, and 99.7
masked. The [reproduction notes](../../../experiments/qa/baselines/rae/README.md) record the scripts and remaining uncertainty.

## What the block does to the language model

Freezing the LM preserves its weights, but an attached block can still change predictions.
`experiments/qa/lm_preservation.py` measures 300 WikiText-2 test paragraphs with the MQuAKE graph loaded,
using up to 256 tokens per paragraph. It scores both the LM logits and the decoded mixture that
includes the copy head.

| WikiText-2 test, 300 paragraphs: nats/token (perplexity) | LM logits | decoded |
|---|---|---|
| SmolLM2-135M as released | 3.2647 (26.17) | 3.2647 (26.17) |
| **kgmem**, the shipped adapter | **3.2647 (26.17)** | **3.2647 (26.17)** |
| the same without `rel_exists_soft` | 3.2690 (26.29) | 3.2693 (26.29) |
| kgmem with the decoder, no prose in training | 3.5729 (35.6) | 3.6064 (36.8) |
| the same without `rel_exists_soft` | 6.2421 (514) | 6.5089 (671) |

The shipped adapter matches base-model loss to four decimals on these paragraphs. The earlier
decoder adapter does not. `rel_exists` renormalizes over relations for which the linked heads have
edges. Spurious prose matches can therefore turn a small initial probability into a confident walk.
`rel_exists_soft=0.1` scales candidate weights by `m / (m + 0.1)`, where m is relation mass supported
by the heads' edges. The effect is small on confident question walks. Dev accuracy rises from
0.936 to 0.937 on the earlier adapter and from 0.936 to 0.943 on the shipped one.

Prose training supplies WikiText-2 training paragraphs with the graph loaded on a fifth of steps
(`--p_plain 0.2`), minimizing change to the base distribution through `MemoryLM.plain_loss`.
It reaches 3.2690 nats/token without the soft rule and the base-model score with it.
The fully fine-tuned parser goes from 3.285 to 3.422 nats/token, perplexity 26.7 to 30.6, on the same
paragraphs. Both parser and reference base model use plain-transformers scoring. kgmem's wrapper
scores the base model 0.02 lower, so the parser's change is measured against its own scoring path.

## Relations it never trained on

Six of 37 relations are fixed as held out before training: every sixth relation by CF-3k-v2 case
frequency, starting from the fifth commonest. They are continent, creator, chief executive officer,
educated at, chairperson and officeholder. Removing every question whose chain uses one leaves
18,569 training questions and 221 of 300 dev cases. kgmem, the parser and Qwen's worked examples
all use that reduced pool; Qwen's prompt still lists all 37 relation names. `--hold_out_rel`
evaluation applies all edits at once.

| Multi-hop accuracy (%), cases using a held-out relation / the rest | CF-3k-v2 (913 / 2,087 cases) | MQuAKE-T (46 / 1,822) | MQuAKE-hard (207 / 222) |
|---|---|---|---|
| Qwen2.5-7B-Instruct, 12-shot chain | 65.1 / 80.3 | 60.9 / 79.6 | 46.9 / 63.5 |
| SmolLM2-135M fine-tuned as a parser | 3.9 / 96.5 | 2.2 / 97.3 | 1.5 / 96.9 |
| **kgmem** | **35.3 / 98.5** | **80.4 / 98.7** | **21.7 / 98.7** |
| kgmem trained on every relation (the shipped adapter) | 98.8 / 98.7 | 100 / 98.7 | 98.6 / 97.8 |

The parser rarely emits a relation absent from training. kgmem can still link its name to a key
computed by the frozen LM, as in the [strandedcpu experiments](../../explanation/strandedcpu.md). It answers roughly a
third of CF-3k-v2 held-out cases, four in five of the 46 MQuAKE-T cases, and a fifth of MQuAKE-hard,
whose four-hop chains require every relation to be correct. This is well below the 98.8 CF-3k-v2
score when those relations are trained.

On cases without held-out relations, reduced-pool kgmem scores 98.5 versus 98.7 on CF-3k-v2,
98.7 versus 98.7 on T, and 98.7 versus 97.8 on hard. This experiment finds no corresponding loss on
retained relations. Qwen is stronger on held-out relations, while scoring 18 to 35 points below
kgmem on the retained ones.

## Decoder and inference rules

`rel_decoder=256` selects `RelationDecoder`. A GRU starts from the query state, attends to frozen
token states up to that position at each hop, and receives the preceding relation. It scores against
relation keys and predicts whether to stop. This replaces independent per-hop relation and halting
heads, which confused hop order and near-synonyms. With the same losses and inference rules on both,
dev accuracy rises from 0.896 to 0.936; all-edit scores rise from 96.8 to 98.3 on CF-3k-v2,
96.9 to 98.2 on MQuAKE-2002 and 95.1 to 98.8 on hard.

`rel_exists` weights each relation by the head mass with a matching edge, from `ExactScorer.has_rel`,
then renormalizes. Before the decoder, walks often repeated an unavailable relation: “the country of
origin of the sport of KK Crvena Zvezda” could ask for sport again at basketball. Removing the rule
reduces CF-3k-v2 from 95.9 to 93.5, 2002 from 95.8 to 94.5, hard from 94.9 to 92.8 and T from 97.3 to 95.9.

`rel_named=30` multiplies each fully spelled question relation's weight by 31 at every hop.
Combined with `rel_exists`, it favors that relation where the current heads support it.
Before the decoder, the relation-mention mixing weight at hop 2 was only 0.02 and the query linker
confused head of government with head of state. The rule raises T at k=1 from 95.9 to 98.7.
Dev per-question scores are 0.888 at setting 0 and 0.896 at both 30 and 100; 30 was selected on dev.
With the decoder it still raises dev from 0.932 to 0.936.

MQuAKE also uses v9's `carry_rel` and `carry_fresh=0.3`. MetaQA's `carry_fresh=1.0` lets fresh candidates
interrupt the carried answer on 6% of otherwise correct graph questions, such as “Bond Christianity”
for “Bondi Junction”. The v9 settings make decoding agree with ranking on all 450 questions checked.
Prose training (`--p_plain 0.2`) and `rel_exists_soft=0.1` reduce ordinary-text interference as measured
above. All inference settings are recorded in `mquake.MQUAKE_INFERENCE` and the shipped config.

Several tested changes failed. A learned, zero-initialized distance bias anchoring a hop near the
previous hop's attention leaves the dev curve unchanged within noise and was removed. Feeding the
old heads the previous relation gives best dev 0.876 versus 0.889; the attentive decoder works better.
The calibrated `hop_order` rule reduces T at k=1 from 95.9 to 88.6. A floor that stops a walk when
`rel_exists` support is small reduces dev from 0.896 to 0.861 at threshold 0.01. Replacing SmolLM2 with
Gemma 4 26B-A4B gives dev multi-hop 0.85 at step 3,500 versus 0.90 before the decoder was introduced.

## Checkpoint and runtime

The [training recipe](../../how-to/run-benchmarks.md#mquake) selects `checkpoints/mquake.pt` at step
16,000 by dev score and records its inference settings in the config. The 25,000 H100 steps take
about an hour at 0.15 s/step. The full evaluation takes about 30 minutes, mostly k=1 and k=masked,
which rebuild graphs per case. On a Mac, steps take 0.5 s and a 3,000-case all-edit evaluation takes
about two minutes. `--sets` selects cf3k-v2, cf3k, 2002, hard, T, rCF3k or rT, defaulting to files found
by prep. LLM baselines take a few minutes with `--engine vllm` in a separate vLLM environment.
