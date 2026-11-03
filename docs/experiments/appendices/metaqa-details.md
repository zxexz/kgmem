# MetaQA results and evaluation protocol

[Experiments](../index.md) / [Appendices](index.md)

[MetaQA](https://github.com/yuyuz/MetaQA) (Zhang et al., AAAI 2018) asks 1-, 2- and 3-hop questions over
WikiMovies: 43,234 entities, 9 relations (18 with inverses) and 134,741 triples. Its test sets contain
9,947 / 14,872 / 14,274 questions. Hits@1 accepts any gold answer. This experiment uses frozen
SmolLM2-135M and a 3-hop adapter with 2.3M parameters.

The headline results are in the [main report](../metaqa.md).

The comparison rows are copied from their papers; they are not reruns on this implementation.
kgmem ranks below TransferNet on the official labels. Its retrieved-entity and decoded-answer rows
agree within sampling uncertainty, about ±0.3–0.5 points for 1,000 questions.

## Supervision and ambiguous labels

Training supplies a relation pseudo-label at each hop. For each question template, preparation finds
a relation walk whose result exactly equals the training answer set. These labels use the training
answers and KB; Subgraph Retrieval likewise derives supervision from shortest paths. Removing them
reduces retrieved accuracy to 94.8 / 95.6 / 57.1. kgmem truncates each hop to `k_cand` candidates, which
cuts gradients to discarded earlier relation choices. TransferNet's soft transfer propagates through
every hop.

The KB merges films with the same title, including “A Star Is Born” from 1937 and 1976. For 5.4% of
one-hop test questions, the gold answers include only a subset of the matching KB facts. Simple order
heuristics do not identify that subset: it includes the first-listed fact 51% of the time, the last
46%, and the lowest entity id 50%. Treating these as roughly even choices suggests an official-label
ceiling near 97.3% (±0.3), consistent with TransferNet's 97.5 and kgmem's 97.2. This is an estimate from
the label ambiguity, not a proven bound on every possible system. Two- and three-hop scores can reach
100. kgmem's remaining errors, 25 and 37 of about 14,500 questions each, mostly involve unusual titles
such as “Who Am I?” or walks through merged, same-titled films.

## Retrieval and decoding changes

The first adapter scored 96.6 / 99.3 / 79.2 when ranking entities and 89.7 / 80.7 / 63.3 when decoding
names. Tracing its failures led to four inference changes.

`exclude_head` prevents a later hop from returning through the question's topic entity. MetaQA's
three-hop answer paths do not revisit that topic. With the rule, recovered paths exactly reproduce
98.5% of three-hop answer sets, compared with 53.4% without it, and three-hop accuracy rises from
79.2 to 98.4.

`carry_top=0.5` lets a name continue when its weight is at least half the strongest candidate's.
“Who acted in The Butterfly Effect 2” retrieves five actors at 0.196 each. All fall below
`carry_share`'s 0.2 threshold, so the earlier decoder could mix names into “Eric Dustin”. The rank-relative
rule raises decoded two-hop accuracy from 80.7 to 97.1.

Longest-match handling in `exact_link` suppresses mention positions inside a complete longer name.
Without it, “The writer of Irma Vep” could attend to “ V” and select “Rocky V”; “the film Sukiyaki
Western Django” could select the genre “Western”.

Wider retrieval sets `max_tails=64` and `k_cand=64`. `ExactScorer` retains the first `max_tails` tails
for a (head, relation) pair. Peter Hyams has more than 16 directed films, and films with language
triples occurred later in that list. The truncated walk therefore found no third-hop answer for
“the languages of the films directed by the director of Stay Tuned”. In a 2,000-question sample,
widening retrieval reduces three-hop errors from 14 to 5.

## Graph edits and counterfactual facts

`experiments/qa/metaqa.py edit` moves 500 single-valued facts, such as a director or release year, to another tail
already used by that relation. It edits both directions of each triple without training the adapter.
Before editing, 99.8% of the affected questions decode the old answer; afterward, 99.8% decode the new
one. Of 500 questions about other entities, 99.6% keep the same decoded answer.

`experiments/qa/metaqa.py counterfactual` shuffles tails within each relation, preserving entity names and types.
Each film can receive another film's director, year or actors. It recomputes gold answers by walking
the shuffled graph and evaluates the unchanged adapter. On 1,000 questions per hop, Hits@1 is
99.8 / 99.6 / 98.9. The score supports graph-dependent lookup rather than recall of the original facts.

`--no_shuffle` recomputes answers from the real KB instead, accepting every answer reached by the
question's relation path. These clean-label scores are 99.8 / 99.8 / 99.8. Much of the gap from official
labels comes from merged film titles. Published comparison systems do not report this clean-label
evaluation, so these scores cannot be compared directly with their table rows.

## Checkpoint and runtime

The [how-to guide](../../how-to/run-benchmarks.md#metaqa) records the three training stages.
`checkpoints/metaqa.pt` is the third stage's best dev checkpoint, at step 8,000. The H100 stages took
about 14, 6 and 14 minutes at 0.08 s/step. Test evaluation took about 10 minutes; ranking all 39,093
questions took 3. `--answers_only` removes relation pseudo-label supervision.

The first stage used `max_tails=16, k_cand=32` before longest-match handling was added. Current defaults
include those later changes, so rerunning the recipe does not reproduce that first stage exactly.
The recorded data was the vanilla text release. The files at
`huggingface.co/datasets/camazlucas/MetaQA` matched the official question counts. The official Google
Drive download required a resource key that the tested `gdown` path could not pass.
