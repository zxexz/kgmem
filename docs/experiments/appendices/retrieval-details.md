# Retrieval details and controls

[Experiments](../index.md) / [Appendices](index.md)

`experiments/retrieval/funclib.py` tests relation selection with a synthetic function library. Each function has a typed signature,
a snake_case name and a one-line description, such as
`count_vowels_left(word) -> number: count the vowels to the left`. A verb, noun and qualifier determine the
name. Functions map numbers, words or codes to values of their output type. This isolates retrieval from
program execution; [strandedcpu](../../explanation/strandedcpu.md) supplies the separate program-backed scorer.

Questions compose one or two functions using different words from their names, for example
“What do you get if you tally the vowel letters leftward of Kazor?” The whole library is loaded at once.
A name hash excludes a fifth of the functions from training questions, while their verbs, nouns and
qualifiers still appear in other combinations. The following results score the composed answer with
Hits@1 in fresh worlds: frozen SmolLM2-135M, initialized from the MQuAKE adapter, 24 questions per step.

The headline results are in the [main report](../retrieval.md).

Increasing the 5,000-function run from 4,000 to 20,000 steps raises trained-function accuracy from 40.0 to
99.0 and unseen-combination accuracy from 24.4 to 97.9. That run had not exhausted the model's capacity at
4,000 steps. Adding functions after training still hurts: the 200-function adapter falls to 90.0 with
1,000 loaded and 68.3 with 5,000. The 1,000-function adapter scores 80.4 with 2,000 loaded and 70.8 with
5,000. These runs train with the complete target library present.

## Question-only training and input calibration

The custom Gemma-4-31B run uses bidirectional attention in its global layers. Placing an answer after
the question would expose it to the question-end state. `--question_only` instead uses one unpadded
question per row, trains only the linking, relation and halting losses, and evaluates the graph head's
top candidate at the question end. With a fresh block and 200 functions, SmolLM2 reaches 99.6 on trained
functions and 55.8 on unseen combinations in this format.

`--backbone-loader` imports an external model/attention loader. `--standardize` calls `MemoryLM.calibrate`
to scale its states before the block reads them. At Gemma-4-31B layer 41, one channel was about 186 on
every token while the others were near 1. Relation keys then had raw cosine 0.89, and relation accuracy
stayed near a quarter for 1,600 steps. Calibration applies `(h - shift) / scale` per dimension to both
tokens and keys. The saved shift and scale default to the identity, so older adapters retain their
behavior; the MQuAKE adapter's dev score agrees to four decimals.

## Retrieval with held-out wording

`--recall_eval` scores whether the first function and the whole chain occur among the top k candidates.
`--bm25` ranks the same name-signature-description text on the same questions. With `--hold_phrasing`,
training omits one synonym per word and one template per hop count. Questions about trained functions
use those withheld wordings; the other functions remain unseen compositions under either wording split.
These are Recall@5 results, first function / whole chain, after 4,000 steps at learning rate 3e-4.

| Recall@5 (%) | trained functions, unseen wording: 200 loaded | 1,000 loaded | unseen compositions: 200 loaded | 1,000 loaded |
|---|---|---|---|---|
| BM25 | 51.9 / 6.9 | 43.3 / 2.5 | 58.5 / 12.7 | 42.7 / 2.9 |
| kgmem, SmolLM2-135M, layer 19 | 98.1 / 96.7 | 92.3 / 86.7 | 79.6 / 61.9 | 59.0 / 35.2 |
| kgmem, Gemma-4-31B (custom loader), layer 41 | **99.4 / 99.0** | **96.5 / 94.4** | 34.8 / 8.3 | 24.4 / 0.2 |

The 31B backbone retrieves trained functions in unseen wording more accurately than SmolLM2, including
with 800 untrained functions added. Its accuracy on new compositions is much lower. Without
`--hold_phrasing`, trained-function cells reach 100 / 100, but every wording has then appeared in training.

The [how-to guide](../../how-to/run-benchmarks.md#function-library) gives commands and prerequisites.

## Tool catalogue experiments

`experiments/retrieval/toolmem.py` represents each of ToolRet's 44,453 tool documents (`mangopy/ToolRet-Tools`) as a relation,
keyed by its name and description. The relation decoder learns to select tools from a request's task
and query fields. `experiments/retrieval/toolret_prep.py` traces the positive documents in ToolRet-Training-20w to ToolBench,
APIGen and ToolACE by verbatim matches to the corpus.

This experiment trains on ToolBench rows and holds APIGen and ToolACE out as complete sources.
It does not use ToolRet's query sets, which belong to that benchmark's test data. Recall over the full
catalogue is:

The headline results are in the [main report](../retrieval.md).

For new requests involving trained tools, kgmem exceeds BM25 by 12 to 19 points at the reported recall
cutoffs. The gain does not transfer to unseen sources. When reranking BM25's 32 candidates there,
kgmem's first correct tool has MRR 0.20, compared with 0.67 in BM25's original order.

Adding or removing the 12k held-out tools barely changes recall on trained sources, 57.2 versus 57.3.
The catalogue can change without retraining, but these runs supply little evidence that the adapter can
select tools outside its training sources. SmolLM2 shows the same pattern at a lower trained-tool
Recall@10 of 32.9. The [function-library experiments](../retrieval.md) in `experiments/retrieval/funclib.py` also test this distinction
between replacing stored relations and learning to retrieve them.

## Confidence on a source omitted from training

`experiments/retrieval/toolret_loso.py` reserves a gate-fit pool of 1,000 requests and an evaluation pool of 1,500 per source.
`experiments/retrieval/toolmem.py --rows` trains one memory with each source omitted and one on all three, for 6k steps.
A logistic gate in `experiments/retrieval/toolmem_gate.py` estimates whether the highest-ranked candidate is correct using
the top score, margin, entropy and share of trained tools among candidates. Its fit uses only the
training sources' reserved pool.

`experiments/retrieval/toolmem_loso_report.py` fits another logistic model on the same pool to rerank BM25's 32 candidates
using [BM25 rank, memory score, score × gate, gate]. The control in `experiments/retrieval/toolret_knn.py` ranks a candidate
by the TF-IDF cosine of the nearest training request for which it is a gold tool.

| eval pool, nDCG@10 over BM25's 32 | BM25 | memory | nearest training request | stack − BM25 | gate AUC |
|---|---|---|---|---|---|
| ToolBench, memory trained without it | 0.330 | 0.104 | 0.166 | −0.032 | 0.23 |
| APIGen, memory trained without it | 0.526 | 0.177 | 0.211 | −0.049 | 0.28 |
| ToolACE, memory trained without it | 0.574 | 0.239 | 0.395 | −0.034 | 0.30 |
| APIGen, memory trained on all three | 0.526 | 0.652 | 0.740 | +0.133 | 0.82 |
| ToolBench, memory trained on all three | 0.330 | 0.464 | 0.562 | +0.106 | 0.80 |
| ToolACE, memory trained on all three | 0.574 | 0.440 | 0.302 | +0.002 | 0.56 |

On each omitted source, gate AUC is below chance at 0.23 to 0.30. Its mean predicted correctness stays
at 0.27 to 0.38 even though its chosen tool is correct on only 2 to 8% of requests. The stacked reranker
consequently falls below BM25 on all three omitted sources.

On ToolBench requests, a quarter of BM25's candidates are ToolACE or APIGen tools seen in training,
and the memory chooses one of them 77% of the time. Most are APIGen tools from the same domain;
APIGen uses ToolBench's RapidAPI pool. Some are functional duplicates, such as another VIN decoder.
Others only share a topic, such as a generic news list for a wrestling-news request.

On seen sources, nearest-training-request lookup usually helps more than the learned memory.
Adding memory scores to that lookup in a stack contributes another 0.01 to 0.04 nDCG@10.
These controls limit the claim to retrieval near the requests represented in training.
