# WebQSP results and comparison with RoG

[Experiments](../index.md) / [Appendices](index.md)

WebQSP (Yih et al., 2016) contains Google Suggest questions over Freebase, such as “what does jamaican
people speak”. The RoG split has 2,826 training, 246 dev and 1,628 test questions. RoG (Luo et al.,
ICLR 2024) releases its code, a Llama-2-7B-chat model fine-tuned on WebQSP and CWQ, and the subgraph for
each question. These contain about 5,000 triples and 1,500 entities within two hops of the given topic,
with 6,094 relation types across the dataset. About 4% of questions have no answer in the subgraph.
NSM, UniKGQA and RoG evaluations also use these subgraphs with a supplied topic entity.

`experiments/qa/webqsp.py` loads each question's own subgraph, adds inverse edges and labels its topic from the
question. A 4-hop adapter starts from MQuAKE and trains on WebQSP answer paths. Among shortest paths
from the topic to an answer, supervision chooses the path whose endpoints are most often gold answers,
as in RoG's path training. Unnamed Freebase `m.*` / `g.*` compound-value nodes share a key. Predictions
rank named entities; every question has at least one named gold answer. Keys for 540k named entities
are encoded once and cached.

## Scoring RoG's output

The rerun uses RoG's `gen_rule_path.py`, `predict_answer.py` and `evaluate_results.py` with its pinned
versions. Its scorer gives 86.1, compared with the paper's 85.7. That scorer accepts a normalized gold
answer as a substring anywhere in the joined output, which averages 6.6 nonempty answer lines per
question. Scoring only the first nonempty answer line by normalized equality gives 79.2.
`experiments/qa/webqsp_compare.py` applies both rules to both systems.

| WebQSP test, Hits@1 (%) | first answer correct | any answer in the output (RoG's scorer) |
|---|---|---|
| RoG, Llama-2-7B fine-tuned on WebQSP + CWQ (its released model, rerun) | 79.2 | 86.1 |
| **kgmem**, SmolLM2-135M frozen, 3.9M-parameter adapter, WebQSP's training questions only | **65.2** | **65.5** |
| a walker that takes the right relation path and ranks an answer first | 95.6 | |

RoG's Table 1 reports KV-Mem 46.7, KGT5 56.1, Llama-2-Chat-7B 64.4, GraftNet 66.4, EmbedKGQA 66.6,
ChatGPT 66.8, NSM 68.7, SR+NSM 68.9, TransferNet 71.4, UniKGQA 77.2 and DECAF 82.1. These are published
scores under their respective implementations. Frozen SmolLM2 kgmem's 65.2 is below GraftNet,
EmbedKGQA and NSM's 66.4–68.7, and below this RoG rerun.

Relation selection is the main measured error source. Training relation accuracy reaches 96–99%
while dev stops near 65, from only 2,715 questions with a usable answer path across 6,094 relation
types. Restricting predictions to named entities raises dev from 63.8 to 65.4 and test from 62.3 to
65.2. Of all test questions, 24.6% take a wrong relation path, 6.0% reach the right path but choose
another endpoint, and 4.4% lack a subgraph answer.

Two frozen-backbone variants do not improve test accuracy:

| WebQSP test, first answer correct (%) | all | 923 CWQ-overlap | 705 other |
|---|---|---|---|
| kgmem, SmolLM2-135M, WebQSP's training questions | **65.2** | 68.4 | 61.0 |
| + CWQ's 19,058 training questions not built from a WebQSP test question (20k steps, 30% WebQSP) | 64.3 | 67.5 | 60.1 |
| the same adapter on Qwen2.5-7B-Instruct, frozen, layer 19, calibrated (fresh block) | 63.6 | 65.9 | 60.7 |

Both are selected on dev, scoring 64.2 and 65.9 versus the initial 65.4. These experiments do not show
that a larger frozen backbone or the additional nonoverlapping CWQ data fixes relation selection.
The 3.9M block on frozen states still trails the fine-tuned 7B chain generator. For the initial
retriever, a gold entity is in the top 1 / 3 / 5 / 10 for 64.5 / 76.1 / 79.1 / 82.6% of questions.
Even a perfect top-five reader would therefore reach only about RoG's first-answer score of 79.2.

## LoRA retrievers and answer readers

`--lora_r 16` (`kgmem/lora.py`) trains low-rank adapters in backbone layers 0..L along with the memory
block. Cached entity and relation keys remain those of the frozen backbone, avoiding re-encoding
540k names each step. This fine-tunes the question side of a dual encoder with fixed document keys.

`experiments/qa/webqsp_reader.py` presents the question, topic and top k named candidates, in retriever order, to a
reader. The frozen Qwen2.5-7B-Instruct reader selects the candidate number with highest probability
after “Answer:”, in one forward pass without sampling. k is chosen on dev before scoring test.

| WebQSP test, first answer correct (%) | all | 923 CWQ-overlap | 705 other |
|---|---|---|---|
| RoG (Llama-2-7B fine-tuned on WebQSP + CWQ train) | 79.2 | 87.7 | 68.2 |
| kgmem, Qwen2.5-7B + LoRA r=16 on layers 0..18 (27M) + block | 71.3 | 74.8 | 66.7 |
| the same, with the Qwen2.5-7B-Instruct reader over its top 10 | 75.0 | 78.9 | 69.9 |
| the same, the reader LoRA-trained to pick (cross-fitted candidates) | 77.9 | 81.3 | 73.5 |
| the same, trained readers: Qwen2.5-7B / Qwen3-8B / Qwen2.5-14B | 77.7 / 77.6 / 78.4 | 81.0 / 81.4 / 80.9 | 73.3 / 72.6 / 75.2 |
| **the same, trained Qwen2.5-32B-Instruct reader (best of the four on dev)** | **79.6** | 82.6 | **75.6** |
| the same, the four readers averaged (equal weights, fixed before scoring) | 79.4 | 82.7 | 75.2 |
| kgmem, SmolLM2-135M + LoRA r=16 + block | 64.4 | 67.3 | 60.7 |

The trained reader uses `experiments/qa/webqsp_reader.py --train_dumps`, LoRA r=16 on all layers and two epochs over top-10 candidate
lists. Two retrievers each train on half the questions and generate lists for the other half
(`experiments/qa/webqsp.py --fold 2:k`). Their first candidates score 68.1 and 70.0, so the reader sees retriever
errors during training rather than lists from a model trained on those same questions.

The first trained 7B reader scores 73.5 on the 705 test questions not extended by RoG training data,
compared with RoG's 68.2, a 5.3-point difference or about three standard errors. Overall it trails
RoG by 1.3 points, with 81.3 versus 87.7 on the 923 overlapping questions. Its 77.9 overall is above
UniKGQA's published 77.2 and below DECAF's 82.1, subject to their scoring differences.

LoRA raises the 7B retriever from 65.9 to 71.5 on dev and 63.6 to 71.3 on test. The 135M run's dev gain
does not carry to test. The LoRA retriever includes a gold answer in its top 1 / 3 / 5 / 10 for
71.3 / 81.4 / 84.8 / 87.7% of test questions, leaving a top-10 reader ceiling around 88.

Increasing the frozen reader to Qwen2.5-32B-Instruct does not help: it selects k=5 on dev at 73.2,
versus the 7B reader's 74.8 at k=10, and scores 73.5 on test. Training the larger readers does help.
Qwen2.5-14B and -32B use `--grad_ckpt` and accumulation to keep batch size 8, scoring 78.4 and 79.6.
The 32B model has the best dev score of the four tried, 78.5. Averaging them gives no improvement.
Candidate shuffling at test time (`--tta`) costs 4 to 7 dev points, consistent with the trained reader
using candidate rank as evidence.

`experiments/qa/webqsp_paired.py` compares strict first answers on the same questions:

| 32B reader against RoG | kgmem | RoG | difference (95% bootstrap) | McNemar p |
|---|---|---|---|---|
| all 1,628 test questions | 79.55 | 79.24 | +0.31 (−1.84, +2.46) | 0.82 |
| the 923 CWQ's training questions extend | 82.56 | 87.65 | −5.09 (−7.80, −2.49) | 0.0003 |
| the 705 none of RoG's training questions touches | 75.60 | 68.23 | **+7.38 (+3.97, +10.78)** | < 0.0001 |

The overall 0.31-point difference is unresolved by this test. The two source-defined subsets differ:
kgmem is about seven points ahead where RoG training has no extension of the test question, and
about five behind where it does.

## Using RoG's full training data

CWQ (Talmor & Berant, 2018) extends WebQSP queries with a constraint or another hop. Its ids retain
the source question: 8,581 of its 27,639 training questions extend 923 WebQSP test questions.
RoG's `scripts/train.sh` uses all CWQ training questions.

The overlap is relevant even before extra training. RoG scores 87.7 on those 923 questions and 68.2
on the other 705. WebQSP-only kgmem scores 68.4 and 61.0. The overlapping subset is about 7 points
easier for kgmem, while RoG's advantage increases by about 12 points on it. Thus the subset split
alone does not isolate the effect of overlap, but the training provenance belongs with the comparison.

`experiments/qa/webqsp.py prep --keep_webqtest` gives kgmem the same training sources as RoG: 30,465 questions and
908,949 named entity keys. The LoRA retriever uses the CWQ recipe for 15k steps. The unchanged trained
32B reader reads its top 10, with k chosen on dev at 80.1.

| same training data as RoG | kgmem | RoG | difference (95% bootstrap) | McNemar p |
|---|---|---|---|---|
| all 1,628 test questions | **81.39** | 79.24 | **+2.15 (+0.06, +4.24)** | 0.053 |
| the 923 CWQ's training questions extend | 83.86 | 87.65 | −3.79 (−6.50, −1.08) | 0.008 |
| the 705 none of RoG's training questions touches | 78.16 | 68.23 | +9.93 (+6.67, +13.19) | < 0.0001 |

Under this strict first-answer comparison, the overall bootstrap interval excludes zero but exact
McNemar p is 0.053. The gain is about ten points on the 705 questions not extended by training data;
the overlapping subset remains lower. The retriever alone scores 72.7 versus 71.3 when trained only
on WebQSP. Extra retriever data also improves the 705-question reader result from 75.6 to 78.2.

Three further readers train on candidates from two cross-fitted retrievers, each using half of the
30,465 questions. `experiments/qa/webqsp_reader.py --train_max` retains every WebQSP training question and adds a
seeded CWQ sample. Selection again uses dev accuracy.

| reader on the RoG-protocol candidates | dev | test | 923 overlap | 705 other |
|---|---|---|---|---|
| **Qwen2.5-32B, 8,000 questions, one epoch (best on dev)** | **80.9** | **81.57** | 85.05 | 77.02 |
| Qwen2.5-32B, 4,000 questions, two epochs | 79.7 | 81.63 | 84.72 | 77.59 |
| Qwen2.5-14B, 8,000 questions, two epochs | 78.9 | 80.22 | 83.21 | 76.31 |
| the 32B reader trained on WebQSP-only candidates (above) | 80.1 | 81.39 | 83.86 | 78.16 |

The headline results are in the [main report](../webqsp.md).

The selected reader's overall difference has McNemar p=0.034. Its gain is about nine points on the
nonoverlapping subset. On the overlapping subset the interval includes zero, which does not establish
equivalence. Averaging four readers with equal weights ties the best dev score, 80.9, and gives
81.94 on test, a +2.70 difference with McNemar p=0.014. The reported main result remains the single
reader selected on dev, not that higher test score.

The [how-to guide](../../how-to/run-benchmarks.md#webqsp) records the frozen-backbone recipe.
LoRA retriever and reader runs require the additional options described here.
