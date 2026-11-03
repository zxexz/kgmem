# WebQSP

[Experiments](index.md)

WebQSP asks Freebase questions against per-question subgraphs with a supplied topic entity.
The RoG split has 2,826 training, 246 dev and 1,628 test questions. The best reported kgmem run uses
a LoRA-trained Qwen2.5-7B retriever plus a LoRA-trained Qwen2.5-32B candidate reader. This experiment
fine-tunes the backbone; it is separate from the frozen-backbone MetaQA and MQuAKE results.

The following comparison uses the same WebQSP and CWQ training sources as RoG. Both outputs are scored
by normalized equality of the first nonempty answer line. The reader is selected on dev, not test.

| best-on-dev reader against RoG, same training data | kgmem | RoG | difference (95% bootstrap) | McNemar p |
|---|---|---|---|---|
| all 1,628 test questions | **81.57** | 79.24 | **+2.33 (+0.25, +4.42)** | **0.034** |
| the 923 CWQ's training questions extend | 85.05 | 87.65 | −2.60 (−5.31, +0.11) | 0.069 |
| the 705 none of RoG's training questions touches | 77.02 | 68.23 | +8.79 (+5.53, +12.06) | < 0.0001 |

RoG's released scorer instead accepts a gold answer as a substring anywhere in its output, averaging
6.6 answer lines. That rerun scores 86.1 under its own rule and 79.24 under the strict first-answer rule.
The table therefore describes the strict comparison, not a claim against its paper's headline metric.

## What changes the result

The frozen SmolLM2 block reaches 65.2%, well below RoG. LoRA improves relation selection, and a reader
then chooses among retrieved candidates. Reader training uses cross-fitted lists from retrievers that
did not train on those questions. CWQ extends 923 WebQSP test questions, so results are also reported
on those questions and the other 705. The split alone does not isolate the causal effect of overlap.

No WebQSP retriever or reader checkpoint is shipped. The public [preparation and baseline guide](../how-to/run-benchmarks.md#webqsp)
starts the experiment; [full tables and training conditions](appendices/webqsp-details.md) preserve
frozen-backbone failures, reader sizes, overlap analysis and paired tests.

## CWQ, RoG's other benchmark

The same RoG-protocol retriever and dev-best reader, applied unchanged to CWQ's test set, lose clearly to RoG rerun
with its own code. RoG's rerun reproduces its paper (lenient hit 61.9 against 62.6); both are scored by the strict
first-answer rule, paired on the 3,531 test questions:

| CWQ test | kgmem | RoG | difference (95% bootstrap) | McNemar p |
|---|---|---|---|---|
| all 3,531 questions | 43.78 | 57.15 | −13.37 (−15.29, −11.47) | < 0.0001 |
| with the constraint step, reader at k=10 | 47.01 | 57.15 | −10.14 (−12.06, −8.24) | < 0.0001 |

The retriever's first candidate is right on 29.3% of the test questions (dev 29.0), and a reader cannot recover an
answer the retriever's top ten does not hold. CWQ's questions extend WebQSP's with constraints, comparatives,
superlatives and further hops; the graph walk follows relation paths of up to three hops and had no constraint step.

A first constraint step (`experiments/qa/cwq_constrain.py`) narrows a quarter of the gap. For questions naming two or
more entities (1,702 of the test set), the retriever's top 50 candidates that lie within two hops of every named entity
in the question's subgraph move ahead of the rest, each group keeping the retriever's order; 1,188 test lists change.
Nothing is retrained. Recall of
a gold answer in the top 1 / 10 rises from 29.3 / 61.1 to 34.4 / 64.3 on test (dev 29.0 / 62.7 to 35.1 / 65.5). The
unchanged reader then picks k=20 on dev (50.67, from 46.43 at k=10 before the step) and scores 47.55 on test; the
paired row uses its k=10 probabilities. Comparatives, superlatives and longer chains are still unhandled, and the
reader sees candidate names only, not the paths that would let it check a constraint.
