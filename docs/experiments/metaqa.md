# MetaQA

[Experiments](index.md)

[MetaQA](https://github.com/yuyuz/MetaQA) tests 1-, 2- and 3-hop questions over WikiMovies.
The graph has 43,234 entities, 9 relations before inverses and 134,741 triples. kgmem uses frozen
SmolLM2-135M with a 2.3M-parameter adapter. Hits@1 accepts any gold answer.

| Hits@1 (%), vanilla test sets | 1-hop | 2-hop | 3-hop |
|---|---|---|---|
| KV-MemNN | 95.8 | 25.1 | 10.1 |
| GraftNet | 97.0 | 94.8 | 77.7 |
| EmbedKGQA | 97.5 | 98.8 | 94.8 |
| NSM | 97.1 | 99.9 | 98.9 |
| TransferNet | 97.5 | 100 | 100 |
| **kgmem**: top entity the graph head retrieves, every test question | **97.2** | **99.8** | **99.7** |
| **kgmem**: the decoded answer, 1,000 questions per hop | **97.0** | **99.1** | **99.1** |
| kgmem trained on answers only (no relation pseudo-labels), retrieved | 94.8 | 95.6 | 57.1 |

The other systems' rows are paper results. kgmem's ranking row covers all 39,093 test questions;
decoding covers 1,000 per hop. Training supplies relation pseudo-labels recovered from training answers
and the KB. The answers-only row shows how much that supervision matters, especially at three hops.

## Graph dependence and limits

Without retraining, moving 500 single-valued facts changes 99.8% of affected decoded answers to the
new value, while 99.6% of 500 unrelated questions retain their answer. Shuffling facts and recomputing
gold answers gives 99.8 / 99.6 / 98.9 Hits@1 on 1,000 questions per hop.

Official labels sometimes omit valid answers because different films share a title. Clean labels
computed by walking the unshuffled KB give 99.8 / 99.8 / 99.8, but published baselines do not report
that protocol. Those numbers should not be compared directly with the paper rows above.

[Evaluate the shipped adapter](../how-to/run-benchmarks.md#metaqa).
[Detailed protocol and ablations](appendices/metaqa-details.md) cover ambiguous labels, inference
rules, checkpoint selection and runtime; [training recipes](appendices/benchmark-recipes.md#metaqa)
record the staged run.
