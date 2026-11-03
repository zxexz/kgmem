# MQuAKE

[Experiments](index.md)

[MQuAKE](https://github.com/princeton-nlp/MQuAKE) asks multi-hop questions after edits to one to four
facts. Each case has three paraphrases and counts as solved if any yields the new answer or an alias.
kgmem applies edits to its graph without retraining. The frozen SmolLM2-135M backbone uses a
3.9M-parameter, four-hop adapter trained on questions and chains disjoint from the test cases.

| Dataset, all edits loaded | kgmem | Fully fine-tuned SmolLM2 parser |
|---|---|---|
| CF-3k-v2 | 98.7 | 96.3 |
| MQuAKE-2002 | 98.6 | 97.4 |
| MQuAKE-hard | 98.1 | 96.0 |
| MQuAKE-T | 98.8 | 97.7 |

Both systems use the same graph, edit batches and scoring. kgmem's numbers rank retrieved entities;
decoded-answer checks on samples agree within sampling error. The parser trains all 135M weights on
the same questions; kgmem trains only its block. These comparisons differ from methods that keep only
edits and ask the LM to supply every unedited fact.

## Conflict policy and preservation

The original CF-3k has conflicting edits: another case can change a fact required by the gold chain.
With every edit applied, a gold-path walker matches 67.5% of labels and kgmem matches 67.0%.
CF-3k-v2 fixes those conflicts. Masked evaluation restores required facts per case and is a separate
protocol, not an interchangeable full-batch score.

With the graph and block attached, the shipped adapter matches the base model's loss to four decimals
on 300 WikiText-2 test paragraphs. Holding out relation types sharply reduces accuracy, though name
linking transfers more than the parser's output vocabulary. These observations bound the claims about
LM preservation and unseen schemas.

[Evaluate the shipped adapter](../how-to/run-benchmarks.md#mquake).
[Full results and protocols](appendices/mquake-details.md) retain paper comparisons, masked and unedited
scores, held-out relations and [RAE reruns](appendices/mquake-details.md#rae-reruns).
[Training and baselines](appendices/benchmark-recipes.md#mquake) give commands and checkpoint conditions.
