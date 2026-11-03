# Experimental appendices

[Experiments](../index.md)

These pages retain the evidence behind the concise reports, including failed runs and historical
checkpoint settings. They are optional reading rather than prerequisites for using kgmem.

| Appendix | Contents |
|---|---|
| [Adapter development](adapter-development.md) | Original through v9: name splits, scratchpad failures, gates and inference settings. |
| [MetaQA details](metaqa-details.md) | Label ambiguity, supervision, edits, counterfactual graphs and runtime. |
| [MQuAKE details](mquake-details.md) | Edit conflicts, paper and same-protocol baselines, RAE, prose and relation holdouts. |
| [WebQSP details](webqsp-details.md) | Retrievers, reader sizes, CWQ overlap, ensembles and paired tests. |
| [Retrieval details](retrieval-details.md) | Function wording splits, candidate growth, ToolRet holdouts and confidence controls. |
| [Statistics details](statistics-details.md) | Optional custom-backbone store, licences, abstention, negatives and latency. |
| [Program-backed relations](strandedcpu-details.md) | Relation linking, name completion, halting and execution timings. |
| [Gemma measurements](gemma4.md) | Attachment layer, training rounds, quantization and scaling. |
| [Synthetic training lineage](training-recipes.md) | Recorded v3–v9 commands and required intermediate artifacts. |
| [Gemma training fragments](gemma4-recipes.md) | First-run and round-6 fragments, with their prerequisites. |
| [Benchmark training](benchmark-recipes.md) | QA adapter, parser and synthetic-retrieval training commands. |

The historical [inspector audit](../../development/inspector-audit.md) is a development note.
RAE's original pipeline has separate [reproduction notes](../../../experiments/qa/baselines/rae/README.md).
