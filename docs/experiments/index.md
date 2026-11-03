# Experiments

[Documentation](../index.md)

kgmem separates facts from the learned question-to-graph mapping. The experiments test that mapping,
answer decoding and interference with the frozen backbone. Scores below use different datasets and
scoring rules; the reports state the supervision and comparison conditions.

| Experiment | Main result | Scope |
|---|---|---|
| [MetaQA](metaqa.md) | 97.2 / 99.8 / 99.7 Hits@1 for 1 / 2 / 3 hops | Frozen SmolLM2-135M; graph-head ranking on official test labels. |
| [MQuAKE](mquake.md) | 98.7% on CF-3k-v2 with all edits loaded | Frozen SmolLM2-135M; graph-backed knowledge editing. |
| [WebQSP](webqsp.md) | 81.57% strict first-answer accuracy, versus RoG's 79.24% | LoRA retriever and trained 32B reader, using RoG's training data. |
| [Function and tool retrieval](retrieval.md) | Strong trained-library retrieval; weak transfer to unseen tool sources | Synthetic functions and ToolRet, with BM25 and nearest-request controls. |
| [Optional statistics store](statistics-store.md) | 99.73% combined accuracy on held-out stored keys, versus 70.70% backbone alone | Separate custom-backbone setup; selective country–indicator–year lookup. |

[Run the public QA experiments](../how-to/run-benchmarks.md) or inspect
[the shipped checkpoints](../reference/adapters.md). WebQSP retriever/reader checkpoints and the optional
custom-backbone artifacts are not included.

## Detailed evidence

The [appendices](appendices/index.md) retain full comparison tables, failed experiments, checkpoint
lineage, Gemma measurements and reproduction conditions. Start there for ablations rather than reading
the development sequence as setup instructions.
