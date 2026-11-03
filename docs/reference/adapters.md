# Included adapters

[Documentation](../index.md) / [Reference](index.md)

Each checkpoint requires the base model, configuration and attachment layer used in training.
Construct `MemoryLM` from its saved `cfg` and `layer_idx` before calling `load()`.

| Checkpoint | Base model | Purpose | Results |
|---|---|---|---|
| `checkpoints/smollm2-v9.pt` | `HuggingFaceTB/SmolLM2-135M` | Demo, synthetic graph questions and scratchpad | [Adapter development](../experiments/appendices/adapter-development.md) |
| `checkpoints/metaqa.pt` | `HuggingFaceTB/SmolLM2-135M` | Three-hop WikiMovies QA | [MetaQA](../experiments/metaqa.md) |
| `checkpoints/mquake.pt` | `HuggingFaceTB/SmolLM2-135M` | Four-hop edited-graph QA | [MQuAKE](../experiments/mquake.md) |
| `checkpoints/gemma4.pt` | `google/gemma-4-26B-A4B-it` | Graph questions and scratchpad, layer 23 | [Gemma appendix](../experiments/appendices/gemma4.md) |

The files contain adapter weights and state, not the backbone or datasets.
Build benchmark preparation files under `runs/` before evaluation. No WebQSP reader/retriever or optional
custom-backbone checkpoint is included.

`checkpoints/archive/` retains the original and v2–v8 SmolLM2 files under their historical names for
reproduction. They are not needed for the first run. The [comparison and ablations](../experiments/appendices/adapter-development.md#checkpoint-comparison)
explain their differences. The original demo transcript is at
[`examples/output/original-adapter.txt`](../../examples/output/original-adapter.txt).
