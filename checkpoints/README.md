# Checkpoints

Each file contains adapter weights, configuration, and training metadata. Base-model weights are separate.

| File | Base model | Use |
|---|---|---|
| `smollm2-v9.pt` | `HuggingFaceTB/SmolLM2-135M` | CPU demo, synthetic graphs, scratchpad, and arithmetic relations |
| `metaqa.pt` | `HuggingFaceTB/SmolLM2-135M` | MetaQA movie-graph questions |
| `mquake.pt` | `HuggingFaceTB/SmolLM2-135M` | MQuAKE knowledge editing |
| `gemma4.pt` | `google/gemma-4-26B-A4B-it` | Gemma 4 graph and scratchpad demo; recorded bf16 run needs an 80 GB GPU |

Run the default demo from the repository root after installing the package:

```sh
python examples/demo.py
```

The older SmolLM2 adapters are in `archive/`, under their original names. They are retained for
the [development comparisons](../docs/experiments/appendices/adapter-development.md); the current examples
use v9. Training metadata uses artifact names rather than paths from the original training machines.
