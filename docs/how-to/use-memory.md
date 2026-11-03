# Use and persist memory in Python

[Documentation](../index.md) / [How-to guides](index.md)

Run this example from the repository root with the [core dependencies](setup.md).
It loads the shipped SmolLM2 v9 adapter on CPU. Construct the wrapper from the saved configuration and
attach layer before calling `load()`; that method restores tensors, not the configuration.

## Load the adapter and a graph

```python
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kgmem import KnowledgeGraph, MemConfig, MemoryLM

adapter = "checkpoints/smollm2-v9.pt"
model_name = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
saved = torch.load(adapter, map_location="cpu", weights_only=True)
mlm = MemoryLM(model, tok, MemConfig(**saved["cfg"]), layer_idx=saved["layer_idx"])
mlm.load(adapter)
mlm.block.eval()

kg = KnowledgeGraph(
    entities=["Velmora", "Drakenhold", "Pellin"],
    relations=["capital"],
)
kg.add("Velmora", "capital", "Drakenhold")
mlm.load_graph(kg)
print(mlm.generate("Q: What is the capital of Velmora?\nA:"))
```

The adapter was trained on `Q: ...\nA:` prompts. Pass `use_memory=False` to compare with the base model.

## Edit an edge or replace the graph

```python
kg.set("Velmora", "capital", "Pellin")
print(mlm.generate("Q: What is the capital of Velmora?\nA:"))
```

`set()` replaces all tails for that head and relation. The default `ExactScorer` rebuilds its edge table
on the next lookup. Existing keys remain valid because the names have not changed.

For a new vocabulary, construct another `KnowledgeGraph` and call `mlm.load_graph(new_graph)`.
`add()` and `set()` require names already present in the graph.

## Remember values across calls

```python
mlm.reset_memory()
mlm.remember("Set kel to fox. Set vom to tea. Set kel to rice.\n")
Path("runs").mkdir(exist_ok=True)
mlm.save("runs/my_memory.pt")
```

`remember()` commits the write session to the persistent scratchpad. The checkpoint contains that buffer,
the adapter's configuration and its block weights. It excludes the base model, tokenizer and graph.

Reload the saved buffer before each independent question, since `persist=True` also commits writes made during generation:

```python
for name in ("kel", "vom"):
    mlm.load("runs/my_memory.pt")
    print(name, mlm.generate(f"Q: What is {name}?\nA:", persist=True))
```

The expected values are `kel = rice` and `vom = tea`. Use `persist=True` to read remembered values:
the default starts generation with an empty scratchpad.

To resume in another process, reconstruct the matching base model and wrapper from `runs/my_memory.pt`'s
`cfg` and `layer_idx` as in the first example, then call `load()`. Reload the graph separately if needed.
The [API reference](../reference/python-api.md#persistence) lists what the checkpoint contains.
