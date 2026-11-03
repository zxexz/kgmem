# Function and tool retrieval

[Experiments](index.md)

These experiments test whether naming a function or tool as a graph relation is enough to retrieve it
from a request. Replacing a catalogue is mechanically easy; selecting unfamiliar relations remains a
learned task.

## Synthetic functions

A function has a typed signature, a name and a description assembled from a verb, noun and qualifier.
Questions compose one or two functions in different words. A fifth of functions are absent from training
questions, though their name components occur in other combinations. Frozen SmolLM2-135M starts from
the MQuAKE adapter and trains with the complete target library loaded.

| Functions in the library | 50 | 200 | 1,000 | 5,000 | 5,000, 5x the steps |
|---|---|---|---|---|---|
| trained functions | 99.4 | 99.0 | 87.9 | 40.0 | 99.0 |
| combinations never trained on | 24.2 | 30.4 | 53.3 | 24.4 | 97.9 |

The 5,000-function result improves substantially with more steps. Adding large numbers of functions
after training still hurts. This synthetic compositional transfer does not establish transfer to
unrelated real tools.

## ToolRet

The 44,453-tool catalogue comes from ToolRet. Training requests are split by source: ToolBench,
APIGen and ToolACE. The following run trains on ToolBench only and evaluates new requests across
the whole catalogue. ToolRet's own benchmark query sets are not used.

| Recall (%) over 44,453 tools | @1 | @5 | @10 | @32 | @100 |
|---|---|---|---|---|---|
| new ToolBench requests (tools trained on): kgmem, Gemma-4-31B layer 41 | 17.5 | 46.1 | 57.2 | 72.2 | 83.1 |
| the same requests: BM25 over the tool documents | 11.0 | 34.2 | 43.6 | 54.9 | 64.3 |
| APIGen and ToolACE requests (tools never trained on): kgmem | 0.0 | 0.1 | 0.5 | 1.8 | 5.4 |

On trained tools, memory improves over BM25. It performs poorly on tools from omitted sources.
A confidence gate fitted only on seen sources does not repair this: its left-out-source AUC is
0.23–0.30, and the resulting reranker falls below BM25. Nearest-training-request lookup is a stronger
control on seen sources than the memory alone.

[Run synthetic retrieval](../how-to/run-benchmarks.md#function-library).
[Detailed retrieval controls](appendices/retrieval-details.md) retain Recall@5, larger candidate sets,
source holdouts, gate calibration and custom-backbone measurements. The custom loader and its artifacts
are optional; the SmolLM2 synthetic experiment runs with the public dependencies.
