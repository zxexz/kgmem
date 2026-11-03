# Repository layout

[Documentation](../index.md) / [Reference](index.md)

| Path | Purpose |
|---|---|
| [`kgmem/`](../../kgmem) | Installable library: graph scorers, memory block, model wrapper, synthetic tasks and CUDA kernels. |
| [`kgmem/inspector/`](../../kgmem/inspector) | Local inspector, static assets and shared graph fixtures. Run with `python -m kgmem.inspector`. |
| [`examples/`](../../examples) | Demo and small usage examples. |
| [`experiments/synthetic/`](../../experiments/synthetic) | Adapter training, cached pretraining, linking checks and synthetic evaluation. |
| [`experiments/qa/`](../../experiments/qa) | MetaQA, MQuAKE, WebQSP and their baselines/readers. |
| [`experiments/retrieval/`](../../experiments/retrieval) | Function libraries, ToolRet, confidence gates and nearest-request controls. |
| [`experiments/statistics/`](../../experiments/statistics) | Optional OWID store, hooks, latency checks and runtime packaging. |
| [`checkpoints/`](../../checkpoints) | Current adapters; historical versions are in `archive/`. |
| [`tests/`](../../tests) | Unit, integration, scoring and optional CUDA/backend tests. |
| [`docs/`](..) | Tutorials, task guides, API reference, design and experiment reports. |

Experiments run as `python -m experiments.<group>.<module>` from the checkout after an editable install.
They are not part of the installed library. Downloaded datasets and generated outputs are not source:
use `data/` and `runs/`, respectively.
