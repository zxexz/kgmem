# kgmem

kgmem adds an editable knowledge graph and a writable scratchpad to a frozen language model.
A trained block links names in a question, follows graph edges, and copies the answer's name.
The graph can be replaced without retraining. Scratchpad state can be saved between calls.

The default demo uses SmolLM2-135M and the included adapter. It runs on CPU:

```sh
python -m pip install -e .
python examples/demo.py
```

The first run downloads the base model and tokenizer. The demo queries a small graph, changes two
facts, replaces the graph, then saves and reads two variables. See the
[first-run tutorial](docs/tutorials/first-run.md) for expected answers and
[setup guide](docs/how-to/setup.md) for dependencies and tests.

To inspect the graph, retrieval weights, and scratchpad in a browser:

```sh
python -m kgmem.inspector
```

## Where to start

- [Use the memory API](docs/how-to/use-memory.md), or read the [architecture](docs/explanation/architecture.md).
- [Experiment reports](docs/experiments/index.md) cover MetaQA, MQuAKE, WebQSP, retrieval, and the optional statistics store. Each report states what was trained and how answers were scored.
- [Reproduction guides](docs/how-to/run-benchmarks.md) give commands and dataset requirements. Detailed comparisons and failed runs are in the [appendices](docs/experiments/appendices/index.md).
- [API and checkpoint reference](docs/reference/index.md) lists the public interfaces and included adapters.

Adapters are specific to their base model. Included weights cover the SmolLM2 demo, MetaQA, MQuAKE,
and a Gemma 4 experiment. Datasets, base-model weights, and WebQSP checkpoints are separate downloads.
Arithmetic relations optionally use [strandedcpu](docs/how-to/use-strandedcpu.md).

## Repository layout

```text
kgmem/         Library and browser inspector
examples/      Small end-to-end demo and sample output
experiments/   Training and evaluation modules, grouped by task
checkpoints/   Current adapters; earlier versions in archive/
docs/          Guides, reference, design, and experiment reports
tests/         Library and experiment checks
```

Experiment commands write generated files under `runs/`. Put downloaded datasets in `data/`;
both directories are ignored by Git. See [the experiment entry points](experiments/README.md)
or the [documentation index](docs/index.md) for more.
