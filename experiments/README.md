# Experiment entry points

Run these modules from the repository root after installing the experiment dependencies:

```sh
python -m pip install -e '.[experiments]'
```

| Directory | Tasks | Example |
|---|---|---|
| `synthetic/` | Small graph training, linking, scratchpad, and arithmetic checks | `python -m experiments.synthetic.eval_adapters --help` |
| `qa/` | MetaQA, MQuAKE, WebQSP, and their comparison methods | `python -m experiments.qa.mquake --help` |
| `retrieval/` | Function libraries and ToolRet | `python -m experiments.retrieval.toolret_loso --help` |
| `statistics/` | Country–indicator–year memory and standalone runtime export | `python -m experiments.statistics.owid_store --help` |

Start with the [experiment reports](../docs/experiments/index.md) to choose a setup, then use the
[benchmark guide](../docs/how-to/run-benchmarks.md) or [training guide](../docs/how-to/train-adapters.md).
Commands accept explicit dataset and output paths; defaults for generated files live under `runs/`.
The included adapters are in [`checkpoints/`](../checkpoints/README.md).

The [RAE comparison runner](qa/baselines/rae/README.md) needs an external RAE checkout and its data.
Some retrieval and statistics runs use an optional [custom backbone loader](../docs/how-to/custom-backbones.md).
Neither dependency is needed for the default demo or the public QA adapters.
