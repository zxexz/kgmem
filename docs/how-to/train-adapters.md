# Train and evaluate a synthetic adapter

[Documentation](../index.md) / [How-to guides](index.md)

Install the [experiment dependencies](setup.md) and create an output directory:

```sh
mkdir -p runs/synthetic
python -m experiments.synthetic.train --steps 1500 --p_pad 0.5 --out runs/synthetic/adapter.pt
python -m experiments.synthetic.eval_adapters runs/synthetic/adapter.pt --only wide
```

This trains a fresh SmolLM2 baseline on changing graphs and scratchpad tasks. `--only wide` evaluates
held-out names, graph questions and scratchpad recall without requiring strandedcpu.
Use `--help` on either module for model, layer, mixture and loss settings.

For a new base model, measure name linking before training:

```sh
python -m experiments.synthetic.check_linking --model HuggingFaceTB/SmolLM2-135M --layer 19
```

The [training objectives](../explanation/training.md) explain address and gate supervision.
The shipped v9 checkpoint follows the [recorded staged lineage](../experiments/appendices/training-recipes.md#v9),
which requires earlier checkpoints and cached plain-text data. That appendix records the v3–v9 recipes,
inference changes and reproducibility limits. [Gemma training fragments](../experiments/appendices/gemma4-recipes.md)
are separate because adapters do not transfer between backbones.

## Evaluate the included checkpoint

```sh
python -m experiments.synthetic.eval_adapters checkpoints/smollm2-v9.pt --only wide
python -m experiments.synthetic.eval_adapters checkpoints/smollm2-v9.pt --only strandedcpuwide
```

The second command requires [strandedcpu](use-strandedcpu.md). Checkpoints store training arguments,
warm starts, the last evaluation and recorded inference changes in `meta`.
See [persistence](../reference/python-api.md#persistence) for the saved fields.
