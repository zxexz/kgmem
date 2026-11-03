# Install and test

[Documentation](../index.md) / [How-to guides](index.md)

Use Python 3.10 or later. Run from a checkout so the experiment modules and example files are available.
Only `kgmem` is installed as a library; `experiments` remains checkout tooling.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
mkdir -p runs data
python examples/demo.py
```

The first run downloads SmolLM2-135M and its tokenizer. Scripts use CUDA when available and CPU otherwise;
they do not select Apple MPS automatically. The included checkpoints are listed in the
[adapter reference](../reference/adapters.md).

## Experiments and tests

```sh
python -m pip install -e '.[experiments,test]'
python -m pytest -q tests
```

The `experiments` extra adds datasets, numpy, scipy and pyarrow. The `test` extra adds pytest and numpy.
Integration tests need downloaded tokenizers or a populated cache. Gemma tests construct a tiny random
model from the real configuration and skip when its class, configuration or tokenizer is unavailable.
CUDA kernel tests need a CUDA device and nvcc.

## Optional integrations

- [strandedcpu](use-strandedcpu.md) provides checked arithmetic relations. Its tests skip when unavailable.
- `PyKEENScorer` requires `pykeen` and a trained model with its matching `TriplesFactory`.
- Fused CUDA extensions build on first use when nvcc is available.
- The [Gemma 4 recipe](run-gemma4.md) needs a Transformers release supporting `Gemma4ForConditionalGeneration`;
  the recorded bf16 run used one 80 GB GPU.
- vLLM baselines need their own vLLM environment.
- [Custom backbone loaders](custom-backbones.md) are optional and must provide their own model and attention code.
