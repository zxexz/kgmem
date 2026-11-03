# Supply a custom backbone loader

[Documentation](../index.md) / [How-to guides](index.md)

Some retrieval and statistics experiments can load a model with custom attention behavior.
The public SmolLM2 demo and QA benchmarks do not require this interface.

`--backbone-loader` names a Python file supplied by the caller. It must export:

- `register()`, returning the Transformers attention implementation name to select.
- `require_policy(text_config, policy)`, checking that the model's text configuration supports the requested policy.

The loader must make its model and attention dependencies importable in the current environment.
This optional path loads CUDA bf16 weights; `--policy` defaults to `B` in the current modules.
Pass the file explicitly, for example `--backbone-loader plugins/backbone.py`.
Use `python -m experiments.retrieval.funclib --help` for model and policy arguments supported by that module.
No external attention code or evaluated custom-backbone checkpoint is bundled here.

With bidirectional prompt attention, use `--question_only`: answer tokens appended after the question
would otherwise be visible to the representation used for retrieval. `--standardize` calibrates the
memory block's input dimensions when that model requires it. These alter the experiment's conditions;
report the loader, attention policy and calibration with its results.

The optional [statistics experiment](../experiments/statistics-store.md) and
[retrieval appendix](../experiments/appendices/retrieval-details.md) describe the recorded custom-backbone runs.
