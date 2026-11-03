# Use the Gemma 4 adapter

[Documentation](../index.md) / [How-to guides](index.md)

Use access to `google/gemma-4-26B-A4B-it` and a Transformers release supporting its model class.
The recorded bf16 run used one 80 GB GPU. The checkpoint attaches at layer 23.

```sh
python examples/demo.py --model google/gemma-4-26B-A4B-it --dtype bfloat16 --adapter checkpoints/gemma4.pt
python -m experiments.synthetic.eval_adapters checkpoints/gemma4.pt --model google/gemma-4-26B-A4B-it --dtype bfloat16 --only wide
```

Remove `--only wide` for the full evaluation if [strandedcpu](use-strandedcpu.md) is installed.
Before training a new adapter, compare attachment layers with the zero-shot linker:

```sh
python -m experiments.synthetic.check_linking --model google/gemma-4-26B-A4B-it --dtype bfloat16 --layer 23
```

`--distractors N` adds unused entity names to test a larger key table.
The [Gemma appendix](../experiments/appendices/gemma4.md) records layer selection, memory requirements,
quantization checks and results. The [training fragments](../experiments/appendices/gemma4-recipes.md)
require intermediate artifacts; use the included checkpoint to run the final recorded model.
