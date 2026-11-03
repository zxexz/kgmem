# Benchmark training recipes

[Experiments](../index.md) / [Appendices](index.md)

Run these from a checkout after `python -m pip install -e '.[experiments,test]'`.
The [evaluation guide](../../how-to/run-benchmarks.md) describes input layouts and shipped-adapter evaluation.
These recorded training runs write to `runs/`; datasets and generated baselines are not included.

```sh
mkdir -p runs/metaqa runs/mquake runs/webqsp runs/retrieval
```

## MetaQA

The three training stages initialize one another. For a newly trained adapter, pass the final run's path to evaluation. The included checkpoint is selected by dev score as described in [MetaQA details](metaqa-details.md).

```sh
python -m experiments.qa.metaqa train --prep runs/metaqa/prep.pt --init checkpoints/smollm2-v9.pt --steps 10000 --out runs/metaqa/round1.pt
python -m experiments.qa.metaqa train --prep runs/metaqa/prep.pt --init runs/metaqa/round1.pt --steps 4000 --lr 3e-4 --out runs/metaqa/round2.pt
python -m experiments.qa.metaqa train --prep runs/metaqa/prep.pt --init runs/metaqa/round2.pt --steps 10000 --lr 3e-4 --aux 2.0 --out runs/metaqa/adapter.pt
```

## MQuAKE

Train the graph adapter from the included MetaQA checkpoint:

```sh
python -m experiments.qa.mquake train --prep runs/mquake/prep.pt --init checkpoints/metaqa.pt --steps 25000 --eval_every 1000 \
    --seed 1 --p_plain 0.2 --out runs/mquake/adapter.pt
```

The parser, few-shot LLM and preservation measurement are separate runs. Preservation uses the parser
saved below; the LLM command needs a separate vLLM environment.

```sh
python -m experiments.qa.mquake_baselines parser --prep runs/mquake/prep.pt --model HuggingFaceTB/SmolLM2-135M \
    --out runs/mquake/parser.json --save_model runs/mquake/parser-model
python -m experiments.qa.mquake_baselines llm --engine vllm --prep runs/mquake/prep.pt \
    --model Qwen/Qwen2.5-7B-Instruct --out runs/mquake/qwen.json
python -m experiments.qa.lm_preservation --prep runs/mquake/prep.pt --adapter checkpoints/mquake.pt --parser runs/mquake/parser-model
```

Relation holdouts require another trained checkpoint. The parser and LLM baseline commands also accept
`--hold_out_rel` for the corresponding comparison.

```sh
HOLD="continent,creator,chief executive officer,educated at,chairperson,officeholder"
python -m experiments.qa.mquake train --prep runs/mquake/prep.pt --init checkpoints/metaqa.pt --steps 25000 --eval_every 1000 \
    --seed 1 --p_plain 0.2 --hold_out_rel "$HOLD" --out runs/mquake/held-out.pt
python -m experiments.qa.mquake eval --prep runs/mquake/prep.pt --adapter runs/mquake/held-out.pt --k all --sets cf3k-v2,T,hard \
    --hold_out_rel "$HOLD"
```

## WebQSP

This is the frozen SmolLM2 baseline. Comparing with RoG additionally requires its prediction JSONL and a JSON list of CWQ-overlap test ids. LoRA and trained-reader runs use `experiments/qa/webqsp.py` / `experiments/qa/webqsp_reader.py` options described in [the detailed report](webqsp-details.md).

```sh
python -m experiments.qa.webqsp_compare --prep runs/webqsp/prep.pt --kgmem runs/webqsp/predictions.test.jsonl --rog runs/webqsp/rog-predictions.jsonl \
                       --overlap runs/webqsp/cwq-overlap.json          # RoG's results/KGQA/... predictions
```

## Function library

The main evaluation guide covers catalogue size. These commands cover added functions, question-only training, held-out wording and BM25. An external backbone loader is optional and is not used in this SmolLM2 recipe.

```sh
python -m experiments.retrieval.funclib --n_funcs 1000 --steps 4000 --swap_every 4 --eval_libs 2000,5000 --out runs/retrieval/distractors.json
python -m experiments.retrieval.funclib --init "" --layer 19 --question_only --standardize --hold_phrasing --lr 3e-4 --n_funcs 200 \
                   --steps 4000 --save runs/retrieval/question-only.pt --out runs/retrieval/question-only.json
python -m experiments.retrieval.funclib --recall_eval runs/retrieval/question-only_200.pt --layer 19 --eval_libs 1000 --out runs/retrieval/recall.json
python -m experiments.retrieval.funclib --bm25 --hold_phrasing --n_funcs 200 --eval_libs 1000 --out runs/retrieval/bm25.json
```
