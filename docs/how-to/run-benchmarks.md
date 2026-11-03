# Evaluate QA benchmarks

[Documentation](../index.md) / [How-to guides](index.md)

Install the [experiment dependencies](setup.md). Datasets are downloaded separately; preparation files
and predictions are generated under `runs/`. The [reports](../experiments/index.md) state each scoring protocol.

## MetaQA

Place the vanilla MetaQA release under `data/metaqa`, with `kb.txt` and
`{1,2,3}-hop/vanilla/qa_{train,dev,test}.txt`.

```sh
mkdir -p runs/metaqa
python -m experiments.qa.metaqa prep --data data/metaqa --out runs/metaqa/prep.pt
python -m experiments.qa.metaqa eval --prep runs/metaqa/prep.pt --adapter checkpoints/metaqa.pt --generate 1000
python -m experiments.qa.metaqa edit --prep runs/metaqa/prep.pt --adapter checkpoints/metaqa.pt --n 500
python -m experiments.qa.metaqa counterfactual --prep runs/metaqa/prep.pt --adapter checkpoints/metaqa.pt --n 1000
```

Add `--no_shuffle` to `counterfactual` for clean labels on the original KB.
See [results](../experiments/metaqa.md) and the [training recipe](../experiments/appendices/benchmark-recipes.md#metaqa).

## MQuAKE

Place `MQuAKE-CF.json`, `MQuAKE-CF-3k.json`, `MQuAKE-CF-3k-v2.json` and `MQuAKE-T.json` under
`data/mquake`. Optional subsets are `MQuAKE-2002.json`, `MQuAKE-hard.json` and
`remastered/{CF3k,T,CF9k}.parquet`.

```sh
mkdir -p runs/mquake
python -m experiments.qa.mquake prep --data data/mquake --out runs/mquake/prep.pt
python -m experiments.qa.mquake eval --prep runs/mquake/prep.pt --adapter checkpoints/mquake.pt --k all --unedited --generate 300
```

Use `--k 1,100,1000,all,masked` for all edit-batch protocols. Masked evaluation restores facts per case;
it must be reported separately from applying every edit consistently.
See [results and protocol](../experiments/mquake.md) and [training/baseline commands](../experiments/appendices/benchmark-recipes.md#mquake).

## WebQSP

Place RoG-webqsp's parquet files under `data/webqsp/rog`.
No WebQSP adapter is included, so this starts a frozen SmolLM2 baseline:

```sh
mkdir -p runs/webqsp
python -m experiments.qa.webqsp prep --data data/webqsp/rog --out runs/webqsp/prep.pt
python -m experiments.qa.webqsp keys --prep runs/webqsp/prep.pt --adapter checkpoints/mquake.pt --out runs/webqsp/keys.pt
python -m experiments.qa.webqsp train --prep runs/webqsp/prep.pt --keys runs/webqsp/keys.pt --init checkpoints/mquake.pt --steps 10000 --eval_every 1000 --out runs/webqsp/adapter.pt
python -m experiments.qa.webqsp eval --prep runs/webqsp/prep.pt --keys runs/webqsp/keys.pt --adapter runs/webqsp/adapter.pt --splits validation,test --dump runs/webqsp/predictions
```

The [WebQSP report](../experiments/webqsp.md) describes the separate LoRA and trained-reader runs.
RoG comparisons additionally need its predictions and a CWQ-overlap file; the
[comparison recipe](../experiments/appendices/benchmark-recipes.md#webqsp) lists their formats.

## Function library

```sh
mkdir -p runs/retrieval
python -m experiments.retrieval.funclib --n_funcs 50,200,1000,5000 --init checkpoints/mquake.pt --steps 4000 --swap_every 4 --out runs/retrieval/functions.json
```

The [retrieval report](../experiments/retrieval.md) explains trained-function and unseen-composition scores.
More [retrieval recipes](../experiments/appendices/benchmark-recipes.md#function-library) cover held-out wording,
BM25 and added functions. ToolRet and statistics modules expose their own `--help`; custom-backbone
runs require an [external loader](custom-backbones.md).
