# RAE baseline

These tools run [RAE](https://github.com/sycny/RAE)'s original GPT-2 XL beam-search pipeline on a
seeded 300-case sample of MQuAKE-CF-3k. This is a separate comparison from kgmem's graph protocol.

## Inputs

- A RAE checkout and an environment with its dependencies and model weights.
- `MQuAKE-CF-3k.json` and `MQuAKE-CF.json` in that checkout's `data/` directory.
- RAE's edited Wikidata5m graph, prepared using its `preprocess/edit_KG.py` procedure.
- Wikidata5m's entity-alias TSV.

If rebuilding RAE's initial graph from the public triples, create its head dictionary with:

```sh
python -m experiments.qa.baselines.rae.run build-graph \
  --triples data/wd5m/train.txt data/wd5m/valid.txt data/wd5m/test.txt \
  --out data/rae/Wikidata_triplets_dict.pkl
```

Configure RAE's graph-preparation script to use that dictionary and the MQuAKE data, then apply the edits.
The runner does not replace this upstream preparation step.

## Run and score

From the kgmem repository root, pass the locations of your inputs:

```sh
python -m experiments.qa.baselines.rae.run run \
  --rae data/RAE --aliases data/wd5m/wikidata5m_entity.txt --out runs/rae
```

Use `--python /path/to/rae-environment/bin/python` if RAE needs a separate environment.
The defaults select 300 of the dataset's cases with seed 0 and split them across three workers.
`sample.json`, per-worker logs, the patched entry script, and `score.json` go under the output directory.
RAE's model, pruning, graph-editing code, and default `correctConflict` behavior remain in use.

The generated entry script adds a descriptive User-Agent, reads labels from the alias file, and filters
the case list. Entity search still calls Wikidata, with backoff for rate limits. The original RAE source
is not modified. After an interrupted run, repeat the command with `--resume` to run unfinished cases.

To score existing logs and check sample coverage:

```sh
python -m experiments.qa.baselines.rae.run score runs/rae/shard_*.log --sample runs/rae/sample.json
```

A case is correct when any question has `Raw:1` or `Pruned:1`. Scoring counts completed cases only.

## Recorded comparison

The recorded 300-case run scored **37.7% with pruning and 35.0% without**, versus the paper's **62.8%**
for GPT-2 on all MQuAKE-CF-3k. It rebuilt the graph from public Wikidata5m triples because the supplied
graph required a university sign-in. It also used the 2019 alias snapshot instead of live 2024 labels.
The rebuilt graph, sample size, and changes in live entity search prevent attributing that gap to a
single cause. See the [MQuAKE report](../../../../docs/experiments/mquake.md) for the comparison on kgmem's graph.
