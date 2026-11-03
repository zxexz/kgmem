# Recorded synthetic training lineage

[Experiments](../index.md) / [Appendices](index.md)

These are the recorded SmolLM2 recipes. Run them from the repository root in the
[core environment](../../how-to/setup.md). Each version's warm start, cache, and post-training inference settings are
part of the run. The original v2 initialization was unseeded, so a rerun need not match its scores exactly.

```sh
mkdir -p runs/synthetic
```

[Training design](../../explanation/training.md) explains the losses;
[adapter history](adapter-development.md) records ablations and timings.

<a id="v3"></a>

## Train v3 from a fresh cache

The v3 run predates the relation-world and plain-text batches used by later versions.

```sh
python -m experiments.synthetic.pretrain_linkers cache --n_kg 400 --n_pad 400 --out runs/synthetic/cache_smol     # ~2 min on a 14-core Mac
python -m experiments.synthetic.pretrain_linkers train --cache runs/synthetic/cache_smol --epochs 4 --out runs/synthetic/stage1.pt
python -m experiments.synthetic.train --init runs/synthetic/stage1.pt --steps 1500 --p_pad 0.5 --eval_every 250 --out runs/synthetic/v3.pt
```

<a id="v6"></a>

## Train v6

```sh
python -m experiments.synthetic.pretrain_linkers cache --n_kg 400 --n_pad 400 --n_rel 400 --n_plain 100 --rel_key_pool last --out runs/synthetic/cache_v6
python -m experiments.synthetic.pretrain_linkers train --cache runs/synthetic/cache_v6 --epochs 4 --rel_mention --out runs/synthetic/stage1.pt
python -m experiments.synthetic.train --init runs/synthetic/stage1.pt --rel_mention --keep_base 1.0 --steps 1500 --p_pad 0.4 --p_rel 0.3 --p_plain 0.1 \
    --plain_cache runs/synthetic/cache_v6/plain_pool.pt --eval_every 250 --out runs/synthetic/v6.pt
```

This took about 35 minutes on a 14-core Mac. `checkpoints/archive/kgmem_adapter_v6.pt` is that output with `relm_raw_cos=30` set in its
config. This change is recorded in `meta["inference_changes"]`; see
[computed relations](../../explanation/strandedcpu.md). Stage 1 links held-out relation names at 0.995 / 1.00
through the mention path.

<a id="v7"></a>

## Train v7

This creates `runs/synthetic/cache_v7`, which the v8 and v9 recipes also use.

```sh
F="--rel_mention --relm_raw_cos 30 --relm_first --rel_halt --l2_keys"
python -m experiments.synthetic.pretrain_linkers cache --n_kg 400 --n_pad 400 --n_rel 500 --n_plain 100 --rel_key_pool last --rel_first_keys --out runs/synthetic/cache_v7
python -m experiments.synthetic.pretrain_linkers train --cache runs/synthetic/cache_v7 --epochs 4 $F --out runs/synthetic/stage1.pt
python -m experiments.synthetic.train --init runs/synthetic/stage1.pt $F --keep_base 1.0 --steps 1800 --p_pad 0.4 --p_rel 0.35 --p_plain 0.1 \
    --plain_cache runs/synthetic/cache_v7/plain_pool.pt --eval_every 300 --out runs/synthetic/v7.pt
```

This took about 40 minutes. `checkpoints/archive/kgmem_adapter_v7.pt` is that output with `relm_raw_cos=60, hop_order=0.4,
carry_fresh=0.3` set for inference (see [computed relations](../../explanation/strandedcpu.md); recorded in `meta["inference_changes"]`).

<a id="v8"></a>

## Retrain the v8 scratchpad

Build `runs/synthetic/cache_v7` with the v7 cache command first. This run starts from the shipped v7 adapter and freezes its graph path.

```sh
python -m experiments.synthetic.pretrain_linkers train --cache runs/synthetic/cache_v7 --init checkpoints/archive/kgmem_adapter_v7.pt --cont_keys --only_addr --epochs 8 --out runs/synthetic/addr_v8.pt
python -m experiments.synthetic.train --init runs/synthetic/addr_v8.pt --rel_mention --relm_raw_cos 30 --relm_first --rel_halt --l2_keys \
    --cont_keys --readout_gate --gate_pad_value --only_pad --keep_base 1.0 --steps 1200 --p_pad 0.4 --p_rel 0.35 \
    --p_plain 0.1 --plain_cache runs/synthetic/cache_v7/plain_pool.pt --eval_every 300 --out runs/synthetic/v8.pt
```

`runs/synthetic/cache_v7` is the v7 recipe's cache. The address map takes about a minute and the scratchpad about 15.
`checkpoints/archive/kgmem_adapter_v8.pt` is that output with v7's three inference settings and `carry_share=0.2` (recorded in
`meta["inference_changes"]`).

<a id="v9"></a>

## Train the v9 stages

Start from the shipped v8 adapter and reuse `runs/synthetic/cache_v7/plain_pool.pt`. Build the cache with the v7 command first.

```sh
F="--rel_mention --relm_raw_cos 60 --relm_first --rel_halt --l2_keys --cont_keys --readout_gate --gate_pad_value \
   --carry_share 0.2 --hop_order 0.4 --carry_fresh 0.3 --end_gate --halt_rel_aux 1.0 --halt_rel_same"
P="--keep_base 1.0 --p_pad 0.1 --p_plain 0.2 --plain_cache runs/synthetic/cache_v7/plain_pool.pt"
python -m experiments.synthetic.train --init checkpoints/archive/kgmem_adapter_v8.pt $F $P --train_only end_gate.,kg.halt_rel. --steps 600 --lr 3e-2 \
    --p_rel 0.45 --eval_every 300 --out runs/synthetic/v9a.pt
python -m experiments.synthetic.train --init runs/synthetic/v9a.pt $F $P --carry_commit --numeric_digits balanced \
    --train_only out_kg.,kg.val.,kg.vnorm.,copy_gate.,end_gate.,kg.halt_rel. --steps 800 --lr 2e-3 --p_rel 0.5 \
    --eval_every 400 --out runs/synthetic/v9b.pt
python -m experiments.synthetic.train --init runs/synthetic/v9b.pt $F --carry_commit --train_only pad.,out_pad.,pad_readout.,read_norm.,ro_gate. \
    --keep_base 1.0 --steps 3000 --lr 1e-3 --p_pad 0.6 --p_rel 0.15 --p_plain 0.15 \
    --plain_cache runs/synthetic/cache_v7/plain_pool.pt --eval_every 1000 --out runs/synthetic/v9.pt
```

`--train_only` takes parameter-name prefixes of the block; everything else stays as `--init` loaded it.
`--hop_order`, `--carry_fresh`, `--carry_share` and `--carry_commit` set the inference rules during training, so
training-time evaluations include those rules. `checkpoints/smollm2-v9.pt` is that output with `exact_link=0.9` and
`carry_rel=True` (recorded in `meta["inference_changes"]`). These post-training settings distinguish the shipped
checkpoint from the raw recipe output. See [the v9 experiment](adapter-development.md#v9) for the
decoding failures they address and the evaluation's CPU/GPU differences.

For current evaluation commands, see [Train and evaluate](../../how-to/train-adapters.md).
