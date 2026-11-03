# Recorded Gemma training fragments

[Experiments](../index.md) / [Appendices](index.md)

These fragments record the first and final training rounds. They require the model and hardware
conditions in the [Gemma guide](../../how-to/run-gemma4.md); intermediate artifacts are not included.

```sh
mkdir -p runs/gemma4
```

<a id="first-run"></a>

## Train the first adapter

```sh
G="--model google/gemma-4-26B-A4B-it --dtype bfloat16"
F="--rel_mention --relm_raw_cos 30 --relm_first --rel_halt --l2_keys --cont_keys"
S2="--readout_gate --gate_pad_value --carry_share 0.2 --end_gate --halt_rel_aux 1.0 --halt_rel_same --carry_commit \
    --numeric_digits balanced"
python -m experiments.synthetic.pretrain_linkers cache $G --layer 23 --n_kg 400 --n_pad 400 --n_rel 500 --n_plain 100 --rel_key_pool last \
    --rel_first_keys --out runs/gemma4/cache_g4                                                   # ~6 min
python -m experiments.synthetic.pretrain_linkers train --cache runs/gemma4/cache_g4 --epochs 4 $F --out runs/gemma4/stage1.pt    # 34 s/epoch, no weights loaded
python -m experiments.synthetic.train $G --layer 23 --init runs/gemma4/stage1.pt $F $S2 --keep_base 1.0 --steps 1500 --lr 1e-3 --p_pad 0.4 \
    --p_rel 0.35 --p_plain 0.1 --plain_cache runs/gemma4/cache_g4/plain_pool.pt --eval_every 300 --out runs/gemma4/first.pt   # 0.5-1.3 s/step
```

This produces the first adapter. The experiment set `exact_link=0.9` after training.
Later prose and halting passes produced the shipped checkpoint; see [Gemma results](gemma4.md).

<a id="round-6"></a>

## Resume the recorded round-6 experiment

This fragment requires the round-5 adapter `runs/gemma4/round5.pt`, the `F` variable from the first recipe, and
`runs/gemma4/plain.pt`. Those intermediate artifacts are not included, and the intermediate rounds do not have a
complete command recipe here. Use the shipped adapter for inference unless you already have them.

```sh
G="--model google/gemma-4-26B-A4B-it --dtype bfloat16 --layer 23"
S6="--readout_gate --gate_pad_value --carry_share 0.2 --carry_fresh 0.3 --end_gate --halt_rel_aux 1.0 --halt_rel_same \
    --carry_commit --numeric_digits balanced"
python -m experiments.synthetic.train $G --init runs/gemma4/round5.pt $F $S6 --train_only kg.halt.,kg.halt_rel.,copy_gate.,end_gate. --keep_base 1.0 \
    --steps 1200 --lr 3e-3 --p_pad 0.1 --p_rel 0.5 --p_plain 0.2 --plain_assign 300 --plain_cache runs/gemma4/plain.pt --out runs/gemma4/round6a.pt
python -m experiments.synthetic.train $G --init runs/gemma4/round6a.pt $F $S6 --train_only pad.,out_pad.,pad_readout.,read_norm.,ro_gate. --keep_base 1.0 \
    --steps 3000 --lr 1e-3 --p_pad 0.5 --p_rel 0.1 --p_plain 0.4 --plain_assign 300 --plain_cache runs/gemma4/plain.pt --out runs/gemma4/round6.pt
```

Both outputs were evaluated with `exact_link=0.9` set afterward. The shipped `checkpoints/gemma4.pt` is the
round-6 result; use the evaluation command at the top of this guide to evaluate that checkpoint. The
[Gemma report](gemma4.md) records the intermediate passes and the final comparison.
