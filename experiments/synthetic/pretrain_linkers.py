"""Cache attach-layer states and train the block's auxiliary losses from them.

The cache command runs the frozen decoder only through the attach layer.
The train command builds a meta-device decoder skeleton and trains the block
without loading base-model weights. Follow with train.py to train through the
LM head, including the copy gate and output projections.

  python -m experiments.synthetic.pretrain_linkers cache --model google/gemma-4-26B-A4B-it --dtype bfloat16 --n_kg 2000 --n_pad 2000 --out runs/synthetic/cache_g4
  python -m experiments.synthetic.pretrain_linkers train --cache runs/synthetic/cache_g4 --epochs 5 --out runs/synthetic/stage1.pt
  python -m experiments.synthetic.train --model google/gemma-4-26B-A4B-it --dtype bfloat16 --init runs/synthetic/stage1.pt --steps 1000
"""
import argparse
import dataclasses
import json
import os
import random
import time
from collections import defaultdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from kgmem import MemConfig, MemoryLM
from kgmem.graph import ExactScorer, GraphTensors
from kgmem.memlm import Batch
from kgmem.tasks import RelWorld, World, kg_batch, plain_batch, scratchpad_batch
from experiments.synthetic.train import plain_pool
from kgmem.paths import ensure_parent


def batch_to(b, device):
    if b is None:
        return None
    kw = {}
    for f in dataclasses.fields(b):
        v = getattr(b, f.name)
        kw[f.name] = v.to(device) if torch.is_tensor(v) else (batch_to(v, device) if isinstance(v, Batch) else v)
    return Batch(**kw)


# Cache generation
@torch.no_grad()
def cache(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 device_map="auto" if torch.cuda.is_available() else None).eval()
    mlm = MemoryLM(model, tok, MemConfig(rel_key_pool=args.rel_key_pool, relm_first=args.rel_first_keys),
                   layer_idx=args.layer)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"model": args.model, "layer_idx": mlm.layer_idx, "rel_key_pool": args.rel_key_pool},
              open(ensure_parent(f"{args.out}/meta.json"), "w"))
    plain = plain_pool(mlm, path=f"{args.out}/plain_pool.pt") if args.n_plain else None
    for split, seed, counts in [("train", 0, (args.n_kg, args.n_pad, args.n_rel, args.n_plain)),
                                ("val", 12345, tuple(max(8, n // 10) if n else 0 for n in
                                                     (args.n_kg, args.n_pad, args.n_rel, args.n_plain)))]:
        rng = random.Random(seed)
        names = "train" if split == "train" else "eval"            # val uses names (and relations) training never sees
        worlds, recs, shard, t0 = {}, [], 0, time.time()
        kinds = [k for k, n in zip(("kg", "pad", "rel", "plain"), counts) for _ in range(n)]
        rng.shuffle(kinds)
        last_world = None

        def keep(w):
            mlm.load_graph(w.kg)
            g = mlm.ctx.graph
            worlds[len(worlds)] = dict(kg=w.kg, ent_keys=g.ent_keys.half().cpu(), rel_keys=g.rel_keys.half().cpu(),
                                       name_tok=g.name_tok.cpu(), name_len=g.name_len.cpu(),
                                       rel_keys_first=None if g.rel_keys_first is None else g.rel_keys_first.half().cpu())
            return len(worlds) - 1

        for i, kind in enumerate(kinds):
            if kind == "kg":
                w = World(rng, names=names); wid = last_world = keep(w)
                b = kg_batch(mlm, w, rng, args.batch)
            elif kind == "rel":
                w = RelWorld(rng, names=names, relations=names, numeric=rng.random() < 0.5); wid = keep(w)
                b = kg_batch(mlm, w, rng, args.batch, p_two_hop=0.5)
            elif kind == "plain":                                  # gate supervision needs a graph loaded
                wid = last_world if last_world is not None else keep(World(rng, names=names))
                b = plain_batch(mlm, rng, plain, args.batch)
            else:
                wid, b = None, scratchpad_batch(mlm, rng, args.batch, names=names)
            h = mlm.hidden_at_layer(b.ids, b.attn).half().cpu()
            ph = mlm.hidden_at_layer(b.prefix.ids, b.prefix.attn).half().cpu() if b.prefix is not None else None
            recs.append(dict(kind=kind, world=wid, batch=batch_to(b, "cpu"), h=h, prefix_h=ph))
            if len(recs) == args.shard or i == len(kinds) - 1:
                torch.save(recs, ensure_parent(f"{args.out}/{split}_{shard:04d}.pt")); recs, shard = [], shard + 1
            if i % 50 == 0:
                print(f"{split} {i}/{len(kinds)}  {time.time() - t0:.0f}s", flush=True)
        torch.save(worlds, ensure_parent(f"{args.out}/{split}_worlds.pt"))
    print("cached to", args.out)


# Cached-state training
def load_split(path, split):
    files = sorted(f for f in os.listdir(path) if f.startswith(split + "_") and f[-7:-3].isdigit())
    recs = [r for f in files for r in torch.load(f"{path}/{f}", weights_only=False)]
    return recs, torch.load(f"{path}/{split}_worlds.pt", weights_only=False)


def graph_for(mlm, worlds, wid, cache_):
    if wid not in cache_:
        w = worlds[wid]
        cache_[wid] = GraphTensors(ent_keys=w["ent_keys"].float().to(mlm.device),
                                   rel_keys=w["rel_keys"].float().to(mlm.device),
                                   name_tok=w["name_tok"].to(mlm.device), name_len=w["name_len"].to(mlm.device),
                                   scorer=ExactScorer(w["kg"], mlm.cfg.max_tails, mlm.device), kg=w["kg"],
                                   rel_keys_first=(w["rel_keys_first"].float().to(mlm.device)
                                                   if w.get("rel_keys_first") is not None else None))
    return cache_[wid]


def run_record(mlm, r, worlds, gcache, train=True):
    b = batch_to(r["batch"], mlm.device)
    mlm.ctx.graph = graph_for(mlm, worlds, r["world"], gcache) if r["world"] is not None else None
    h = r["h"].float().to(mlm.device)
    ph = r["prefix_h"].float().to(mlm.device) if r["prefix_h"] is not None else None
    with torch.set_grad_enabled(train):
        loss, logs = mlm.stage1_loss(b, h, ph)
    if r["kind"] in ("rel", "plain"):                       # report relation linking on RelWorlds separately
        logs = {f"{r['kind']}_{k}": v for k, v in logs.items()}
    if r["kind"] == "pad":                                  # Additional address metrics from scratchpad activations.
        cur, pre = mlm.block.pad.last_aux, mlm._prefix_aux
        logs["ptr_r_acc"] = (cur["ptr_r"][b.ptr_r[:, 0], b.ptr_r[:, 1]].argmax(-1) == b.ptr_r[:, 2]).float().mean().item()
        logs["ptr_w_acc"] = (pre["ptr_w"][b.prefix.ptr_w[:, 0], b.prefix.ptr_w[:, 1]].argmax(-1)
                             == b.prefix.ptr_w[:, 2]).float().mean().item()
        p, q = b.addr_pair, b.prefix.addr_neg
        if mlm.cfg.cont_keys:   # Measure read/write alignment with unit-vector addresses.
            last = defaultdict(set)                     # Distinct variables should have nonparallel addresses.
            for x, y1, y2 in q.tolist():
                last[x].update((y1, y2))
            ok = []
            for x, t, v in p.tolist():
                ys = sorted(last[x] | {v})
                ok.append(ys[int((pre["kw"][x, ys] @ cur["kr"][x, t]).argmax())] == v)
            logs["slot_agree"] = sum(ok) / len(ok)
            logs["slot_apart"] = ((pre["kw"][q[:, 0], q[:, 1]] * pre["kw"][q[:, 0], q[:, 2]]).sum(-1).abs() < 0.5).float().mean().item()
            return loss, logs
        logs["slot_agree"] = (cur["kr"][p[:, 0], p[:, 1]].argmax(-1) == pre["kw"][p[:, 0], p[:, 2]].argmax(-1)).float().mean().item()
        # Read/write agreement alone does not detect all-variable address collapse.
        logs["slot_apart"] = (pre["kw"][q[:, 0], q[:, 1]].argmax(-1) != pre["kw"][q[:, 0], q[:, 2]].argmax(-1)).float().mean().item()
    return loss, logs


def train(args):
    torch.manual_seed(args.seed)                             # Separate parameter initialization from deterministic record ordering.
    meta = json.load(open(f"{args.cache}/meta.json"))
    tok = AutoTokenizer.from_pretrained(meta["model"])
    with torch.device("meta"):                               # Instantiate a decoder skeleton without backbone weights.
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(meta["model"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MemConfig(read_norm=True, hard_addressing=args.hard, rel_mention=args.rel_mention,
                    rel_key_pool=meta.get("rel_key_pool", "mean"),           # Use the pooling mode recorded with cached relation keys.
                    relm_raw_cos=args.relm_raw_cos, relm_first=args.relm_first, rel_halt=args.rel_halt,
                    l2_keys=args.l2_keys, cont_keys=args.cont_keys)
    if args.only_addr and args.init:                         # Preserve other saved architecture settings.
        cfg = MemConfig(**dict(torch.load(args.init, map_location="cpu")["cfg"], cont_keys=args.cont_keys))
    mlm = MemoryLM(model, tok, cfg, layer_idx=meta["layer_idx"], device=dev)
    if args.init:
        print(mlm.load(args.init, strict=False))
    tr, tr_worlds = load_split(args.cache, "train")
    va, va_worlds = load_split(args.cache, "val")
    if args.only_addr:      # retrain the scratchpad's address map alone (e.g. to switch an adapter to cont_keys)
        tr, va = [r for r in tr if r["kind"] == "pad"], [r for r in va if r["kind"] == "pad"]
        for n, p in mlm.block.named_parameters():
            p.requires_grad_(n in ("pad.addr.weight", "pad.addr.bias", "pad.addr_scale"))
    print(f"{len(tr)} train / {len(va)} val records; block {sum(p.numel() for p in mlm.trainable_parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(mlm.trainable_parameters(), lr=args.lr)
    sched = (torch.optim.lr_scheduler.ConstantLR(opt, 1.0) if args.only_addr else
             torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(tr), pct_start=0.05))
    rng, gtr, gva = random.Random(0), {}, {}

    def evaluate():
        mlm.block.eval(); acc = defaultdict(list)
        for r in va:
            _, logs = run_record(mlm, r, va_worlds, gva, train=False)
            for k, v in logs.items():
                if k.endswith("acc") or k.startswith("slot_"):
                    acc[k].append(v)
        mlm.block.train()
        return {k: round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}

    res = evaluate()
    print("val before:", res, flush=True)
    for ep in range(args.epochs):
        rng.shuffle(tr); t0 = time.time(); tot = 0.0
        for r in tr:
            loss, _ = run_record(mlm, r, tr_worlds, gtr)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mlm.trainable_parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        res = evaluate()
        print(f"epoch {ep}: loss {tot/len(tr):.3f}  {time.time()-t0:.0f}s  val {res}", flush=True)
    parent = torch.load(args.init, map_location="cpu").get("meta") if args.init else None
    mlm.save(ensure_parent(args.out), {"args": vars(args), "stage": 1, "cache": meta, "parent": parent, "eval": res})
    print("saved", args.out, "-> next: python -m experiments.synthetic.train --init", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M"); c.add_argument("--dtype", default="float32")
    c.add_argument("--layer", type=int, default=None); c.add_argument("--batch", type=int, default=8)
    c.add_argument("--n_kg", type=int, default=2000); c.add_argument("--n_pad", type=int, default=2000)
    c.add_argument("--shard", type=int, default=100); c.add_argument("--out", default='runs/synthetic/cache')
    c.add_argument("--n_rel", type=int, default=0); c.add_argument("--n_plain", type=int, default=0)
    c.add_argument("--rel_key_pool", default="mean", help="'last' for --rel_mention adapters")
    c.add_argument("--rel_first_keys", action="store_true", help="also cache first-token relation keys")
    t = sub.add_parser("train")
    t.add_argument("--cache", default='runs/synthetic/cache'); t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--init", default=None)
    t.add_argument("--out", default='runs/synthetic/stage1.pt')
    t.add_argument("--hard", action="store_true", help="straight-through one-hot addressing (as train.py --hard)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--rel_mention", action="store_true", help='mix query-based relation scores with relation-name token scores')
    t.add_argument("--relm_raw_cos", type=float, default=0.0)
    t.add_argument("--relm_first", action="store_true")
    t.add_argument("--rel_halt", action="store_true")
    t.add_argument("--l2_keys", action="store_true")
    t.add_argument("--cont_keys", action="store_true", help="unit-vector scratchpad keys instead of a slot softmax")
    t.add_argument("--only_addr", action="store_true", help='with --init, train the scratchpad address map on cached scratchpad records only')
    a = ap.parse_args()
    cache(a) if a.cmd == "cache" else train(a)
