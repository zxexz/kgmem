"""Prepare, train, and evaluate WebQSP on RoG-format per-question subgraphs.

  python -m experiments.qa.webqsp prep --data data/webqsp --out runs/webqsp/prep.pt
  python -m experiments.qa.webqsp keys --prep runs/webqsp/prep.pt --adapter checkpoints/mquake.pt --out runs/webqsp/keys.pt
  python -m experiments.qa.webqsp train --prep runs/webqsp/prep.pt --keys runs/webqsp/keys.pt --init checkpoints/mquake.pt --out runs/webqsp/adapter.pt
  python -m experiments.qa.webqsp eval --prep runs/webqsp/prep.pt --keys runs/webqsp/keys.pt --adapter runs/webqsp/adapter.pt

Add inverse edges and include the supplied topic entity in each prompt.
Training uses the union of a batch's subgraphs; evaluation loads one question's
subgraph at a time. Rank retrieved entities at the answer position for Hits@1.
Training paths are shortest paths to gold answers, selected for answer precision.
Unnamed m.* and g.* nodes share one encoded key.
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import time
import zlib

import numpy as np
import torch

from kgmem import MemConfig
from kgmem.graph import ExactScorer, GraphTensors, KnowledgeGraph
from kgmem.lora import add_lora, load_lora, lora_state
from kgmem.memlm import Batch, find_decoder_layers
from kgmem.tasks import _find_last, _pack
from experiments.qa.metaqa import build
from kgmem.paths import ensure_parent

UNNAMED = re.compile(r"^[mg]\.[0-9a-z_]+$")
SPLITS = ("train", "validation", "test")


def rel_name(r: str, inverse: bool) -> str:
    """'people.person.nationality' -> 'people person nationality' (inverse: 'reverse of people person nationality')."""
    s = r.replace(".", " ").replace("_", " ")
    return f"reverse of {s}" if inverse else s


def answer_path(tri, topic, answers, NR, H=2, cap=500000):
    """Find a shortest relation path to an answer, up to H hops.

    Among paths at that depth, maximize answer precision, then answer count.
    Inverse IDs are r + NR. Return (path, endpoints), or (None, None) if no path
    is found or a frontier without answers exceeds cap (path, entity) pairs.
    """
    adj = collections.defaultdict(list)
    for h, r, t in tri.tolist():
        adj[h].append((r, t)); adj[t].append((r + NR, h))
    ans, frontier = set(answers), {(): set(topic)}
    for _ in range(H):
        nxt = collections.defaultdict(set)
        for p, ents in frontier.items():
            for e in ents:
                for r, t in adj.get(e, ()):
                    nxt[p + (r,)].add(t)
        hits = [(p, ends) for p, ends in nxt.items() if ends & ans]
        if hits:
            p, ends = max(hits, key=lambda x: (len(x[1] & ans) / len(x[1]), len(x[1] & ans)))
            return list(p), ends
        if sum(map(len, nxt.values())) > cap:
            return None, None
        frontier = nxt
    return None, None


_PATH_JOBS = []


def _path_job(i):
    """Search a prep item in a forked worker: up to two WebQSP hops or three CWQ hops."""
    tri, topic, answers, NR, source = _PATH_JOBS[i]
    return answer_path(tri, topic, answers, NR, 2) if source == "webqsp" else \
        answer_path(tri, topic, answers, NR, 3, cap=100000)


def prep(args):
    import pyarrow.parquet as pq
    ents, ent_index, rels, rel_index = [], {}, [], {}

    def eid(n):
        if n not in ent_index:
            ent_index[n] = len(ents); ents.append(n)
        return ent_index[n]
    raw = {}

    def read(d, split, source, keep=lambda r: True):
        out = []
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(d), f"{split}-*.parquet"))):
            for b in pq.ParquetFile(f).iter_batches(batch_size=16):
                for r in b.to_pylist():
                    if not keep(r):
                        continue
                    tri = np.array([(eid(h), rel_index.setdefault(rr, len(rel_index)), eid(t)) for h, rr, t in r["graph"]],
                                   dtype=np.int32).reshape(-1, 3)
                    out.append(dict(id=r["id"], q=r["question"], topic=[eid(x) for x in r["q_entity"]],
                                    answers=[eid(x) for x in r["a_entity"]], tri=tri, source=source))
        return out
    for split in SPLITS:
        raw[split] = read(args.data, split, "webqsp")
    for d in [x for x in args.extra_train.split(",") if x]:
        # Exclude CWQ questions derived from WebQSP test IDs to avoid topic/path leakage.
        # --keep_webqtest retains them when reproducing the RoG training-data protocol.
        keep = (lambda r: True) if args.keep_webqtest else (lambda r: not r["id"].startswith("WebQTest"))
        extra = read(d, "train", os.path.basename(d.rstrip("/")), keep=keep)
        print(f"{d}: {len(extra)} more training questions", flush=True)
        raw["train"] += extra
    base = sorted(rel_index, key=rel_index.get)
    NR = len(base)
    rels = [rel_name(r, False) for r in base] + [rel_name(r, True) for r in base]
    t0, P = time.time(), dict(entities=ents, relations=rels, freebase_relations=base, NR=NR, splits={})
    global _PATH_JOBS
    for split in SPLITS:
        _PATH_JOBS = [] if split == "test" else [(it["tri"], it["topic"], it["answers"], NR, it["source"])
                                                  for it in raw[split]]
        if _PATH_JOBS:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(args.workers) as pool:
                found = pool.map(_path_job, range(len(_PATH_JOBS)), chunksize=16)
        else:
            found = [(None, None)] * len(raw[split])
        for it, (p, ends) in zip(raw[split], found):
            it["path"] = p
            it["path_answers"] = sorted(ends & set(it["answers"])) if p else []
        P["splits"][split] = raw[split]
        print(f"{split}: paths in {time.time() - t0:.0f}s", flush=True)
        n = len(raw[split])
        print(f"{split}: {n} questions" + ("" if split == "test" else
              f", {sum(it['path'] is not None for it in raw[split])} with an answer path of <= 2 hops"), flush=True)
    print(f"{len(ents)} entities ({sum(bool(UNNAMED.match(e)) for e in ents)} unnamed), {NR} relations, "
          f"paths in {time.time() - t0:.0f}s")
    torch.save(P, ensure_parent(args.out))


@torch.no_grad()
def keys(args):
    """Cache named-entity keys plus pooled and first-token relation keys."""
    P = torch.load(args.prep, weights_only=False)
    if args.adapter:
        sd = torch.load(args.adapter, map_location="cpu", weights_only=True)
        args.layer = sd["layer_idx"]
        cfg = MemConfig(**sd["cfg"])
    else:                                              # Build a new block at the requested layer.
        from experiments.qa import mquake
        cfg = mquake.mquake_cfg(None, rel_named=0.0)
    mlm, _ = build(args, cfg)
    named = [i for i, e in enumerate(P["entities"]) if not UNNAMED.match(e)]
    if args.adapter and "in_shift" in sd["block"]:     # Reuse the adapter's calibration for its key table.
        mlm.block.in_shift.copy_(sd["block"]["in_shift"]); mlm.block.in_scale.copy_(sd["block"]["in_scale"])
    elif args.standardize:                             # Calibrate before encoding keys.
        rng = random.Random(0)
        texts = [prompt(P, it) for it in rng.sample(P["splits"]["train"], 300)] + \
            [P["entities"][i] for i in rng.sample(named, 300)]
        mlm.calibrate(texts)
        print(f"calibrated: scale median {float(mlm.block.in_scale.median()):.3g}, "
              f"max {float(mlm.block.in_scale.max()):.3g}", flush=True)
    row = torch.zeros(len(P["entities"]), dtype=torch.long)          # row 0: the shared key of unnamed nodes
    row[named] = torch.arange(1, len(named) + 1)
    t0, chunks = time.time(), [mlm.encode_names(["unnamed node"], pool=mlm.cfg.key_pool).half().cpu()]
    for i in range(0, len(named), 65536):
        chunks.append(mlm.encode_names([P["entities"][j] for j in named[i:i + 65536]], pool=mlm.cfg.key_pool).half().cpu())
        print(f"{min(i + 65536, len(named))}/{len(named)} entity keys, {time.time() - t0:.0f}s", flush=True)
    out = dict(layer=mlm.layer_idx, key_pool=mlm.cfg.key_pool, ent=torch.cat(chunks), ent_row=row, model=args.model,
               in_shift=mlm.block.in_shift.cpu(), in_scale=mlm.block.in_scale.cpu(),
               rel=mlm.encode_names(P["relations"], pool=mlm.cfg.rel_key_pool).half().cpu(),
               rel_first=mlm.encode_names(P["relations"], pool="first").half().cpu())
    torch.save(out, ensure_parent(args.out))
    print(f"saved {args.out}: {out['ent'].shape[0]} entity keys, {out['rel'].shape[0]} relation keys")


class Graphs:
    """Load a union of question subgraphs, with inverse edges and cached name keys."""

    def __init__(self, mlm, P, K):
        self.mlm, self.P, self.K, self.NR = mlm, P, K, P["NR"]
        dev, dt = mlm.device, next(mlm.lm.parameters()).dtype
        self.rel_keys = K["rel"].to(dev, dt)
        self.rel_first = K["rel_first"].to(dev, dt)
        self.rel_alias = mlm._alias(P["relations"])
        self.tok_cache = {}
        self.unnamed = torch.tensor([bool(UNNAMED.match(e)) for e in P["entities"]])
        self.loaded_unnamed = None

    def name_tok(self, e):
        t = self.tok_cache.get(e)
        if t is None:
            t = self.tok_cache[e] = self.mlm.tokenize_name(self.P["entities"][e])[:12]
        return t

    def load(self, items):
        tri = np.concatenate([it["tri"] for it in items]).astype(np.int64)
        extra = [e for it in items for e in it["topic"] + it["answers"]]
        ents = np.unique(np.concatenate([tri[:, 0], tri[:, 2], np.array(extra, dtype=np.int64)]))
        h, t = np.searchsorted(ents, tri[:, 0]), np.searchsorted(ents, tri[:, 2])
        trip = np.concatenate([np.stack([h, tri[:, 1], t], 1), np.stack([t, tri[:, 1] + self.NR, h], 1)])
        trip = np.unique(trip, axis=0)
        kg = KnowledgeGraph([self.P["entities"][e] for e in ents.tolist()], self.P["relations"], [])
        kg.triples = trip.tolist()
        mlm, dev = self.mlm, self.mlm.device
        toks = [self.name_tok(e) for e in ents.tolist()]
        L = max(map(len, toks))
        name_tok = torch.full((len(toks), L), -1, dtype=torch.long)
        alias = {}
        for i, tk in enumerate(toks):
            name_tok[i, :len(tk)] = torch.tensor(tk)
            alias[tuple(tk)] = i if tuple(tk) not in alias else -1
        rows = self.K["ent_row"][torch.from_numpy(ents)]
        self.loaded_unnamed = self.unnamed[torch.from_numpy(ents)]
        mlm.ctx.graph = GraphTensors(
            ent_keys=self.K["ent"][rows].to(dev, self.rel_keys.dtype), rel_keys=self.rel_keys,
            name_tok=name_tok.to(dev), name_len=torch.tensor(list(map(len, toks)), device=dev),
            scorer=ExactScorer(kg, mlm.cfg.max_tails, dev), kg=kg, rel_keys_first=self.rel_first,
            alias=alias, rel_alias=self.rel_alias)
        return {int(g): i for i, g in enumerate(ents.tolist())}


def prompt(P, it):
    return f"Q: {it['q']} (topic: {P['entities'][it['topic'][0]]})\nA:"


def webqsp_batch(mlm, graphs, P, draw, rng, B=8):
    """Build one question per row against the batch's combined subgraph.

    Sample a gold answer reached by the supervised path and label the topic
    mention, relation path, and hop count.
    """
    tok, dev, H = mlm.tok, mlm.device, mlm.cfg.hops
    items = [draw() for _ in range(B)]
    loc = graphs.load(items)
    nl = tok("\n", add_special_tokens=False).input_ids
    examples, spans, aux = [], [], []
    for b, it in enumerate(items):
        p = tok(prompt(P, it), add_special_tokens=False).input_ids
        topic = it["topic"][0]
        mention = _find_last(p, mlm.tokenize_name(P["entities"][topic]))
        a = mlm.tokenize_name(P["entities"][rng.choice(it["path_answers"])])
        ids = list(mlm.bos) + p
        aux.append((b, len(ids) - 1, loc[topic], it["path"], len(it["path"]),
                    len(mlm.bos) + mention if mention >= 0 else -1))
        spans.append((b, len(ids), len(ids) + len(a), f"{len(it['path'])}-hop"))
        examples.append((ids + a + nl, [-100] * len(ids) + a + nl, None))
    ids, labels, attn = _pack(examples, tok.pad_token_id or 0, dev)
    T = ids.shape[1]
    rel = torch.full((len(aux), H), -100, dtype=torch.long)
    for i, x in enumerate(aux):
        rel[i, :len(x[3])] = torch.tensor(x[3], dtype=torch.long)
    none = torch.full((len(aux), H), -1, dtype=torch.long, device=dev)
    return Batch(ids, labels, attn, spans,
                 aux_idx=torch.tensor([b * T + t for (b, t, *_) in aux], device=dev),
                 aux_ent=torch.tensor([x[2] for x in aux], device=dev), aux_rel=rel.to(dev),
                 aux_hops=torch.tensor([x[4] - 1 for x in aux], device=dev),
                 aux_mention=torch.tensor([x[5] for x in aux], device=dev),
                 aux_rel_mention=none, aux_rel_span_start=none)


@torch.no_grad()
def hits_at_1(mlm, graphs, P, items, dump=None, named_only=True, topk=10):
    """Score the top retrieved entity against each question's gold answer IDs.

    Each question uses its own subgraph. named_only removes unnamed nodes before
    ranking. Questions without a topic entity count as misses.
    """
    tok, dev = mlm.tok, mlm.device
    was = mlm.block.training
    mlm.block.eval()
    hit = 0
    for it in items:
        if not it["topic"]:                    # RoG's data has no topic entity for a few questions: counted as misses
            if dump is not None:
                dump.write(json.dumps(dict(id=it["id"], q=it["q"], pred=None, hit=False)) + "\n")
            continue
        loc = graphs.load([it])
        ids = torch.tensor([mlm.bos + tok(prompt(P, it), add_special_tokens=False).input_ids], device=dev)
        mlm.forward(ids, keep=1)
        cid, cw = mlm.ctx.cand_ids[0, -1], mlm.ctx.cand_w[0, -1].float()
        dense = torch.zeros(len(loc), device=dev).scatter_add_(0, cid, cw)
        if named_only:
            dense[graphs.loaded_unnamed.to(dev)] = 0
        pred = int(dense.argmax()) if float(dense.max()) > 0 else -1
        ok = pred >= 0 and pred in {loc[a] for a in it["answers"] if a in loc}
        hit += ok
        if dump is not None:                   # Save ranked candidates for the reader.
            name = mlm.ctx.graph.kg.entities[pred] if pred >= 0 else None
            w, e = dense.topk(min(topk, len(loc)))
            top = [(mlm.ctx.graph.kg.entities[int(x)], round(float(v), 4)) for x, v in zip(e, w) if v > 0]
            dump.write(json.dumps(dict(id=it["id"], q=it["q"], pred=name, hit=bool(ok), top=top)) + "\n")
    mlm.block.train(was)
    return hit / max(1, len(items))


def in_fold(it, fold):
    """fold "K:k": whether the question is in the k-th of K folds (by qid hash); None/"" matches nothing."""
    if not fold:
        return False
    K, k = map(int, fold.split(":"))
    return zlib.crc32(it["id"].encode()) % K == k


def load_model(args, cfg):
    mlm, tok = build(args, cfg)
    P = torch.load(args.prep, weights_only=False)
    K = torch.load(args.keys, weights_only=False)
    assert K["layer"] == mlm.layer_idx and K["key_pool"] == mlm.cfg.key_pool, "keys were cached for another layer"
    assert K.get("model", args.model) == args.model, f"keys were cached for {K.get('model')}"
    if "in_shift" in K:                                # the standardisation the keys were encoded through
        mlm.block.in_shift.copy_(K["in_shift"]); mlm.block.in_scale.copy_(K["in_scale"])
    return mlm, P, Graphs(mlm, P, K)


def train(args):
    over = {"max_tails": args.max_tails, "k_cand": args.k_cand, "rel_named": 0.0, **json.loads(args.cfg)}
    init = torch.load(args.init, map_location="cpu", weights_only=True) if args.init else None
    if init:
        args.layer = init["layer_idx"]
        cfg = MemConfig(**{**init["cfg"], **over})
    else:                                              # a fresh block, at the layer the keys were cached for
        from experiments.qa import mquake
        args.layer = torch.load(args.keys, map_location="cpu", weights_only=False)["layer"]
        cfg = mquake.mquake_cfg(None, **over)
    mlm, P, graphs = load_model(args, cfg)
    if init:
        print("warm start:", mlm.load(args.init, strict=False))
    pool = [it for it in P["splits"]["train"] if it["path"] and len(it["path"]) <= cfg.hops and not in_fold(it, args.fold)]
    own = [it for it in pool if it.get("source", "webqsp") == "webqsp"]
    extra = [it for it in pool if it.get("source", "webqsp") != "webqsp"]
    dev_items = P["splits"]["validation"]
    print(f"{len(own)} WebQSP and {len(extra)} other training questions with an answer path, {len(dev_items)} dev",
          flush=True)
    rng = random.Random(args.seed)
    if extra and args.p_webqsp is not None:
        draw = lambda: rng.choice(own) if rng.random() < args.p_webqsp else rng.choice(extra)
    else:
        draw = lambda: rng.choice(pool)
    groups, lora = [dict(params=mlm.trainable_parameters(), lr=args.lr)], None
    if args.lora_r:                                    # Adapt backbone layers through the attach layer.
        lora = add_lora(find_decoder_layers(mlm.lm), mlm.layer_idx, args.lora_r, 2.0 * args.lora_r)
        groups.append(dict(params=[p for m in lora.values() for p in (m.A, m.B)], lr=args.lora_lr))
        print(f"LoRA r={args.lora_r} on layers 0..{mlm.layer_idx}: {len(lora)} linears, "
              f"{sum(p.numel() for p in groups[1]['params']) / 1e6:.1f}M parameters", flush=True)
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=args.steps,
                                                pct_start=0.05)

    def save(path, info):
        mlm.save(ensure_parent(path), info)
        if lora is not None:
            torch.save(dict(r=args.lora_r, alpha=2.0 * args.lora_r, upto=mlm.layer_idx, state=lora_state(lora)),
                       ensure_parent(path.replace(".pt", "_lora.pt")))
    hist, t0, best, skipped = [], time.time(), -1.0, 0
    meta = {"args": vars(args), "parent": init.get("meta") if init else None}
    for step in range(1, args.steps + 1):
        b = webqsp_batch(mlm, graphs, P, draw, rng, args.batch)
        loss, logs = mlm.loss(b, args.aux, keep_base=0.0)
        opt.zero_grad()
        if not torch.isfinite(loss):                   # Skip non-finite updates.
            skipped += 1
            sched.step()
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            skipped += 1
            opt.zero_grad(); sched.step()
            continue
        opt.step(); sched.step()
        hist.append(logs)
        if step % 50 == 0:
            keys_ = {k for h in hist for k in h}
            avg = {k: sum(h[k] for h in hist if k in h) / sum(k in h for h in hist) for k in sorted(keys_)}
            print(f"step {step:5d}  " + "  ".join(f"{k} {v:.3f}" for k, v in avg.items())
                  + f"  {(time.time() - t0) / step:.2f}s/step" + (f"  skipped {skipped}" if skipped else ""), flush=True)
            hist = []
        if step % args.eval_every == 0 or step == args.steps:
            res = round(hits_at_1(mlm, graphs, P, dev_items), 4)
            print(f"dev Hits@1 @ {step}: {res}", flush=True)
            save(args.out.replace(".pt", "_last.pt"), dict(meta, step=step, dev=res))
            if res > best:
                best = res
                save(args.out, dict(meta, step=step, dev=res))
    print("saved", args.out, "best dev Hits@1", best)


def evaluate(args):
    sd = torch.load(args.adapter, map_location="cpu", weights_only=True)
    args.layer = sd["layer_idx"]
    mlm, P, graphs = load_model(args, MemConfig(**{**sd["cfg"], **json.loads(args.cfg)}))
    mlm.load(args.adapter)
    lp = args.adapter.replace(".pt", "_lora.pt")
    if os.path.exists(lp):                             # the backbone's adapters, saved beside the block
        L = torch.load(lp, map_location="cpu", weights_only=True)
        load_lora(add_lora(find_decoder_layers(mlm.lm), L["upto"], L["r"], L["alpha"]), L["state"])
        print(f"LoRA r={L['r']} on layers 0..{L['upto']} loaded", flush=True)
    mlm.block.eval()
    res = {}
    for split in args.splits.split(","):
        items = P["splits"][split]
        if args.fold:                              # cross-fitting: only the held-out fold of this split
            items = [it for it in items if in_fold(it, args.fold)]
        items = items[:args.limit or None]
        t0 = time.time()
        with open(ensure_parent(f"{args.dump}.{split}.jsonl"), "w") if args.dump else open(os.devnull, "w") as f:
            res[split] = round(hits_at_1(mlm, graphs, P, items, dump=f if args.dump else None,
                                         named_only=not args.all_nodes, topk=args.dump_k), 4)
        print(f"{split}: Hits@1 {res[split]} over {len(items)} questions ({time.time() - t0:.0f}s)", flush=True)
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", default="data/webqsp")
    p.add_argument("--keep_webqtest", action="store_true", help='retain CWQ training questions derived from WebQSP test IDs, as in RoG training')
    p.add_argument("--extra_train", default="", help='comma-separated RoG-format directories whose train splits join training')
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out", default='runs/webqsp/prep.pt')
    for name in ("keys", "train", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--prep", default='runs/webqsp/prep.pt')
        s.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
        s.add_argument("--dtype", default="float32")
        s.add_argument("--cfg", default="{}", help="JSON overrides of the config")
    k = sub.choices["keys"]
    k.add_argument("--adapter", default="", help='adapter defining key layer, pooling, and calibration; empty builds a fresh block')
    k.add_argument("--layer", type=int, default=None)
    k.add_argument("--standardize", action="store_true", help='calibrate input standardization before encoding keys')
    k.add_argument("--out", default='runs/webqsp/keys.pt')
    t = sub.choices["train"]
    t.add_argument("--keys", default='runs/webqsp/keys.pt')
    t.add_argument("--init", default="", help='adapter to initialize from; empty builds a fresh block')
    t.add_argument("--steps", type=int, default=10000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--aux", type=float, default=1.0)
    t.add_argument("--p_webqsp", type=float, default=None, help='fraction of training draws from WebQSP; default follows dataset proportions')
    t.add_argument("--lora_r", type=int, default=0, help='positive rank enables backbone LoRA through layer L; cached graph keys remain fixed')
    t.add_argument("--lora_lr", type=float, default=1e-4)
    t.add_argument("--fold", default="", help="K:k: leave the k-th of K folds of the training questions out")
    t.add_argument("--max_tails", type=int, default=256)
    t.add_argument("--k_cand", type=int, default=64)
    t.add_argument("--eval_every", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default='runs/webqsp/adapter.pt')
    e = sub.choices["eval"]
    e.add_argument("--keys", default='runs/webqsp/keys.pt')
    e.add_argument("--adapter", required=True)
    e.add_argument("--splits", default="test")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--dump", default=None)
    e.add_argument("--dump_k", type=int, default=10, help="candidates kept per question in the dump")
    e.add_argument("--all_nodes", action="store_true", help='include unnamed nodes in answer ranking')
    e.add_argument("--fold", default="", help="K:k: score only the k-th of K folds of each split (cross-fitting)")
    a = ap.parse_args()
    dict(prep=prep, keys=keys, train=train, eval=evaluate)[a.cmd](a)


if __name__ == "__main__":
    main()
