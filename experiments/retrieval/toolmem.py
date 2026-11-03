"""Train a relation decoder to retrieve tools from a catalogue.

  python -m experiments.retrieval.toolmem --dir OUT_OF_toolret_prep --model MODEL --dtype bfloat16 --layer 41 --backbone-loader backbone_loader.py --steps 6000 --save tm.pt --out tm.json --features tm_features.jsonl

Each tools.jsonl entry becomes a relation keyed by its name and description; one
placeholder entity has all tool edges. Each forward reads a request's task and
query without candidate text. Gold tools supervise hops in their supplied order,
up to --hops. A tool's score is its maximum probability over hops.

Train on train.jsonl and score heldout.jsonl from other sources. Report catalogue
Recall@k and first-gold ranks within BM25's 32 candidates. --features writes a score
per (query, candidate). --swap repeats evaluation with held-out sources' tools
masked, including comparisons on training-source rows."""
from kgmem.paths import ensure_parent

import argparse
import collections
import json
import os
import random
import time

import torch

from experiments.retrieval import funclib
from kgmem import MemConfig
from kgmem.graph import KnowledgeGraph
from kgmem.memlm import Batch

KS = (1, 5, 10, 32, 100)


def key_text(doc, limit=400):
    try:
        j = json.loads(doc)
        name = j.get("name") or j.get("api_name") or j.get("tool_name") or ""
        desc = j.get("description") or j.get("api_description") or ""
        t = f"{name}: {desc}".strip(": ")
    except Exception:
        t = doc
    return (t or doc)[:limit]


def read_jsonl(p, ids=None):
    out = []
    for l in open(p):
        r = json.loads(l)
        if ids is None or r.get("qid", r.get("id")) in ids:
            out.append(r)
    return out


def request(r):
    return f"Task: {r['task']}\nRequest: {r['query']}\nTools:"


def row_batch(mlm, r, index, H):
    tok, dev = mlm.tok, mlm.device
    ids = list(mlm.bos) + tok(request(r), add_special_tokens=False).input_ids[-1000:]
    gold = [index[g] for g in r.get("gold", []) if g in index][:H]
    rel = torch.full((1, H), -100, dtype=torch.long)
    if gold:
        rel[0, :len(gold)] = torch.tensor(gold)
    t = torch.tensor([ids], device=dev)
    none = torch.full((1, H), -1, dtype=torch.long, device=dev)
    return Batch(t, torch.full_like(t, -100), torch.ones_like(t), [], aux_idx=torch.tensor([len(ids) - 1], device=dev),
                 aux_ent=torch.zeros(1, dtype=torch.long, device=dev), aux_rel=rel.to(dev),
                 aux_hops=torch.tensor([max(len(gold), 1) - 1], device=dev),
                 aux_mention=torch.tensor([-1], device=dev), aux_rel_mention=none, aux_rel_span_start=none), gold


@torch.no_grad()
def tool_scores(mlm, r, H):
    b, _ = row_batch(mlm, r, {}, H)
    mlm.forward(b.ids, b.attn, aux_idx=b.aux_idx, aux_mention=b.aux_mention, keep=1)
    return torch.stack([x[0].float().softmax(-1) for x in mlm.ctx.kg_aux["rel"]]).max(0).values      # (R,)


def gate_features(s, cids, trained):
    """Gate features from catalogue scores s and candidate ids cids.

    cids uses -1 for absent tools; trained lists tools seen as training gold."""
    top2 = s.topk(20)
    cs = [float(s[i]) if i >= 0 else 0.0 for i in cids]
    p = top2.values.clamp(min=1e-9)
    p = p / p.sum()
    return dict(max_cand=max(cs) if cs else 0.0, max_all=float(top2.values[0]),
                top_in_cand=float(int(top2.indices[0]) in set(cids)),
                margin=float(top2.values[0] - top2.values[1]),
                trained_share=sum(i in trained for i in cids) / max(1, len(cids)),
                entropy=float(-(p * p.log()).sum()))


def score_rows(mlm, rows, index, H, mask=None, features=None, gate=None, trained=frozenset(), counts=None):
    """Measure catalogue Recall@k and first-gold ranks within BM25 candidates."""
    rec = {f"recall@{k}": 0.0 for k in KS}
    cand = {"kgmem first-gold rank among BM25 32": [], "BM25 first-gold rank among its 32": []}
    for r in rows:
        s = tool_scores(mlm, r, H)
        if mask is not None:
            s = torch.where(mask, s, torch.full_like(s, -1.0))
        gold = {index[g] for g in r["gold"] if g in index}
        order = s.topk(max(KS)).indices.tolist()
        for k in KS:
            rec[f"recall@{k}"] += len(gold & set(order[:k])) / max(1, len(gold))
        if "candidates" in r:
            cids = [index.get(c["tool_id"], -1) for c in r["candidates"]]
            cs = [float(s[i]) if i >= 0 else -1.0 for i in cids]
            rk = sorted(range(len(cids)), key=lambda j: -cs[j])
            g = [j for j, i in enumerate(cids) if i in gold]
            if g:
                cand["kgmem first-gold rank among BM25 32"].append(1 + min(rk.index(j) for j in g))
                cand["BM25 first-gold rank among its 32"].append(1 + min(g))
            if features is not None:
                for c, v in zip(r["candidates"], cs):
                    features.write(json.dumps(dict(qid=r["qid"], tool_id=c["tool_id"], kgmem=round(v, 6),
                                                   train_count=(counts or {}).get(c["tool_id"], 0))) + "\n")
            if gate is not None:
                best = max(range(len(cids)), key=lambda j: cs[j])
                gate.write(json.dumps(dict(qid=r["qid"], source=r.get("source"), pool=r.get("pool"),
                                           label=int(cids[best] in gold), **gate_features(s, cids, trained))) + "\n")
    n = max(1, len(rows))
    out = {k: round(100 * v / n, 2) for k, v in rec.items()} | {"n": len(rows)}
    for k, v in cand.items():
        if v:
            out[k.replace("rank", "MRR")] = round(sum(1 / x for x in v) / len(v), 4)
            out[k.replace("rank", "top-1")] = round(100 * sum(x == 1 for x in v) / len(v), 2)
            out[k.replace("rank", "top-5")] = round(100 * sum(x <= 5 for x in v) / len(v), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--backbone-loader", dest="backbone_loader", default="",
                    help='optional Python loader exporting register() -> attention name and require_policy(text_config, policy); uses CUDA bf16')
    ap.add_argument("--policy", default="B")
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--dev_n", type=int, default=400)
    ap.add_argument("--train_ids", default="", help="file of training qids that passed the decontamination gate")
    ap.add_argument("--heldout_ids", default="", help="file of held-out qids that passed decontamination")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--rows", default="", help="toolret_loso.py's rows.jsonl: train on its 'train' rows of --train_sources")
    ap.add_argument("--train_sources", default="toolbench,apigen,toolACE")
    ap.add_argument("--pools", default="", help="comma-separated pool files to score (toolret_loso's gatefit/eval)")
    ap.add_argument("--gate_out", default="", help="write per-query gate features and labels")
    ap.add_argument("--eval_only", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--features", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = lambda p: set(open(p).read().split()) if p else None
    tools = read_jsonl(os.path.join(a.dir, "tools.jsonl"))
    index = {t["id"]: i for i, t in enumerate(tools)}
    if a.rows:                                       # Restrict training to selected sources and pool=train.
        srcs = set(a.train_sources.split(","))
        train = [r for r in read_jsonl(a.rows, ids(a.train_ids)) if r["pool"] == "train" and r["source"] in srcs]
        held = [r for p in a.pools.split(",") for r in read_jsonl(p, ids(a.train_ids))][:a.limit or None]
    else:
        train = read_jsonl(os.path.join(a.dir, "train.jsonl"), ids(a.train_ids))
        held = read_jsonl(os.path.join(a.dir, "heldout.jsonl"), ids(a.heldout_ids))[:a.limit or None]
    rng = random.Random(a.seed)
    rng.shuffle(train)
    dev, train = train[:a.dev_n], train[a.dev_n:]          # Dev uses training-source rows for checkpoint selection.
    if a.eval_only:
        sd = torch.load(a.eval_only, map_location="cpu", weights_only=True)
        a.layer = sd["layer_idx"]
        cfg = MemConfig(**sd["cfg"])
    else:
        from experiments.qa import mquake
        # One tail per tool; limit candidate-key gathers at the large backbone width.
        cfg = mquake.mquake_cfg(None, hops=a.hops, rel_named=0.0, rel_exists=False, kg_gate_loss=0.0,
                                max_tails=1, k_cand=4)
    mlm = funclib.build_model(a, cfg)
    mlm.question_only, mlm.unpadded = True, bool(a.backbone_loader)
    if a.eval_only:
        mlm.load(a.eval_only)
    else:
        mlm.calibrate([request(r) for r in rng.sample(train, 300)] + [key_text(t["doc"]) for t in tools[:300]])
    t0 = time.time()
    keys = [key_text(t["doc"]) for t in tools]
    funclib.Keys(mlm, [dict(key=k) for k in keys], first=not cfg.rel_decoder).load(
        KnowledgeGraph(["value"], keys, [(0, r, 0) for r in range(len(keys))]))
    print(f"{len(tools)} tools keyed in {time.time() - t0:.0f}s; {len(train)} train, {len(dev)} dev, "
          f"{len(held)} held-out rows", flush=True)
    H, res = a.hops, {"args": vars(a)}
    if not a.eval_only:
        params = mlm.trainable_parameters()
        opt = torch.optim.AdamW(params, a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
        best, hist, t0 = -1.0, [], time.time()
        for step in range(1, a.steps + 1):
            logs, bad = {}, False
            opt.zero_grad()
            for r in (rng.choice(train) for _ in range(a.batch)):
                b, _ = row_batch(mlm, r, index, H)
                mlm.forward(b.ids, b.attn, aux_idx=b.aux_idx, aux_mention=b.aux_mention, keep=1)
                l, lg = mlm.aux_loss(b, 1.0)
                if not torch.isfinite(l):
                    bad = True
                    break
                (l / a.batch).backward()            # Release each row's autograd graph before the next key-table gather.
                for k, v in lg.items():
                    logs[k] = logs.get(k, 0.0) + v / a.batch
            if bad:
                opt.zero_grad(); sched.step()
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            hist.append(logs)
            if step % 200 == 0:
                ks = {k for h in hist for k in h}
                avg = {k: sum(h[k] for h in hist if k in h) / sum(k in h for h in hist) for k in ks}
                print(f"step {step} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(avg.items()))
                      + f" {(time.time() - t0) / step:.2f}s/step", flush=True)
                hist = []
            if step % a.eval_every == 0 or step == a.steps:
                d = score_rows(mlm, dev, index, H)
                print(f"dev @ {step}: {json.dumps(d)}", flush=True)
                if a.save and d["recall@10"] > best:
                    best = d["recall@10"]
                    mlm.save(ensure_parent(a.save), {"args": vars(a), "step": step, "dev": d})
        if a.save:
            mlm.load(a.save)
    trained = {index[g] for r in train for g in r["gold"] if g in index}
    counts = collections.Counter(g for r in train for g in r["gold"])
    with open(ensure_parent(a.features), "w") if a.features else open(os.devnull, "w") as f, \
            open(ensure_parent(a.gate_out), "w") if a.gate_out else open(os.devnull, "w") as g:
        res["heldout"] = score_rows(mlm, held, index, H, features=f if a.features else None,
                                    gate=g if a.gate_out else None, trained=trained, counts=counts)
    print("held out:", json.dumps(res["heldout"]), flush=True)
    for src in sorted({r["source"] for r in held}):
        res[f"heldout [{src}]"] = score_rows(mlm, [r for r in held if r["source"] == src], index, H)
    if a.swap:
        held_srcs = {r["source"] for r in held}
        keep = torch.tensor([t["source"] not in held_srcs for t in tools], device=mlm.device)
        res["swap: held-out rows, their sources' tools unloaded"] = score_rows(mlm, held[:500], index, H, mask=keep)
        res["swap: training-source rows, all tools loaded"] = score_rows(mlm, dev, index, H)
        res["swap: training-source rows, held-out sources' tools unloaded"] = score_rows(mlm, dev, index, H, mask=keep)
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1), flush=True)
    json.dump(res, open(ensure_parent(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
