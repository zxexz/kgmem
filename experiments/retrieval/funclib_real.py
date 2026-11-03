"""Retrieve chains from extracted typed function-library files.

  python -m experiments.retrieval.funclib_real --dir LIBDIR --n_lib 200 --model MODEL --dtype bfloat16 --layer 41 --backbone-loader backbone_loader.py --lr 3e-4 --steps 6000 --save out.pt --out out.json

Keys contain signatures and formula descriptions, as in the BM25 baseline.
Supervision follows evaluation order: innermost first, repeats retained. Training
excludes empty chains and, by default, chains longer than --hops. --max_chain can
raise that limit; supervision still uses only the first --hops calls. One
placeholder entity has all function edges; this experiment scores relation choices.
Rows contain one unpadded question without its answer. A twentieth of training keys
form the dev set; the eval split is scored afterward.

On library_retrieval_eval_N{n}, all@k requires every gold function in the top k;
per@k measures the share retrieved. Functions rank by maximum probability over hops.
chain@k ranks chains by P(length) times hop probabilities, conditioned on previous
decoder choices. The report includes bm25_top on the same items and separates
seen_by_final_v1 items. For --eval_only, reuse the training model and attention flags."""
from kgmem.paths import ensure_parent

import argparse
import ast
import json
import os
import random
import time
import zlib

import torch

from experiments.retrieval import funclib
from kgmem import MemConfig
from kgmem.graph import KnowledgeGraph
from kgmem.memlm import Batch

KS = (1, 3, 5, 10, 20)


def load(d, n):
    lib = json.load(open(os.path.join(d, "library.json")))["functions"][:n]
    keys = [f"{f['signature']}: {f['description']}" for f in lib]
    index = {f["name"]: i for i, f in enumerate(lib)}
    rows = [json.loads(l) for l in open(os.path.join(d, f"library_calls_N{n}.jsonl"))]
    ev = [json.loads(l) for l in open(os.path.join(d, f"library_retrieval_eval_N{n}.jsonl"))]
    return lib, keys, index, rows, ev


def graph(keys):
    return KnowledgeGraph(["value"], list(keys), [(0, r, 0) for r in range(len(keys))])


def row_batch(mlm, question, chain, index, H):
    tok, dev = mlm.tok, mlm.device
    ids = list(mlm.bos) + tok(f"Q: {question}\nA:", add_special_tokens=False).input_ids
    rel = torch.full((1, H), -100, dtype=torch.long)
    if chain:
        rel[0, :len(chain)] = torch.tensor([index[c] for c in chain])
    t = torch.tensor([ids], device=dev)
    none = torch.full((1, H), -1, dtype=torch.long, device=dev)
    return Batch(t, torch.full_like(t, -100), torch.ones_like(t), [], aux_idx=torch.tensor([len(ids) - 1], device=dev),
                 aux_ent=torch.zeros(1, dtype=torch.long, device=dev), aux_rel=rel.to(dev),
                 aux_hops=torch.tensor([max(len(chain), 1) - 1], device=dev),
                 aux_mention=torch.tensor([-1], device=dev), aux_rel_mention=none, aux_rel_span_start=none)


@torch.no_grad()
def decode(mlm, question, H, beam=5, width=50):
    """Rank functions by maximum hop probability and chains by joint probability.

    Each hop extends the best width partial chains with its best beam functions."""
    b = row_batch(mlm, question, [], {}, H)
    mlm.forward(b.ids, b.attn, aux_idx=b.aux_idx, aux_mention=b.aux_mention)
    aux = mlm.ctx.kg_aux
    p = torch.stack([r[0].float().softmax(-1) for r in aux["rel"]])          # (H, R)
    halt = aux["halt"][0].float().softmax(-1)
    funcs = p.max(0).values.argsort(descending=True).tolist()
    tops = [p[h].topk(min(beam, p.shape[1])) for h in range(H)]
    chains, partial = [], [((), 1.0)]
    for h in range(H):
        partial = sorted([(c + (int(i),), s * float(v)) for c, s in partial for v, i in zip(*tops[h])],
                         key=lambda x: -x[1])[:width]
        chains += [(c, s * float(halt[h])) for c, s in partial]
    chains.sort(key=lambda x: -x[1])
    return funcs, [c for c, _ in chains[:max(KS)]]


def score(ranked_funcs, ranked_chains, gold_funcs, gold_chain):
    out = {}
    for k in KS:
        top = set(ranked_funcs[:k])
        out[f"all@{k}"] = float(set(gold_funcs) <= top)
        out[f"per@{k}"] = sum(g in top for g in gold_funcs) / len(gold_funcs)
        if ranked_chains is not None:
            out[f"chain@{k}"] = float(tuple(gold_chain) in ranked_chains[:k])
    return out


def mean(rs):
    return {k: round(100 * sum(r[k] for r in rs) / len(rs), 1) for k in rs[0]} | {"n": len(rs)} if rs else {"n": 0}


def evaluate(mlm, ev, index, H, limit=0, trained=None):
    names = {i: n for n, i in index.items()}
    was = mlm.block.training
    mlm.block.eval()
    k_rows, b_rows, t0 = [], [], time.time()
    for r in ev[:limit or None]:
        funcs, chains = decode(mlm, r["query"], H)
        gf = [index[g] for g in r["gold_functions"]]
        gc = [index[g] for g in r["gold_chain"]]
        k_rows.append((r, score(funcs, chains, gf, gc)))
        bm = r["bm25_top"] if isinstance(r["bm25_top"], list) else ast.literal_eval(r["bm25_top"])
        b_rows.append((r, score([index.get(x, -1) for x in bm], None, gf, gc)))
    mlm.block.train(was)
    out = {}
    for name, rows in (("kgmem", k_rows), ("bm25", b_rows)):
        subsets = [("all", lambda r: True), ("not seen by final v1", lambda r: not r["seen_by_final_v1"]),
                   ("seen by final v1", lambda r: r["seen_by_final_v1"])]
        if trained is not None:                    # Functions appearing in training supervision.
            subsets += [("every gold function trained", lambda r: set(r["gold_functions"]) <= trained),
                        ("a gold function untrained", lambda r: not set(r["gold_functions"]) <= trained)]
        for sub, keep in subsets:
            out[f"{name} [{sub}]"] = mean([s for r, s in rows if keep(r)])
    out["kgmem per family"] = {fam: mean([s for r, s in k_rows if r["family"] == fam])
                               for fam in sorted({r["family"] for r, _ in k_rows})}
    out["seconds"] = round(time.time() - t0)
    return out


def build(args):
    if args.eval_only:
        sd = torch.load(args.eval_only, map_location="cpu", weights_only=True)
        args.layer = sd["layer_idx"]
        cfg = MemConfig(**sd["cfg"])
    else:
        from experiments.qa import mquake
        cfg = mquake.mquake_cfg(None, hops=args.hops, rel_named=0.0, rel_exists=False, kg_gate_loss=0.0,
                                **json.loads(args.cfg))
    mlm = funclib.build_model(args, cfg)
    mlm.question_only, mlm.unpadded = True, bool(args.backbone_loader)
    if args.eval_only:
        mlm.load(args.eval_only)
    return mlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n_lib", type=int, default=200)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--backbone-loader", dest="backbone_loader", default="",
                    help='optional Python loader exporting register() -> attention name and require_policy(text_config, policy); uses CUDA bf16')
    ap.add_argument("--policy", default="B")
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--max_chain", type=int, default=0, help="train on chains up to this long (default --hops; "
                    "longer ones supervise their first --hops calls); 0 = --hops")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--dev_n", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="score only the first n eval items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_only", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lib, keys, index, rows, ev = load(a.dir, a.n_lib)
    H = a.hops
    mlm = build(a)
    cap = a.max_chain or H                           # Truncate supervision at H even when max_chain is larger.
    rows = [dict(r, chain=r["chain"][:H]) for r in rows
            if 1 <= len(r["chain"]) <= cap and all(c in index for c in r["chain"])]
    is_dev = lambda r: zlib.crc32(r["split_key"].encode()) % 20 == 0
    train, dev = [r for r in rows if not is_dev(r)], [r for r in rows if is_dev(r)]
    rng = random.Random(a.seed)
    dev = rng.sample(dev, min(a.dev_n, len(dev)))
    dev_ev = [dict(query=r["question"], gold_functions=list(dict.fromkeys(r["chain"])), gold_chain=r["chain"],
                   bm25_top=[], seen_by_final_v1=False, family=r["family"]) for r in dev]
    if not a.eval_only:                              # Calibrate before encoding keys.
        mlm.calibrate([f"Q: {r['question']}\nA:" for r in rng.sample(train, 300)] + keys[:200])
    kg = graph(keys)
    loader = funclib.Keys(mlm, [dict(key=k) for k in keys])
    loader.load(kg)
    print(f"{len(lib)} functions; {len(train)} training and {len(dev)} dev questions with a 1-{H} call chain; "
          f"{len(ev)} eval items", flush=True)
    trained = {c for r in train for c in r["chain"]}
    print(f"{len(trained)} of {len(lib)} functions have a training question", flush=True)
    res = {"args": vars(a), "trained_functions": len(trained)}
    if not a.eval_only:
        params = mlm.trainable_parameters()
        opt = torch.optim.AdamW(params, a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
        best, t0, hist = -1.0, time.time(), []
        for step in range(1, a.steps + 1):
            loss, logs = 0.0, {}
            for r in (rng.choice(train) for _ in range(a.batch)):
                b = row_batch(mlm, r["question"], r["chain"], index, H)
                mlm.forward(b.ids, b.attn, aux_idx=b.aux_idx, aux_mention=b.aux_mention)
                l, lg = mlm.aux_loss(b, 1.0)
                loss = loss + l / a.batch
                for k, v in lg.items():
                    logs[k] = logs.get(k, 0.0) + v / a.batch
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            hist.append(logs)
            if step % 200 == 0:
                ks = {k for h in hist for k in h}           # One-hop batches have no rel1_acc.
                avg = {k: sum(h[k] for h in hist if k in h) / sum(k in h for h in hist) for k in ks}
                print(f"step {step} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(avg.items()))
                      + f" {(time.time() - t0) / step:.2f}s/step", flush=True)
                hist = []
            if step % a.eval_every == 0 or step == a.steps:
                d = evaluate(mlm, dev_ev, index, H)["kgmem [all]"]
                print(f"dev @ {step}: {json.dumps(d)}", flush=True)
                if a.save and d["chain@5"] > best:
                    best = d["chain@5"]
                    mlm.save(ensure_parent(a.save), {"args": vars(a), "step": step, "dev": d, "n_lib": a.n_lib})
        if a.save:
            mlm.load(a.save)
        res["dev_best_chain@5"] = best
    res["eval"] = evaluate(mlm, ev, index, H, a.limit, trained={lib[i]["name"] for i in
                                                               {index[c] for c in trained}})
    print(json.dumps(res["eval"]), flush=True)
    json.dump(res, open(ensure_parent(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
