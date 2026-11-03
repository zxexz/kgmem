"""Compare BM25, memory and gated stacks by nDCG@10 on BM25's 32 candidates.

  python -m experiments.retrieval.toolmem_loso_report --loso data/toolret/loso --run runs/loso_no_apigen --train_sources toolbench,toolACE

Read gatefit/eval pools from toolret_loso.py, memory scores from _features.jsonl, and
gate values from _g.jsonl. The logistic stack uses [-log BM25 rank, s, s*g, g]. Fit on
gatefit rows from training sources and report eval results separately for each source,
including the source excluded from memory training."""
import argparse
import collections
import json
import math
import os

import numpy as np

from experiments.retrieval.toolmem_gate import fit_logistic


def ndcg10(order, gold):
    dcg = sum(1 / math.log2(i + 2) for i, t in enumerate(order[:10]) if t in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    return dcg / idcg if idcg else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loso", required=True)
    ap.add_argument("--run", required=True, help="output prefix, e.g. runs/loso_no_apigen")
    ap.add_argument("--train_sources", required=True)
    ap.add_argument("--knn", default="", help="toolret_knn.py output for kNN baselines and stack features")
    a = ap.parse_args()
    srcs = set(a.train_sources.split(","))
    q = {}
    for pool in ("gatefit", "eval"):
        for l in open(os.path.join(a.loso, f"{pool}.jsonl")):
            r = json.loads(l)
            q[r["qid"]] = dict(source=r["source"], pool=pool, gold=set(r["gold"]),
                               cands=[c["tool_id"] for c in r["candidates"]])
    s = collections.defaultdict(dict)
    for l in open(a.run + "_features.jsonl"):
        r = json.loads(l)
        s[r["qid"]][r["tool_id"]] = r["kgmem"]
    g = {r["qid"]: r["g"] for r in map(json.loads, open(a.run + "_g.jsonl"))}
    sim, knn = {}, {}
    for l in open(a.knn) if a.knn else []:
        r = json.loads(l)
        sim[r["qid"]] = max(r["sim"][x] for x in srcs)
        knn[r["qid"]] = [max(v) for v in zip(*(r["knn"][x] for x in srcs))]
    qids = [k for k in s if k in g and k in q and (k in sim or not a.knn)]    # Restrict reports to rows with all requested features.
    stacks = {"stack": ("-log bm25 rank", "s", "s*g", "g")}
    if a.knn:
        stacks |= {"stack+knn": ("-log bm25 rank", "s", "s*g", "g", "sim", "s*sim", "knn", "s*knn"),
                   "knn stack (no memory)": ("-log bm25 rank", "sim", "knn")}

    def rows(ks, names):
        X, y = [], []
        for k in ks:
            for i, t in enumerate(q[k]["cands"]):
                v, c = s[k].get(t, -1.0), knn[k][i] if a.knn else 0.0
                f = {"-log bm25 rank": -math.log(i + 1), "s": v, "s*g": v * g[k], "g": g[k], "sim": sim.get(k, 0.0),
                     "s*sim": v * sim.get(k, 0.0), "knn": c, "s*knn": v * c}
                X.append([f[n] for n in names]); y.append(t in q[k]["gold"])
        return np.array(X), np.array(y, dtype=np.float64)
    fit = [k for k in qids if q[k]["pool"] == "gatefit" and q[k]["source"] in srcs]
    report, predictors = {"stacks fit on": f"{len(fit)} gate-fit queries of {sorted(srcs)}"}, {}
    for name, feats in stacks.items():
        predictors[name], w = fit_logistic(*rows(fit, feats))
        report[f"{name} weights (intercept, {', '.join(feats)}; standardised)"] = [round(float(x), 3) for x in w]
    for src in sorted({q[k]["source"] for k in qids}):
        ks = [k for k in qids if q[k]["pool"] == "eval" and q[k]["source"] == src]
        o = np.cumsum([0] + [len(q[k]["cands"]) for k in ks])
        p = {name: predictors[name](rows(ks, feats)[0]) for name, feats in stacks.items()}
        res = collections.defaultdict(float)
        for i, k in enumerate(ks):
            c, gold = q[k]["cands"], q[k]["gold"]
            res["bm25"] += ndcg10(c, gold)
            res["memory"] += ndcg10(sorted(c, key=lambda t: -s[k].get(t, -1.0)), gold)
            if a.knn:                                                  # Ties retain BM25 order.
                res["knn"] += ndcg10([c[j] for j in np.argsort(-np.array(knn[k]), kind="stable")], gold)
                res["mean sim"] += sim[k]
            for name in stacks:
                res[name] += ndcg10([c[j] for j in np.argsort(-p[name][o[i]:o[i + 1]], kind="stable")], gold)
            res["mean g"] += g[k]
        out = {m: round(v / len(ks), 4) for m, v in res.items()}
        for name in stacks:
            out[f"{name} - bm25"] = round(out[name] - out["bm25"], 4)
        out["n"] = len(ks)
        report[f"eval nDCG@10 [{src}{'' if src in srcs else ', held out'}]"] = out
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
