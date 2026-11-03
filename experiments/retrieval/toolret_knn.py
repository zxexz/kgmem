"""Compute nearest-training-query features and a kNN tool baseline for LOSO pools.

  python -m experiments.retrieval.toolret_knn --loso data/toolret/loso --out data/toolret/loso/knn.jsonl

Use TF-IDF cosine on task + request, separately by training source. sim is the
nearest training cosine. A candidate's knn is the highest cosine among --k neighbours
whose gold includes it, or zero. Training and pool rows must pass gated_ids.txt.
--runs exports features for selected source combinations."""
from kgmem.paths import ensure_parent

import argparse
import collections
import json
import math
import os

import numpy as np
from scipy import sparse

from experiments.retrieval.toolret_prep import TOKEN


def text(r):
    return f"{r['task']} {r['query']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loso", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--runs", default="", help='also write per-run files, e.g. "all=toolbench,apigen,toolACE;no_apigen=toolbench,toolACE"')
    ap.add_argument("--runs_dir", default="runs/retrieval/toolret_knn")
    a = ap.parse_args()
    keep = set(open(os.path.join(a.loso, "gated_ids.txt")).read().split())
    train = collections.defaultdict(list)
    for l in open(os.path.join(a.loso, "rows.jsonl")):
        r = json.loads(l)
        if r["pool"] == "train" and r["qid"] in keep:
            train[r["source"]].append((text(r), set(r["gold"])))
    pools = [json.loads(l) for p in ("gatefit", "eval") for l in open(os.path.join(a.loso, f"{p}.jsonl"))]
    pools = [r for r in pools if r["qid"] in keep]
    docs = [t for s in sorted(train) for t, _ in train[s]] + [text(r) for r in pools]
    vocab, df = {}, collections.Counter()
    toks = [collections.Counter(TOKEN.findall(d.lower())) for d in docs]
    for c in toks:
        for w in c:
            df[vocab.setdefault(w, len(vocab))] += 1
    idf = np.zeros(len(vocab), dtype=np.float32)
    for i, n in df.items():
        idf[i] = math.log(len(docs) / n) + 1

    def tfidf(cs):
        rows, cols, vals = [], [], []
        for i, c in enumerate(cs):
            for w, n in c.items():
                j = vocab[w]
                rows.append(i); cols.append(j); vals.append((1 + math.log(n)) * idf[j])
        M = sparse.csr_matrix((vals, (rows, cols)), shape=(len(cs), len(vocab)), dtype=np.float32)
        nrm = np.sqrt(M.multiply(M).sum(1)).A1 + 1e-9
        return sparse.diags(1 / nrm) @ M
    n_train = sum(len(v) for v in train.values())
    Q = tfidf(toks[n_train:])
    out = [dict(qid=r["qid"], source=r["source"], pool=r["pool"], sim={}, knn={}) for r in pools]
    off = 0
    for s in sorted(train):
        T = tfidf(toks[off:off + len(train[s])]).T.tocsc()
        golds = [g for _, g in train[s]]
        off += len(train[s])
        for i in range(0, len(pools), 500):
            S = (Q[i:i + 500] @ T).toarray()
            top = np.argpartition(-S, a.k, axis=1)[:, :a.k]
            for j, row in enumerate(top):
                o, r = out[i + j], pools[i + j]
                o["sim"][s] = round(float(S[j, row].max()), 4)
                best = {}
                for t in row:
                    for g in golds[t]:
                        best[g] = max(best.get(g, 0.0), float(S[j, t]))
                o["knn"][s] = [round(best.get(c["tool_id"], 0.0), 4) for c in r["candidates"]]
        print(s, len(train[s]), "training rows", flush=True)
    with open(ensure_parent(a.out), "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    if a.runs:
        export(a, pools, out)


def export(a, pools, out):
    """Export LOSO features using each run's training sources.

    loso_<run>_knn.jsonl holds candidate scores; _knn_gate.jsonl holds sim_any
    (nearest-query cosine) and sim_in32 (best candidate kNN score)."""
    for spec in a.runs.split(";"):
        run, srcs = spec.split("=")
        srcs = srcs.split(",")
        with open(ensure_parent(os.path.join(a.runs_dir, f"loso_{run}_knn.jsonl")), "w") as f, \
                open(ensure_parent(os.path.join(a.runs_dir, f"loso_{run}_knn_gate.jsonl")), "w") as h:
            for r, o in zip(pools, out):
                knn = [max(v) for v in zip(*(o["knn"][s] for s in srcs))]
                for c, v in zip(r["candidates"], knn):
                    f.write(json.dumps(dict(qid=r["qid"], tool_id=c["tool_id"], knn=v)) + "\n")
                h.write(json.dumps(dict(qid=r["qid"], source=r["source"], pool=r["pool"],
                                        sim_any=max(o["sim"][s] for s in srcs), sim_in32=max(knn))) + "\n")


if __name__ == "__main__":
    main()
