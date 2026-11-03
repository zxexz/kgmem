"""Fit a logistic gate from toolmem.py --gate_out features.

  python -m experiments.retrieval.toolmem_gate --gate loso_no_apigen_gate.jsonl --train_sources toolbench,toolACE --out loso_no_apigen_g.jsonl

The label is whether the memory's top choice among BM25's 32 candidates is a gold tool.
Fit only on gatefit rows from --train_sources. Write g for every gatefit/eval row, and
report AUC, label rate and mean g by pool and source."""
from kgmem.paths import ensure_parent

import argparse
import json
import math

import numpy as np

FEATS = ("max_cand", "max_all", "top_in_cand", "margin", "trained_share", "entropy")


def design(rows):
    X = []
    for r in rows:
        X.append([math.log(r["max_cand"] + 1e-6), math.log(r["max_all"] + 1e-6), r["top_in_cand"],
                  math.log(r["margin"] + 1e-6), r["trained_share"], r["entropy"]])
    return np.array(X, dtype=np.float64)


def fit_logistic(X, y, l2=1e-2, iters=200):
    """Newton's method on the L2-penalised log-likelihood (features standardised first)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda Xn: 1 / (1 + np.exp(-np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd]) @ w)), w


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--train_sources", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.gate)]
    srcs = set(a.train_sources.split(","))
    fit_rows = [r for r in rows if r["pool"] == "gatefit" and r["source"] in srcs]
    predict, w = fit_logistic(design(fit_rows), np.array([r["label"] for r in fit_rows], dtype=np.float64))
    g = predict(design(rows))
    with open(ensure_parent(a.out), "w") as f:
        for r, v in zip(rows, g):
            f.write(json.dumps(dict(qid=r["qid"], source=r["source"], pool=r["pool"], g=round(float(v), 6))) + "\n")
    report = {"fit on": f"{len(fit_rows)} gate-fit queries of {sorted(srcs)}",
              "weights (intercept, standardised " + ", ".join(FEATS) + ")": [round(float(x), 3) for x in w]}
    for pool in ("gatefit", "eval"):
        for src in sorted({r["source"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["pool"] == pool and r["source"] == src]
            y = np.array([rows[i]["label"] for i in idx])
            report[f"{pool} [{src}{'' if src in srcs else ', held out'}]"] = dict(
                n=len(idx), label_rate=round(float(y.mean()), 3), mean_g=round(float(g[idx].mean()), 3),
                auc=round(auc(g[idx], y), 3))
        idx = [i for i, r in enumerate(rows) if r["pool"] == pool]
        report[f"{pool} [all]"] = dict(auc=round(auc(g[idx], np.array([rows[i]["label"] for i in idx])), 3))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
