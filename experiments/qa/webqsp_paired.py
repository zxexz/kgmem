"""Paired comparison of reader ensembles and RoG on WebQSP test questions.

  python -m experiments.qa.webqsp_paired --prep runs/webqsp/prep.pt --rog predictions.jsonl --probs runs/webqsp/readers/32b.probs.jsonl --overlap cwq_train_webqsp_test_sources.json

Choose the highest equal-weight reader probability and RoG's first nonempty
answer line. Score normalized exact matches on questions with RoG predictions.
Report accuracy differences, an exact two-sided McNemar test, and a 95%
bootstrap interval from 10,000 question resamples, overall and by overlap subset.
"""
import argparse
import json
import math
import random

import torch

from experiments.qa.webqsp_compare import normalize, strict
from kgmem.paths import ensure_parent


def mcnemar(b, c):
    """Exact two-sided binomial test: b is kgmem-only correct; c is RoG-only correct."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--rog", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--overlap", default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    P = torch.load(a.prep, weights_only=False)
    gold = {it["id"]: [P["entities"][x] for x in it["answers"]] for it in P["splits"]["test"]}
    lines = lambda p: p if isinstance(p, list) else str(p).split("\n")
    rog = {x["id"]: strict(lines(x["prediction"]), gold[x["id"]]) for x in map(json.loads, open(a.rog))
           if x["id"] in gold}
    runs = [{r["id"]: r for r in map(json.loads, open(f)) if r["split"] == "test"} for f in a.probs.split(",")]
    ours = {}
    for i in gold:
        rs = [run[i] for run in runs if i in run]
        if not rs or not rs[0]["cands"]:
            ours[i] = False
            continue
        p = [sum(r["p"][j] for r in rs) for j in range(len(rs[0]["cands"]))]
        pick = rs[0]["cands"][max(range(len(p)), key=p.__getitem__)]
        ours[i] = normalize(pick) in {normalize(g) for g in gold[i]}
    overlap = set(json.load(open(a.overlap))) if a.overlap else set()
    rng = random.Random(0)
    res = {"readers": a.probs.split(",")}
    for sub, keep in (("all", lambda i: True), ("CWQ-train overlap", lambda i: i in overlap),
                      ("no overlap", lambda i: i not in overlap)):
        ids = [i for i in gold if keep(i) and i in rog]
        if not ids:                                    # No overlap list (e.g. CWQ): skip the empty subsets.
            continue
        x, y = [ours[i] for i in ids], [rog[i] for i in ids]
        b = sum(o and not r for o, r in zip(x, y))
        c = sum(r and not o for o, r in zip(x, y))
        d = [o - r for o, r in zip(x, y)]
        boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
        res[sub] = dict(n=len(ids), kgmem=round(100 * sum(x) / len(ids), 2), rog=round(100 * sum(y) / len(ids), 2),
                        diff=round(100 * sum(d) / len(ids), 2),
                        ci95=[round(100 * boots[249], 2), round(100 * boots[9749], 2)],
                        ours_only=b, rog_only=c, mcnemar_p=round(mcnemar(b, c), 4))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(ensure_parent(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
