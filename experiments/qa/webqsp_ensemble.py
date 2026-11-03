"""Combine reader probabilities over identical kgmem candidate lists.

  python -m experiments.qa.webqsp_ensemble --probs runs/webqsp/readers/7b.probs.jsonl,runs/webqsp/readers/14b.probs.jsonl --overlap cwq_train_webqsp_test_sources.json --out runs/webqsp/ensemble.json

Inputs are webqsp_reader.py --dump_probs files with matching candidate order.
Search simplex weights in steps of 0.1 on dev, then score the selected weights
on test. --equal uses fixed equal weights. Report overall accuracy and subsets
from --overlap.
"""
import argparse
import itertools
import json

from experiments.qa.webqsp_compare import normalize
from kgmem.paths import ensure_parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--overlap", default=None)
    ap.add_argument("--equal", action="store_true", help='use equal reader weights without a dev weight search')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = a.probs.split(",")
    runs = [{(r["split"], r["id"]): r for r in map(json.loads, open(f))} for f in files]
    keys = [k for k in runs[0] if all(k in r for r in runs[1:])]
    hit = lambda r, j: j is not None and normalize(r["cands"][j]) in set(r["gold"])

    def acc(split, w):
        ks = [k for k in keys if k[0] == split]
        n = 0
        for k in ks:
            p = [sum(wi * run[k]["p"][j] for wi, run in zip(w, runs)) for j in range(len(runs[0][k]["cands"]))]
            n += hit(runs[0][k], max(range(len(p)), key=p.__getitem__) if p else None)
        return n / max(1, len(ks))
    grid = [] if a.equal else [w for w in itertools.product(*[range(11)] * len(runs)) if sum(w) == 10]
    res = {"files": files, "single on dev": {f: round(100 * acc("validation", [int(i == j) for j in range(len(runs))]), 2)
                                              for i, f in enumerate(files)}}
    if a.equal:
        w = [1 / len(runs)] * len(runs)
        res["weights (equal, fixed in advance)"] = w
    else:
        best = max(grid, key=lambda w: acc("validation", [x / 10 for x in w]))
        w = [x / 10 for x in best]
        res["weights (chosen on dev)"] = w
    res["dev"] = round(100 * acc("validation", w), 2)
    overlap = set(json.load(open(a.overlap))) if a.overlap else set()
    for sub, keep in (("all", lambda k: True), ("CWQ-train overlap", lambda k: k[1] in overlap),
                      ("no overlap", lambda k: k[1] not in overlap)):
        ks = [k for k in keys if k[0] == "test" and keep(k)]
        n = 0
        for k in ks:
            p = [sum(wi * run[k]["p"][j] for wi, run in zip(w, runs)) for j in range(len(runs[0][k]["cands"]))]
            n += hit(runs[0][k], max(range(len(p)), key=p.__getitem__) if p else None)
        res[f"test [{sub}]"] = dict(n=len(ks), strict=round(100 * n / max(1, len(ks)), 2))
    print(json.dumps(res, indent=1))
    json.dump(res, open(ensure_parent(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
