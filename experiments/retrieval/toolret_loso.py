"""Build leave-one-source-out training, gatefit and eval pools from ToolRet-Training-20w.

  python -m experiments.retrieval.toolret_loso --dir data/toolret --out data/toolret/loso

Recover sources by matching positive tool documents, as in toolret_prep.py. Per source,
reserve seeded gatefit and eval pools from memory training; the remaining rows train.
Both reserved pools include BM25's top 32 over the full corpus. Write all_q.jsonl for
decontamination, rows.jsonl with gold tools and pool labels, gatefit.jsonl, eval.jsonl,
and tools.jsonl."""
from kgmem.paths import ensure_parent

import argparse
import collections
import json
import os
import random

from experiments.retrieval import toolret_prep as tp
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_gatefit", type=int, default=1000)
    ap.add_argument("--n_eval", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tools = [json.loads(l) for l in open(os.path.join(a.dir, "out", "tools.jsonl"))]
    by_doc = {t["doc"].strip(): i for i, t in enumerate(tools)}
    import ast
    import glob
    import pyarrow.parquet as pq
    rows = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{a.dir}/train/**/*.parquet", recursive=True)):
        for r in pq.read_table(f, columns=["id", "query", "prompt", "positive"]).to_pylist():
            pos = ast.literal_eval(r["positive"]) if isinstance(r["positive"], str) else r["positive"]
            ids = [by_doc.get(p.strip()) for p in pos]
            if not ids or None in ids:
                continue
            srcs = {tools[i]["source"] for i in ids}
            if len(srcs) == 1:
                rows[srcs.pop()].append(dict(qid=r["id"], query=r["query"], task=r["prompt"],
                                             gold=[tools[i]["id"] for i in ids]))
    rng = random.Random(a.seed)
    out, pools = [], {"gatefit": [], "eval": []}
    for src in sorted(rows):
        rs = rows[src]
        idx = list(range(len(rs)))
        rng.shuffle(idx)
        gf, ev = set(idx[:a.n_gatefit]), set(idx[a.n_gatefit:a.n_gatefit + a.n_eval])
        for i, r in enumerate(rs):
            pool = "gatefit" if i in gf else "eval" if i in ev else "train"
            out.append(dict(r, source=src, pool=pool))
            if pool != "train":
                pools[pool].append(out[-1])
    vocab, W = tp.bm25_index([t["doc"] for t in tools])
    for name, ps in pools.items():
        tops = []
        for i in range(0, len(ps), 500):
            tops += tp.bm25_top(vocab, W, [r["query"] for r in ps[i:i + 500]])
        with open(ensure_parent(os.path.join(a.out, f"{name}.jsonl")), "w") as f:
            for r, t in zip(ps, tops):
                f.write(json.dumps(dict(r, candidates=[dict(tool_id=tools[j]["id"], doc=tools[j]["doc"])
                                                       for j in t])) + "\n")
    with open(ensure_parent(os.path.join(a.out, "rows.jsonl")), "w") as f, open(ensure_parent(os.path.join(a.out, "all_q.jsonl")), "w") as g:
        for r in out:
            f.write(json.dumps(r) + "\n")
            g.write(json.dumps(dict(qid=r["qid"], source=r["source"], query=r["query"], task=r["task"])) + "\n")
    with open(ensure_parent(os.path.join(a.out, "tools.jsonl")), "w") as f:
        for t in tools:
            f.write(json.dumps(t) + "\n")
    print({s: collections.Counter(r["pool"] for r in out if r["source"] == s) for s in sorted(rows)})


if __name__ == "__main__":
    main()
