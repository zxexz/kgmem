"""A constraint step for questions that name several entities: keep the retriever's candidates that every named entity
reaches, ahead of the rest.

  python -m experiments.qa.cwq_constrain --prep cwq_prep.pt --dump cwq_pred50 --out cwq_pred50c --hops 2

CWQ builds conjunction and comparative questions by adding an entity to a WebQSP query: the answer must also relate
to that entity. For each question with two or more topic entities, the candidates of the retriever's dump (its ranked
`top` list) that lie within --hops of every topic entity in the question's own subgraph move to the front, in the
retriever's order; the others follow, also in order. Single-entity questions are left as they are. Recall@k of the
named candidates is reported before and after, per split.
"""
import argparse
import collections
import json
import re

import torch

from experiments.qa.webqsp_compare import normalize
from kgmem.paths import ensure_parent

UNNAMED = re.compile(r"^[mg]\.[0-9a-z_]+$")


def reach(adj, src, hops):
    seen, front = {src}, {src}
    for _ in range(hops):
        nxt = set()
        for e in front:
            nxt |= adj.get(e, set())
        nxt -= seen
        seen |= nxt
        front = nxt
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--dump", required=True, help="webqsp.py eval --dump prefix (X.<split>.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--hops", type=int, default=2)
    a = ap.parse_args()
    P = torch.load(a.prep, weights_only=False)
    ents = P["entities"]
    res = {}
    for split in a.splits.split(","):
        meta = {it["id"]: it for it in P["splits"][split]}
        rows = [json.loads(l) for l in open(f"{a.dump}.{split}.jsonl")]
        ks = (1, 3, 5, 10)
        before, after = collections.Counter(), collections.Counter()
        moved = 0
        with open(ensure_parent(f"{a.out}.{split}.jsonl"), "w") as f:
            for r in rows:
                it = meta[r["id"]]
                gold = {normalize(ents[x]) for x in it["answers"]}
                top = r.get("top") or []
                if len(it["topic"]) >= 2 and len(it["tri"]):
                    adj = collections.defaultdict(set)
                    for h, _, t in it["tri"].tolist():
                        adj[h].add(t)
                        adj[t].add(h)
                    names = set.intersection(*[{ents[e] for e in reach(adj, x, a.hops)} for x in it["topic"]])
                    inside = [c for c in top if c[0] in names]
                    new = inside + [c for c in top if c[0] not in names]
                    moved += new != top
                    r = dict(r, top=new, pred=new[0][0] if new else r.get("pred"))
                f.write(json.dumps(r) + "\n")
                for k in ks:
                    cand0 = [c for c, _ in top if not UNNAMED.match(c)][:k]
                    cand1 = [c for c, _ in r.get("top", []) if not UNNAMED.match(c)][:k]
                    before[k] += any(normalize(c) in gold for c in cand0)
                    after[k] += any(normalize(c) in gold for c in cand1)
        n = max(1, len(rows))
        res[split] = dict(n=len(rows), reordered=moved,
                          before={f"recall@{k}": round(100 * before[k] / n, 1) for k in ks},
                          after={f"recall@{k}": round(100 * after[k] / n, 1) for k in ks})
        print(split, json.dumps(res[split]), flush=True)
    json.dump(res, open(ensure_parent(f"{a.out}.recall.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
