"""Split ToolRet training rows by source and attach BM25 candidates.

  python -m experiments.retrieval.toolret_prep --dir data/toolret --out OUT --heldout apigen,toolACE --n_heldout 3000

ToolRet-Training-20w has no source field. Match positive documents verbatim to
mangopy/ToolRet-Tools and recover sources from tool ids. Keep rows whose positives
share one source. Sample --n_heldout rows per --heldout source; other sources train.
The script does not read mangopy/ToolRet-Queries, the benchmark's test queries.

Write tools.jsonl (id, source, doc), train.jsonl and heldout.jsonl (qid, source,
query, task, gold). Held-out rows include BM25's top 32 over the full tool corpus."""
from kgmem.paths import ensure_parent

import argparse
import ast
import collections
import glob
import json
import math
import os
import random
import re

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

TOKEN = re.compile(r"[a-z0-9]+")


def source_of(tool_id):
    return tool_id.split("_tool_")[0] if "_tool_" in tool_id else tool_id.rsplit("_", 1)[0]


def bm25_index(docs, k1=1.5, b=0.75):
    vocab, rows, cols, vals = {}, [], [], []
    lens = np.zeros(len(docs))
    for i, d in enumerate(docs):
        c = collections.Counter(TOKEN.findall(d.lower()))
        lens[i] = sum(c.values())
        for w, n in c.items():
            rows.append(i); cols.append(vocab.setdefault(w, len(vocab))); vals.append(n)
    tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab)), dtype=np.float32)
    df = np.bincount(tf.indices, minlength=len(vocab))
    idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5)).astype(np.float32)
    avg = lens.mean()
    tf = tf.tocoo()
    norm = k1 * (1 - b + b * lens[tf.row] / avg)
    w = tf.data * (k1 + 1) / (tf.data + norm) * idf[tf.col]
    return vocab, sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()     # (V, N)


def bm25_top(vocab, W, queries, k=32):
    rows, cols = [], []
    for i, q in enumerate(queries):
        for w in set(TOKEN.findall(q.lower())):
            if w in vocab:
                rows.append(i); cols.append(vocab[w])
    Q = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(queries), len(vocab)))
    S = (Q @ W).toarray()
    top = np.argpartition(-S, k, axis=1)[:, :k]
    return [list(t[np.argsort(-S[i, t])]) for i, t in enumerate(top)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default="apigen,toolACE")
    ap.add_argument("--n_heldout", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tools = []
    for f in sorted(glob.glob(f"{a.dir}/tools/*/*.parquet")):
        tools += pq.read_table(f).to_pylist()
    by_doc = {t["documentation"].strip(): i for i, t in enumerate(tools)}
    with open(ensure_parent(f"{a.out}/tools.jsonl"), "w") as f:
        for t in tools:
            f.write(json.dumps(dict(id=t["id"], source=source_of(t["id"]), doc=t["documentation"])) + "\n")
    held = set(a.heldout.split(","))
    rows = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{a.dir}/train/**/*.parquet", recursive=True)):
        for r in pq.read_table(f, columns=["id", "query", "prompt", "positive"]).to_pylist():
            pos = r["positive"]
            pos = ast.literal_eval(pos) if isinstance(pos, str) else pos
            ids = [by_doc.get(p.strip()) for p in pos]
            if not ids or None in ids:
                continue
            srcs = {source_of(tools[i]["id"]) for i in ids}
            if len(srcs) == 1:
                rows[srcs.pop()].append(dict(qid=r["id"], query=r["query"], task=r["prompt"],
                                             gold=[tools[i]["id"] for i in ids]))
    print({s: len(v) for s, v in rows.items()})
    rng = random.Random(a.seed)
    train = [dict(r, source=s) for s, v in rows.items() if s not in held for r in v]
    test = [dict(r, source=s) for s in sorted(held) for r in rng.sample(rows[s], min(a.n_heldout, len(rows[s])))]
    vocab, W = bm25_index([t["documentation"] for t in tools])
    tops = []
    for i in range(0, len(test), 500):
        tops += bm25_top(vocab, W, [r["query"] for r in test[i:i + 500]])
    hit = collections.Counter()
    for r, t in zip(test, tops):
        r["candidates"] = [dict(tool_id=tools[j]["id"], doc=tools[j]["documentation"]) for j in t]
        got = {c["tool_id"] for c in r["candidates"]}
        hit["any gold in BM25 top 32"] += bool(got & set(r["gold"]))
        hit["all gold in BM25 top 32"] += set(r["gold"]) <= got
    for name, data in (("train", train), ("heldout", test)):
        with open(ensure_parent(f"{a.out}/{name}.jsonl"), "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
    print(f"train {len(train)} rows ({collections.Counter(r['source'] for r in train)}), held out {len(test)}; "
          + ", ".join(f"{k}: {100 * v / len(test):.1f}%" for k, v in hit.items()))


if __name__ == "__main__":
    main()
