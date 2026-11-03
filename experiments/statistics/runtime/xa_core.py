import json

import torch

from .xa_num import pick
from kgmem.graph import ExactScorer, GraphTensors, KnowledgeGraph


def text(q):
    return f"Q: {q}\nA:"


class Tab:
    def __init__(self, path):
        d = json.load(open(path))
        t = d["t"]
        self.kg = KnowledgeGraph(d["e"], d["r"], [(t[i], t[i + 1], t[i + 2]) for i in range(0, len(t), 3)])
        self.kg._version += 1
        self.v = {int(k): x for k, x in d["v"].items()}
        self.a = d["a"]

    def put(self, m):
        dev = m.device
        toks = [m.tokenize_name(n)[:12] for n in self.kg.entities]
        L = max(map(len, toks))
        nt = torch.full((len(toks), L), -1, dtype=torch.long)
        for i, t in enumerate(toks):
            nt[i, :len(t)] = torch.tensor(t)
        alias = {}
        for n, i in self.a:
            k = tuple(m.tokenize_name(n))
            alias[k] = i if alias.get(k, i) == i else -1
        m.ctx.graph = GraphTensors(
            ent_keys=enc(m, self.kg.entities, m.cfg.key_pool), rel_keys=enc(m, self.kg.relations, m.cfg.rel_key_pool),
            name_tok=nt.to(dev), name_len=torch.tensor(list(map(len, toks)), device=dev),
            scorer=ExactScorer(self.kg, m.cfg.max_tails, dev), kg=self.kg, rel_keys_first=None, alias=alias,
            rel_alias=m._alias(self.kg.relations))


def enc(m, names, pool):
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(len(m.tokenize_name(n)), []).append(i)
    out = [None] * len(names)
    for idx in groups.values():
        for j in range(0, len(idx), 64):
            part = idx[j:j + 64]
            for i, k in zip(part, m.encode_names([names[i] for i in part], bs=len(part), pool=pool)):
                out[i] = k
    return torch.stack(out)


def top(m, tab, k):
    pos = m.ctx.cand_ids.shape[1] - 1
    cid, cw = m.ctx.cand_ids[0, pos], m.ctx.cand_w[0, pos].float()
    dense = torch.zeros(len(tab.kg.entities), device=cw.device).scatter_add_(0, cid, cw)
    w, e = dense.topk(k)
    vals = [(tab.v[int(i)], float(x)) for x, i in zip(w, e) if x > 0 and int(i) in tab.v]
    return vals, 1.0 - min(1.0, float(dense.sum()))
