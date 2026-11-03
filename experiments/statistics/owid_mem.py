"""Train and query a statistics graph on a supplied backbone.

  python -m experiments.statistics.owid_mem train --store store.json --questions questions --model MODEL --backbone-loader backbone_loader.py --layer 41 --dtype bfloat16 --save runs/owid_v2.pt --out runs/owid_v2.json
  python -m experiments.statistics.owid_mem eval --store store.json --questions questions --model MODEL --backbone-loader backbone_loader.py --adapter runs/owid_v2.pt --out eval.json

The two-hop graph is place -> series node -> value node, using indicator and
"in YEAR" relations. Value names use three significant figures. Only place names
and aliases link to entities; numbers do not link to value nodes. Training rows
contain one unpadded question without its answer, with the external attention loader's policy B.

The reader decodes values, series and years from layer-L question states.
abstain = 1 - retrieved mass; rel_exists_soft reduces mass for missing edges."""
from kgmem.paths import ensure_parent

import argparse
import json
import math
import os
import random
import re
from fractions import Fraction
import time

import torch

from experiments.retrieval import funclib
from kgmem import MemConfig
from kgmem.graph import ExactScorer, GraphTensors, KnowledgeGraph
from kgmem.memlm import Batch
from kgmem.tasks import _find_last
from experiments.statistics.owid_store import fmt, place

SETS = ("dev", "fresh_wording", "held_keys", "held_series", "missing", "unknown_series", "unknown_series_v2",
        "unknown_series_v3")
NULL = "an indicator this store does not hold"


class Store:
    """Graph, numeric value nodes and series units from a store JSON file."""

    def __init__(self, path, licence="all"):
        st = json.load(open(path))
        if licence in ("clean", "clean_no_vdem"):
            st["series"] = [s for s in st["series"] if s["licence_class"] == "clean"]
        if licence == "clean_no_vdem":                 # V-Dem's own dataset licence is share-alike
            st["series"] = [s for s in st["series"] if "V-Dem" not in s.get("citation", "")]
        y0, y1 = st["years"]
        self.series = {s["id"]: s for s in st["series"]}
        self.heads = st["heads"]
        rels = [s["ind"] for s in st["series"]] + [f"in {y}" for y in range(y0, y1 + 1)] + [NULL]
        assert len(set(rels)) == len(rels), "two series share an indicator phrase"
        ri = {n: i for i, n in enumerate(rels)}
        ents, index, self.value_of = [h["name"] for h in self.heads], {}, {}
        index.update({n: i for i, n in enumerate(ents)})
        triples = []
        for s in st["series"]:
            r = ri[s["ind"]]
            for h, ys in st["values"][s["id"]].items():
                node = f"{h} - {s['title']}"
                index[node] = len(ents); ents.append(node)
                triples.append((index[h], r, index[node]))
                for y, v in ys.items():
                    name = fmt(v)
                    if name not in index:
                        index[name] = len(ents); ents.append(name); self.value_of[index[name]] = v
                    triples.append((index[node], ri[f"in {y}"], index[name]))
        self.kg = KnowledgeGraph(ents, rels, triples)
        self.kg._version += 1
        self.n_heads = len(self.heads)
        self.unit = {ri[s["ind"]]: s["unit"] for s in st["series"]}
        self.series_rel = {s["id"]: ri[s["ind"]] for s in st["series"]}
        self.year_rel = {y: ri[f"in {y}"] for y in range(y0, y1 + 1)}
        self.rel_series = {v: k for k, v in self.series_rel.items()}
        self.null = ri[NULL]                           # NULL has no outgoing edges.
        self.rel_year = {v: k for k, v in self.year_rel.items()}

    def load(self, mlm):
        """Encode entity and relation keys; allow entity aliases for places only."""
        dev = mlm.device
        toks = [mlm.tokenize_name(n)[:12] for n in self.kg.entities]
        L = max(map(len, toks))
        name_tok = torch.full((len(toks), L), -1, dtype=torch.long)
        for i, t in enumerate(toks):
            name_tok[i, :len(t)] = torch.tensor(t)
        alias = {}
        for i, h in enumerate(self.heads):
            for n in {h["name"], place(h["name"])[0], *h["aliases"]}:
                t = tuple(mlm.tokenize_name(n))
                alias[t] = i if alias.get(t, i) == i else -1
        t0 = time.time()
        ent_keys = funclib.encode(mlm, self.kg.entities, mlm.cfg.key_pool)
        rel_keys = funclib.encode(mlm, self.kg.relations, mlm.cfg.rel_key_pool)
        mlm.ctx.graph = GraphTensors(
            ent_keys=ent_keys, rel_keys=rel_keys, name_tok=name_tok.to(dev),
            name_len=torch.tensor(list(map(len, toks)), device=dev),
            scorer=ExactScorer(self.kg, mlm.cfg.max_tails, dev), kg=self.kg, rel_keys_first=None,
            alias=alias, rel_alias=mlm._alias(self.kg.relations))
        print(f"keys: {len(self.kg.entities)} entities, {len(self.kg.relations)} relations, "
              f"{len(self.kg.triples)} triples in {time.time() - t0:.0f}s", flush=True)


def prompt(q):
    return f"Q: {q}\nA:"


def row(mlm, store, r):
    """Build a question-only supervised Batch and return the answer value-node id."""
    dev, H = mlm.device, mlm.cfg.hops
    p = mlm.tok(prompt(r["question"]), add_special_tokens=False).input_ids
    ids = list(mlm.bos) + p
    m = _find_last(p, mlm.tokenize_name(r["mention"]))
    rel = torch.full((1, H), -100, dtype=torch.long)
    rel[0, :2] = torch.tensor([store.series_rel[r["series"]] if r["series"] else store.null,
                               store.year_rel.get(r["year"], -100)])
    none = torch.full((1, H), -1, dtype=torch.long, device=dev)
    t = torch.tensor([ids], device=dev)
    b = Batch(t, torch.full_like(t, -100), torch.ones_like(t), [], aux_idx=torch.tensor([len(ids) - 1], device=dev),
              aux_ent=torch.tensor([store.kg.ent_index[r["head"]]], device=dev), aux_rel=rel.to(dev),
              aux_hops=torch.tensor([1], device=dev),
              aux_mention=torch.tensor([len(mlm.bos) + m if m >= 0 else -1], device=dev),
              aux_rel_mention=none, aux_rel_span_start=none)
    ans = store.kg.ent_index.get(fmt(r["value"])) if r.get("value") is not None else None
    return b, ans


def decode(mlm, store, k=5):
    """Decode values, series, years and abstain from the latest forward."""
    aux, pos = mlm.ctx.kg_aux, mlm.ctx.cand_ids.shape[1] - 1
    cid, cw = mlm.ctx.cand_ids[0, pos], mlm.ctx.cand_w[0, pos].float()
    dense = torch.zeros(len(store.kg.entities), device=cw.device).scatter_add_(0, cid, cw)
    w, e = dense.topk(k)
    p1 = aux["rel"][0][0].float().softmax(-1)
    p2 = aux["rel"][1][0].float().softmax(-1) if len(aux["rel"]) > 1 else None
    s1 = p1.topk(3)
    out = dict(values=[dict(value=store.value_of.get(int(i)), name=store.kg.entities[int(i)], weight=float(x))
                       for x, i in zip(w, e) if x > 0 and int(i) in store.value_of],
               series=[(store.rel_series.get(int(i)), float(x)) for x, i in zip(s1.values, s1.indices)],
               mass=float(dense.sum()))
    if p2 is not None:
        s2 = p2.topk(3)
        out["years"] = [(store.rel_year.get(int(i)), float(x)) for x, i in zip(s2.values, s2.indices)]
    out["unit"] = store.unit.get(int(s1.indices[0]), "")
    out["abstain"] = 1.0 - min(1.0, out["mass"])
    return out


SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9,
         "trillion": 1e12, "tn": 1e12}
NUM = re.compile(r"""\s*(?P<sign>[-+\u2212])?\s*[$\u20ac\u00a3]?\s*(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?
                    (?:\s*/\s*(?P<den>\d+))?\s*(?P<scale>%|percent|thousand|million|billion|trillion|k|m|mn|bn|b|tn)?\s*""",
                 re.X | re.I)


def option_value(o):
    """Parse a complete numeric option, following strandedcpu.api.parse_option_number.

    Accept signs, well-formed thousands commas, decimals and simple fractions.
    Also allow currency signs, Unicode minus, scale words and trailing percent
    signs; percentages retain their numeric magnitude. Unsupported text returns
    NaN, including units, brackets, mixed numbers and zero denominators."""
    m = NUM.fullmatch(str(o))
    if not m:
        return float("nan")
    x = Fraction(int(m["int"].replace(",", "")))
    if m["frac"] is not None:
        if m["den"] is not None:
            return float("nan")
        x += Fraction(int(m["frac"]), 10 ** len(m["frac"]))
    if m["den"] is not None:
        if int(m["den"]) == 0 or "," in m["int"]:
            return float("nan")
        x /= int(m["den"])
    x *= SCALE.get((m["scale"] or "").lower(), 1)
    return float(-x if m["sign"] in ("-", "\u2212") else x)


def similarity(v, o):
    """Return min/max magnitude for nonzero values of the same sign.

    Equal values score 1; zero versus nonzero scores 0; opposite signs return None."""
    if v == 0 or o == 0:
        return 1.0 if v == o else 0.0
    if (v > 0) != (o > 0):
        return None
    a, b = abs(v), abs(o)
    return min(a, b) / max(a, b)


def choose(values, options, tie=1e-9):
    """Choose the option with the highest weighted numeric similarity.

    Unsupported options are excluded; opposite-sign values contribute no score.
    Return None for missing values, a nonpositive best score, or a tie.
    Option order does not break ties."""
    if not values:
        return None
    score = []
    for o in map(option_value, options):
        if o != o:
            score.append(float("-inf"))
            continue
        sims = [(v["weight"], similarity(v["value"], o)) for v in values]
        score.append(sum(w * sm for w, sm in sims if sm is not None))
    order = sorted(range(len(score)), key=lambda i: -score[i])
    if not order or score[order[0]] <= 0:
        return None
    if len(order) > 1 and score[order[0]] - score[order[1]] <= tie:
        return None
    return order[0]


@torch.no_grad()
def evaluate(mlm, store, rows, k=5, dump=None):
    """Measure decoded values and option choices; optionally dump each item."""
    was = mlm.block.training
    mlm.block.eval()
    res = dict(n=0, exact=0, mc=0, series1=0, year1=0, null1=0, abstain=[], mass=[])
    for r in rows:
        b, ans = row(mlm, store, r)
        mlm.forward(b.ids, b.attn, aux_idx=b.aux_idx)
        d = decode(mlm, store, k)
        res["n"] += 1
        res["abstain"].append(d["abstain"]); res["mass"].append(d["mass"])
        res["series1"] += d["series"][0][0] == r["series"]
        res["null1"] += d["series"][0][0] is None
        res["year1"] += bool(d.get("years")) and d["years"][0][0] == r["year"]
        c = None
        if r.get("value") is not None:
            res["exact"] += bool(d["values"]) and d["values"][0]["name"] == fmt(r["value"])
            c = choose(d["values"], r["options"])
            res["mc"] += c == r["gold"]
        if dump is not None:
            dump.write(json.dumps(dict(qid=r["qid"], kind=r["kind"], choice=c, **d)) + "\n")
    mlm.block.train(was)
    n = max(1, res["n"])
    return dict(n=res["n"], exact=round(100 * res["exact"] / n, 2), mc=round(100 * res["mc"] / n, 2),
                series_top1=round(100 * res["series1"] / n, 2), year_top1=round(100 * res["year1"] / n, 2),
                null_top1=round(100 * res["null1"] / n, 2),
                mean_abstain=round(sum(res["abstain"]) / n, 4)), res["abstain"]


def auc(pos, neg):
    """Probability a positive outranks a negative; ties count as half."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, s = 0.0, 0, 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        s += avg * sum(1 for x in allv[i:j] if x[1])
        i = j
    return (s - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def read_jsonl(path, n=0, store=None):
    rows = [json.loads(l) for l in open(path)]
    if store is not None:                              # Restrict supervision to loaded series.
        rows = [r for r in rows if r["series"] is None or r["series"] in store.series]
    return rows[:n] if n else rows


def build(a, train_texts):
    from experiments.qa import mquake
    if a.adapter:
        sd = torch.load(a.adapter, map_location="cpu", weights_only=True)
        a.layer = sd["layer_idx"]
        cfg = MemConfig(**{**sd["cfg"], **json.loads(a.cfg)})
    else:
        cfg = mquake.mquake_cfg(None, hops=2, rel_named=0.0, kg_gate_loss=0.0, max_tails=1, k_cand=4,
                                **json.loads(a.cfg))
    mlm = funclib.build_model(a, cfg)
    mlm.question_only, mlm.unpadded = True, bool(a.backbone_loader)
    if a.adapter:
        print("adapter:", mlm.load(a.adapter), flush=True)
    elif train_texts:
        mlm.calibrate(train_texts)
    return mlm


def train(a):
    store = Store(a.store, a.licence)
    rows = read_jsonl(os.path.join(a.questions, f"train.{a.suffix}.jsonl"), store=store)
    neg = read_jsonl(os.path.join(a.questions, f"neg_train{a.neg_tag}.{a.suffix}.jsonl")) if a.neg_frac > 0 else []
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    calib = [prompt(r["question"]) for r in rng.sample(rows, 200)] + rng.sample(store.kg.entities, 200)
    mlm = build(a, calib)
    store.load(mlm)
    dev = read_jsonl(os.path.join(a.questions, f"dev.{a.suffix}.jsonl"), a.dev_n, store)
    params = mlm.trainable_parameters()
    opt = torch.optim.AdamW(params, a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    best, hist, t0, res = -1.0, [], time.time(), {"args": vars(a)}
    for step in range(1, a.steps + 1):
        logs, bad = {}, False
        opt.zero_grad()
        for r in (rng.choice(neg if neg and rng.random() < a.neg_frac else rows) for _ in range(a.batch)):
            b, _ = row(mlm, store, r)
            mlm.forward(b.ids, b.attn, aux_idx=b.aux_idx, aux_mention=b.aux_mention, keep=1)
            l, lg = mlm.aux_loss(b, 1.0)
            if not torch.isfinite(l):
                bad = True
                break
            (l / a.batch).backward()
            for k, v in lg.items():
                logs[k] = logs.get(k, 0.0) + v / a.batch
        if bad:
            opt.zero_grad(); sched.step()
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        hist.append(logs)
        if step % 200 == 0:
            ks = {k for h in hist for k in h}
            avg = {k: sum(h[k] for h in hist if k in h) / sum(k in h for h in hist) for k in ks}
            print(f"step {step} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(avg.items()))
                  + f" {(time.time() - t0) / step:.2f}s/step", flush=True)
            hist = []
        if step % a.eval_every == 0 or step == a.steps:
            d, _ = evaluate(mlm, store, dev)
            print(f"dev @ {step}: {json.dumps(d)}", flush=True)
            if d["exact"] > best:
                best = d["exact"]
                mlm.save(ensure_parent(a.save), {"args": vars(a), "step": step, "dev": d})
    mlm.load(a.save)
    res |= run_sets(a, mlm, store)
    json.dump(res, open(ensure_parent(a.out), "w"), indent=1)


def run_sets(a, mlm, store):
    res, abst = {"rel_exists_soft": a.soft}, {}
    mlm.cfg.rel_exists_soft = a.soft                   # Use soft existence scaling for evaluation.
    for name in SETS:
        path = os.path.join(a.questions, f"{name}.{a.suffix}.jsonl")
        if not os.path.exists(path):
            continue
        rows = read_jsonl(path, a.eval_n, store)
        with open(ensure_parent(f"{a.dump}_{name}.jsonl"), "w") if a.dump else open(os.devnull, "w") as f:
            res[name], abst[name] = evaluate(mlm, store, rows, dump=f if a.dump else None)
        print(name, json.dumps(res[name]), flush=True)
    res["abstain AUC: missing vs held_keys"] = round(auc(abst["missing"], abst["held_keys"]), 4)
    res["abstain AUC: held_series vs held_keys"] = round(auc(abst["held_series"], abst["held_keys"]), 4)
    for u in ("unknown_series", "unknown_series_v2", "unknown_series_v3"):
        if abst.get(u):
            res[f"abstain AUC: {u} vs held_keys"] = round(auc(abst[u], abst["held_keys"]), 4)
    print(json.dumps({k: v for k, v in res.items() if "AUC" in k}), flush=True)
    return res


class StoreReader:
    """Read store values from an attached backbone.

    The hook stays disabled for normal forwards. Initialization loads the adapter
    and encodes store keys once.

        r = StoreReader(model, tokenizer, "owid_v2.pt", "store.json")
        out = r.read(h, ids)  # h: [1, T, d] layer-L states; ids: [1, T]
        out = r.read_text(q)

    Return top-k values, decoded series/years, units, mass and abstain.
    abstain = 1 - mass; low mass can reflect a missing key or uncertain decoding."""

    def __init__(self, model, tokenizer, adapter, store, licence="all", soft=0.1, k=5):
        from kgmem.memlm import MemoryLM
        sd = torch.load(adapter, map_location="cpu", weights_only=True)
        self.mlm = MemoryLM(model, tokenizer, MemConfig(**{**sd["cfg"], "rel_exists_soft": soft}),
                            layer_idx=sd["layer_idx"])
        self.mlm.question_only, self.mlm.unpadded = True, True
        self.mlm.load(adapter)
        self.store, self.k = Store(store, licence), k
        self.store.load(self.mlm)
        self.mlm.ctx.enabled = False

    @torch.no_grad()
    def read(self, h, ids):
        self.mlm.read(h, ids, k=self.k)
        return decode(self.mlm, self.store, self.k)

    @torch.no_grad()
    def read_text(self, question):
        ids = torch.tensor([self.mlm.bos + self.mlm.tok(prompt(question), add_special_tokens=False).input_ids],
                           device=self.mlm.device)
        return self.read(self.mlm.hidden_at_layer(ids, torch.ones_like(ids)), ids)


@torch.no_grad()
def baseline(a):
    """Score lettered options from logits for " A" through " D", with memory off."""
    mlm = build(a, [])
    mlm.ctx.enabled = False
    letters = [mlm.tok(" " + c, add_special_tokens=False).input_ids[-1] for c in "ABCD"]
    res = {"args": vars(a)}
    for name in SETS[:-1]:
        rows = read_jsonl(os.path.join(a.questions, f"{name}.{a.suffix}.jsonl"), a.eval_n)
        hit = 0
        for r in rows:
            text = f"Question: {r['question']}\n" + "".join(f"{c}. {o}\n" for c, o in zip("ABCD", r["options"])) \
                   + "Answer:"
            ids = torch.tensor([mlm.bos + mlm.tok(text, add_special_tokens=False).input_ids], device=mlm.device)
            lg = mlm.lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
            hit += int(lg[letters].argmax()) == r["gold"]
        res[name] = dict(n=len(rows), mc=round(100 * hit / max(1, len(rows)), 2))
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(ensure_parent(a.out), "w"), indent=1)


def evaluate_main(a):
    store = Store(a.store, a.licence)
    mlm = build(a, [])
    store.load(mlm)
    res = {"args": vars(a)} | run_sets(a, mlm, store)
    json.dump(res, open(ensure_parent(a.out), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "baseline"])
    ap.add_argument("--store", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--backbone-loader", dest="backbone_loader", default="",
                    help='optional Python loader exporting register() -> attention name and require_policy(text_config, policy); uses CUDA bf16')
    ap.add_argument("--policy", default="B")
    ap.add_argument("--layer", type=int, default=41)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--cfg", default="{}", help="MemConfig overrides (JSON)")
    ap.add_argument("--suffix", default="gated", help="question files <set>.<suffix>.jsonl (gated: after stored gate verdict filtering)")
    ap.add_argument("--licence", default="all", choices=["all", "clean", "clean_no_vdem"],
                    help="clean: CC BY / CC0 / OGL series only; clean_no_vdem: also without the V-Dem series")
    ap.add_argument("--neg_frac", type=float, default=0.0, help="share of training rows from neg_train (NULL)")
    ap.add_argument("--neg_tag", default="", help="neg_train<tag>.<suffix>.jsonl (_v2: the hard negatives)")
    ap.add_argument("--dump", default="", help="write per-item reads to <dump>_<set>.jsonl")
    ap.add_argument("--soft", type=float, default=0.1, help="rel_exists_soft at evaluation")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--dev_n", type=int, default=400)
    ap.add_argument("--eval_n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(train=train, eval=evaluate_main, baseline=baseline)[a.cmd](a)


if __name__ == "__main__":
    main()
