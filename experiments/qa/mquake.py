"""Knowledge editing evaluation on MQuAKE graphs.

  python -m experiments.qa.mquake prep --data data/mquake --out runs/mquake/prep.pt
  python -m experiments.qa.mquake train --prep runs/mquake/prep.pt --init checkpoints/smollm2-v9.pt --out runs/mquake/adapter.pt
  python -m experiments.qa.mquake eval --prep runs/mquake/prep.pt --adapter runs/mquake/adapter.pt --k 1,100,all,masked

prep combines original chain facts with unedited facts needed by edited
chains. Each test file's facts take precedence in its graph. Training excludes
cases sharing a question or chain with any loaded test file, then reserves dev
cases. Optional Remastered rewrites are matched by chain, not case ID.

Evaluation applies functional (subject, relation) edits in batches of k.
With k=all, edits can conflict with facts another case needs; a gold path may
then disagree with its labeled answer. masked mode restores each case's chain
facts before applying that case's edits. The protocols answer different graphs.

Scores use the top retrieved entity, matched by ID or normalized name/alias.
--generate also reports exact normalized decoded answers on a case sample.
"""
import argparse
import json
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import torch

from kgmem import MemConfig
from kgmem.paths import checkpoint_path, ensure_parent
from kgmem.graph import KnowledgeGraph
from kgmem.memlm import Batch
from kgmem.tasks import _find_last, _pack, plain_batch
from experiments.qa.metaqa import METAQA_CFG, build, metaqa_cfg

TESTS = {"cf3k-v2": "MQuAKE-CF-3k-v2.json", "cf3k": "MQuAKE-CF-3k.json", "2002": "MQuAKE-2002.json",
         "hard": "MQuAKE-hard.json", "T": "MQuAKE-T.json",
         "rCF3k": "remastered/CF3k.parquet", "rT": "remastered/T.parquet"}


def load_cases(path):
    """Read MQuAKE JSON or flatten Remastered parquet fields into the same schema."""
    if path.suffix != ".parquet":
        return json.load(open(path))
    import pyarrow.parquet as pq
    out = []
    for c in pq.read_table(path).to_pylist():
        c["orig"] = dict(triples=c.pop("orig_triples"), triples_labeled=c.pop("orig_triples_labeled"),
                         **{k: c.pop(k) for k in ("new_triples", "new_triples_labeled", "edit_triples")})
        out.append(c)
    return out


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return re.sub(r"[\s.]+$", "", re.sub(r"\s+", " ", s))


def facts(cases):
    """Collect original facts and unedited facts used by edited chains, keyed by (s, r)."""
    out = {}
    for c in cases:
        o = c["orig"]
        edited = {tuple(x) for x in o["edit_triples"]}
        for s, r, t in [tuple(x) for x in o["triples"]] + [tuple(x) for x in o["new_triples"] if tuple(x) not in edited]:
            out.setdefault((s, r), t)
    return out


def walk(kb, topic, path):
    for r in path:
        topic = kb.get((topic, r))
        if topic is None:
            return None
    return topic


def prep(args):
    data = Path(args.data).expanduser()
    cf = json.load(open(data / "MQuAKE-CF.json"))
    tests = {k: load_cases(data / f) for k, f in TESTS.items() if (data / f).exists()}
    label, rel_label = {}, {}
    for cases in [cf, *tests.values()]:
        for c in cases:
            o = c["orig"]
            for key in ("triples", "new_triples"):
                for (s, r, t), (sl, rl, tl) in zip(o[key], o[key + "_labeled"]):
                    label.setdefault(s, sl); label.setdefault(t, tl); rel_label.setdefault(r, rl)
    qids = sorted(label)
    pids = sorted(rel_label, key=lambda p: int(p[1:]))
    eid, rid = {q: i for i, q in enumerate(qids)}, {p: i for i, p in enumerate(pids)}
    enc = lambda kb: {(eid[s], rid[r]): eid[t] for (s, r), t in kb.items()}
    base_cf = facts(cf)
    # Exclude both repeated questions and repeated chains from the training pool.
    seen_q = {q for cases in tests.values() for c in cases for q in c["questions"]}
    chain = lambda c: tuple(map(tuple, c["orig"]["triples"])) + tuple(map(tuple, c["orig"]["new_triples"]))
    seen_c = {chain(c) for cases in tests.values() for c in cases}
    train_cases = [c for c in cf if not set(c["questions"]) & seen_q and chain(c) not in seen_c]
    rng = random.Random(0)
    rng.shuffle(train_cases)
    dev_cases, train_cases = train_cases[:args.dev_cases], train_cases[args.dev_cases:]
    train_kb = enc(base_cf)

    def items(cases):
        out, seen = [], set()
        for c in cases:
            o = c["orig"]
            topic, path = eid[o["triples"][0][0]], tuple(rid[r] for _, r, _ in o["triples"])
            for q in c["questions"]:
                out.append((q, topic, path, walk(train_kb, topic, path)))
            for hops, key in ((c["single_hops"], "triples"), (c["new_single_hops"], "new_triples")):
                for h, (s, r, _) in zip(hops, o[key]):
                    if h["question"] not in seen:
                        seen.add(h["question"])
                        out.append((h["question"], eid[s], (rid[r],), walk(train_kb, eid[s], (rid[r],))))
        return [x for x in out if x[3] is not None]

    def test_case(c):
        o = c["orig"]
        gold = lambda a, al: sorted({norm(x) for x in [a, *al]})
        return dict(case_id=c["case_id"], qs=list(c["questions"]), topic=eid[o["triples"][0][0]],
                    path=[rid[r] for _, r, _ in o["triples"]], edits=[(eid[s], rid[r], eid[t]) for s, r, t in
                                                                        o["edit_triples"]],
                    new=gold(c["new_answer"], c["new_answer_alias"]), new_id=eid[o["new_triples"][-1][2]],
                    orig=gold(c["answer"], c["answer_alias"]), orig_id=eid[o["triples"][-1][2]],
                    chain=[(eid[s], rid[r], eid[t]) for s, r, t in o["new_triples"]])
    dev = [test_case(c) for c in dev_cases]
    report = dict(entities=len(qids), relations=len(pids), facts_cf=len(base_cf), train_cases=len(train_cases),
                  dev_cases=len(dev))
    P = dict(entities=[label[q] for q in qids], qids=qids, relations=[rel_label[p] for p in pids], pids=pids,
             train_kb=train_kb, train=items(train_cases), dev=dev, dev_kb=train_kb, tests={}, report=report)
    report["train_questions"] = dict(Counter(f"{len(x[2])}-hop" for x in P["train"]))
    # Match Remastered rewrites by chain because case IDs changed.
    P["train_rewrites"] = []
    if (data / "remastered" / "CF9k.parquet").exists():
        rewritten = {chain(c): c["questions"] for c in load_cases(data / "remastered" / "CF9k.parquet")}
        for c in train_cases:
            o = c["orig"]
            topic, path = eid[o["triples"][0][0]], tuple(rid[r] for _, r, _ in o["triples"])
            for q in rewritten.get(chain(c), []):
                P["train_rewrites"].append((q, topic, path, walk(train_kb, topic, path)))
        P["train_rewrites"] = [x for x in P["train_rewrites"] if x[3] is not None]
        report["train_rewrites"] = len(P["train_rewrites"])
    for name, cases in tests.items():
        kb = dict(base_cf)
        kb.update(facts(cases))
        P["tests"][name] = dict(kb=enc(kb), cases=[test_case(c) for c in cases])
        report[name] = dict(cases=len(cases), facts=len(kb))
    torch.save(P, ensure_parent(args.out))
    print(json.dumps(report))


def apply_edits(kb, cases):
    kb = dict(kb)
    for c in cases:
        for s, r, t in c["edits"]:
            kb[(s, r)] = t
    return kb


def masked_kb(base, every, c):
    """Apply all edits, restore this case's chain facts, then reapply its own edits."""
    kb = dict(base)
    kb.update(every)
    kb.update({(s, r): t for s, r, t in c["chain"]})
    return apply_edits(kb, [c])


def set_graph(kg, kb):
    kg.triples = [(s, r, t) for (s, r), t in kb.items()]
    kg._version += 1


def grow_hops(block, H):
    """Extend hop-specific parameters by copying the last trained hop.

    Copy per-hop output rows, lowering added halting biases by one. Remap
    halt_rel feature columns to the larger hop count.
    """
    sd = dict(block)
    h0 = sd["kg.halt.weight"].shape[0]
    if h0 >= H:
        return sd
    for k in list(sd):
        m = re.match(r"(kg\.(?:rel|relm\.aq))\.(\d+)\.(.*)", k)
        if m and int(m.group(2)) == h0 - 1:
            for h in range(h0, H):
                sd[f"{m.group(1)}.{h}.{m.group(3)}"] = sd[k].clone()
    for k in ("kg.relm.mix", "kg.halt"):
        if f"{k}.weight" in sd:
            w, b = sd[f"{k}.weight"], sd[f"{k}.bias"]
            sd[f"{k}.weight"] = torch.cat([w] + [w[-1:]] * (H - h0))
            sd[f"{k}.bias"] = torch.cat([b] + [b[-1:] - (1.0 if k == "kg.halt" else 0.0)] * (H - h0))
    if "kg.halt_rel.weight" in sd:                       # features: H confidences, H-1 overlaps, H-1 same-relation
        w, b = sd["kg.halt_rel.weight"], sd["kg.halt_rel.bias"]
        same = w.shape[1] == 3 * h0 - 2
        old = list(range(h0)) + [h0 + j for j in range(h0 - 1)] + ([2 * h0 - 1 + j for j in range(h0 - 1)] if same else [])
        new = list(range(h0)) + [H + j for j in range(h0 - 1)] + ([2 * H - 1 + j for j in range(h0 - 1)] if same else [])
        W = torch.zeros(H, 2 * H - 1 + (H - 1 if same else 0), dtype=w.dtype)
        W[:h0, new] = w[:, old]
        W[h0:] = W[h0 - 1:h0]
        sd["kg.halt_rel.weight"], sd["kg.halt_rel.bias"] = W, torch.cat([b] + [b[-1:]] * (H - h0))
    return sd


# Apply these inference settings over the saved adapter configuration.
# rel_exists acts without gradients; carry rules act while a name is being emitted.
# The dev-selected rel_named=30 gives spelled relations a 31-fold boost before normalization.
# rel_exists_soft=0.1 preserves low edge-availability mass instead of renormalizing it to one.
MQUAKE_INFERENCE = dict(rel_exists=True, carry_rel=True, carry_fresh=0.3, rel_named=30.0, rel_exists_soft=0.1)


def mquake_cfg(base=None, **over):
    """Merge base configuration, MetaQA defaults, four-hop decoder settings, and overrides."""
    return MemConfig(**{**(base or {}), **METAQA_CFG, "hops": 4, "rel_exists": True, "rel_decoder": 256, **over})


def prose_pool(mlm, n=4000, length=96, split="train"):
    """Return up to n WikiText-2 paragraphs as token lists truncated to length."""
    from datasets import load_dataset
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]
    out = []
    for t in rows:
        t = t.strip()
        if len(t) >= 200 and not t.startswith("="):
            out.append(mlm.tok(t, add_special_tokens=False).input_ids[:length])
        if len(out) == n:
            break
    return out


def mquake_batch(mlm, ents, by_hops, rng, B=8, n_qa=3):
    """Pack n_qa questions per row, sampling hop counts uniformly.

    Use unedited-graph answers with topic-mention and relation-path supervision.
    """
    tok = mlm.tok
    nl = tok("\n", add_special_tokens=False).input_ids
    examples, spans, aux = [], [], []
    H = mlm.cfg.hops
    keys = sorted(by_hops)
    for b in range(B):
        ids, labels = list(mlm.bos), [-100] * len(mlm.bos)
        for _ in range(n_qa):
            q, topic, path, ans = rng.choice(by_hops[rng.choice(keys)])
            p = tok(f"Q: {q}\nA:", add_special_tokens=False).input_ids
            a = mlm.tokenize_name(ents[ans])
            mention = _find_last(p, mlm.tokenize_name(ents[topic]))
            base = len(ids)
            ids += p; labels += [-100] * len(p)
            aux.append((b, len(ids) - 1, topic, list(path), len(path), base + mention if mention >= 0 else -1))
            spans.append((b, len(ids), len(ids) + len(a), f"{len(path)}-hop"))
            ids += a + nl; labels += a + nl
        examples.append((ids, labels, None))
    ids, labels, attn = _pack(examples, tok.pad_token_id or 0, mlm.device)
    T = ids.shape[1]
    rel = torch.full((len(aux), H), -100, dtype=torch.long)
    for i, x in enumerate(aux):
        rel[i, :len(x[3])] = torch.tensor(x[3], dtype=torch.long)
    dev = mlm.device
    none = torch.full((len(aux), H), -1, dtype=torch.long, device=dev)
    return Batch(ids, labels, attn, spans,
                 aux_idx=torch.tensor([b * T + t for (b, t, *_) in aux], device=dev),
                 aux_ent=torch.tensor([x[2] for x in aux], device=dev), aux_rel=rel.to(dev),
                 aux_hops=torch.tensor([x[4] - 1 for x in aux], device=dev),
                 aux_mention=torch.tensor([x[5] for x in aux], device=dev),
                 aux_rel_mention=none, aux_rel_span_start=none)


@torch.no_grad()
def top_entities(mlm, questions, n_ent, B=64):
    """Return the highest-weight retrieved entity at each answer position."""
    tok, dev = mlm.tok, mlm.device
    pad, out = tok.pad_token_id or 0, []
    for i in range(0, len(questions), B):
        seqs = [mlm.bos + tok(f"Q: {q}\nA:", add_special_tokens=False).input_ids for q in questions[i:i + B]]
        T = max(map(len, seqs))
        ids = torch.tensor([s + [pad] * (T - len(s)) for s in seqs], device=dev)
        attn = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], device=dev)
        mlm.forward(ids, attn, keep=1)
        last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        rows = torch.arange(len(seqs), device=dev)
        cid, cw = mlm.ctx.cand_ids[rows, last], mlm.ctx.cand_w[rows, last].float()
        out += torch.zeros(len(seqs), n_ent, device=dev).scatter_add_(1, cid, cw).argmax(-1).tolist()
    return out


def score_cases(cases, preds, ents, which="new"):
    """Score any-question case accuracy, all-question accuracy, and per-question accuracy."""
    norm_ents = {}
    hit = lambda p, c: p == c[f"{which}_id"] or norm_ents.setdefault(p, norm(ents[p])) in set(c[which])
    i, per_case = 0, []
    for c in cases:
        n = len(c["qs"])
        per_case.append([hit(p, c) for p in preds[i:i + n]])
        i += n
    by_hops = defaultdict(list)
    for c, h in zip(cases, per_case):
        by_hops[len(c["path"])].append(any(h))
    N = max(1, len(cases))
    return dict(multi_hop=round(sum(map(any, per_case)) / N, 4), all_three=round(sum(map(all, per_case)) / N, 4),
                per_question=round(sum(map(sum, per_case)) / max(1, sum(map(len, per_case))), 4),
                by_hops={f"{h}-hop": round(sum(v) / len(v), 4) for h, v in sorted(by_hops.items())}), per_case


def run_k(mlm, kg, base, cases, ents, k, B=64):
    """Apply edits in groups of k and retrieve answers for each group's questions.

    k=None applies every case's edits at once. k="masked" restores each case's
    chain facts before reapplying its edits, preventing other edits from replacing
    facts that its labeled chain needs.
    """
    if k == "masked":
        every, preds = apply_edits({}, cases), []
        for c in cases:
            set_graph(kg, masked_kb(base, every, c))
            preds += top_entities(mlm, c["qs"], len(ents), B)
        return preds
    groups = [cases] if k is None else [cases[i:i + k] for i in range(0, len(cases), k)]
    preds = []
    for g in groups:
        set_graph(kg, apply_edits(base, g))
        preds += top_entities(mlm, [q for c in g for q in c["qs"]], len(ents), B)
    return preds


def load_adapter(args):
    sd = torch.load(args.adapter, map_location="cpu", weights_only=True)
    args.layer = sd["layer_idx"]
    cfg = {**sd["cfg"], **MQUAKE_INFERENCE, **json.loads(getattr(args, "cfg", None) or "{}")}
    mlm, _ = build(args, MemConfig(**cfg))
    mlm.load(args.adapter)
    mlm.block.eval()
    return mlm


def evaluate(args):
    P = torch.load(args.prep, weights_only=False)
    ents = P["entities"]
    mlm = load_adapter(args)
    kg = KnowledgeGraph(list(ents), list(P["relations"]), [])
    set_graph(kg, P["train_kb"])
    t0 = time.time()
    mlm.load_graph(kg)
    print(f"graph keys for {len(ents)} entities in {time.time() - t0:.0f}s", flush=True)
    res = {}
    for name in (P["tests"] if args.sets == "all" else args.sets.split(",")):
        T = P["tests"][name]
        cases = T["cases"][:args.limit] if args.limit else T["cases"]
        for k in args.k.split(","):
            t0 = time.time()
            preds = run_k(mlm, kg, T["kb"], cases, ents, None if k == "all" else k if k == "masked" else int(k))
            res[f"{name} k={k}"], per_case = score_cases(cases, preds, ents)
            print(f"{name} k={k}: {json.dumps(res[f'{name} k={k}'])} ({time.time() - t0:.0f}s)", flush=True)
            H = held_out(P, args.hold_out_rel)
            if H:
                res[f"{name} k={k} by relation"] = subset_scores(cases, preds, ents, H)
                print(f"{name} k={k} by relation: {json.dumps(res[f'{name} k={k} by relation'])}", flush=True)
            if args.dump and k == "all":
                Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
                with open(ensure_parent(f"{args.dump}.{name}.jsonl"), "w") as f:
                    for i, (c, h) in enumerate(zip(cases, per_case)):
                        f.write(json.dumps(dict(case_id=c["case_id"], ok=h, pred=[ents[p] for p in
                                                preds[3 * i:3 * i + len(c['qs'])]], new=c["new"][:3])) + "\n")
        if args.unedited:
            preds = run_k(mlm, kg, T["kb"], [dict(c, edits=[]) for c in cases], ents, None)
            res[f"{name} unedited"], _ = score_cases(cases, preds, ents, "orig")
            print(f"{name} unedited (no edits, original answers): {json.dumps(res[f'{name} unedited'])}", flush=True)
        if args.generate:
            set_graph(kg, apply_edits(T["kb"], cases))
            sample = random.Random(0).sample(cases, min(args.generate, len(cases)))
            ok = []
            for c in sample:
                got = [norm(mlm.generate(f"Q: {q}\nA:", max_new_tokens=24)) for q in c["qs"]]
                ok.append(any(g in set(c["new"]) for g in got))
            res[f"{name} generated k=all (n={len(sample)})"] = round(sum(ok) / len(ok), 4)
            print(f"{name} decoded, k=all, {len(sample)} cases: {res[f'{name} generated k=all (n={len(sample)})']}",
                  flush=True)
    print(json.dumps(res))


def held_out(P, names):
    """Relation ids named in a comma-separated list (relation names as in the graph), or an empty set."""
    names = [n.strip() for n in (names or "").split(",") if n.strip()]
    missing = [n for n in names if n not in P["relations"]]
    assert not missing, f"no such relations: {missing}"
    return {P["relations"].index(n) for n in names}


def subset_scores(cases, preds, ents, H):
    """Multi-hop accuracy over the cases whose chain uses a relation in H, and over the rest."""
    out, i, parts = {}, 0, {"held-out relation": ([], []), "seen relations": ([], [])}
    for c in cases:
        n = len(c["qs"])
        key = "held-out relation" if set(c["path"]) & H else "seen relations"
        parts[key][0].append(c)
        parts[key][1].extend(preds[i:i + n])
        i += n
    for key, (cs, ps) in parts.items():
        if cs:
            out[key] = dict(cases=len(cs), multi_hop=score_cases(cs, ps, ents)[0]["multi_hop"])
    return out


def dev_scores(mlm, kg, P, n):
    """Score dev cases with all edits applied and gold IDs recomputed by path traversal."""
    cases = P["dev"][:n]
    edited = apply_edits(P["dev_kb"], cases)
    gold = [dict(c, new_id=walk(edited, c["topic"], c["path"]), new=[]) for c in cases]
    preds = run_k(mlm, kg, P["dev_kb"], cases, P["entities"], None)
    return score_cases(gold, preds, P["entities"])[0]


def train(args):
    P = torch.load(args.prep, weights_only=False)
    ents = P["entities"]
    init = torch.load(args.init, map_location="cpu", weights_only=True) if args.init else None
    args.layer = args.layer if args.layer is not None else (init["layer_idx"] if init else None)
    cfg = mquake_cfg(init["cfg"] if init else None, **json.loads(args.cfg))
    mlm, tok = build(args, cfg)
    if init:
        own = mlm.block.state_dict()
        sd = {k: v for k, v in grow_hops(init["block"], cfg.hops).items() if k in own and own[k].shape == v.shape}
        print("warm start:", mlm.block.load_state_dict(sd, strict=False))
    if args.train_only:
        keep = tuple(x for x in args.train_only.split(",") if x)
        for n, p in mlm.block.named_parameters():
            p.requires_grad_(n.startswith(keep))
    kg = KnowledgeGraph(list(ents), list(P["relations"]), [])
    set_graph(kg, P["train_kb"])
    t0 = time.time()
    mlm.load_graph(kg)
    print(f"graph: {len(ents)} entities, {len(P['relations'])} relations, {len(P['train_kb'])} facts, keys in "
          f"{time.time() - t0:.0f}s", flush=True)
    pool = {"none": P["train"], "add": P["train"] + P.get("train_rewrites", []),
            "only": [x for x in P["train"] if len(x[2]) == 1] + P.get("train_rewrites", [])}[args.rewrites]
    H = held_out(P, args.hold_out_rel)
    if H:                                       # relations kept out of training: no question and no dev case uses them
        pool = [x for x in pool if not set(x[2]) & H]
        P["dev"] = [c for c in P["dev"] if not set(c["path"]) & H]
        print(f"held out {sorted(P['relations'][r] for r in H)}: {len(pool)} training questions, "
              f"{len(P['dev'])} dev cases", flush=True)
    by_hops = defaultdict(list)
    for x in pool:
        by_hops[len(x[2])].append(x)
    print({f"{h}-hop": len(v) for h, v in sorted(by_hops.items())}, flush=True)
    prose = prose_pool(mlm) if args.p_plain > 0 else None
    params = mlm.trainable_parameters()
    opt = torch.optim.AdamW(params, args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    rng, hist, t0 = random.Random(args.seed), [], time.time()
    meta = {"args": vars(args), "parent": init.get("meta") if init else None}
    best = -1.0
    for step in range(1, args.steps + 1):
        set_graph(kg, P["train_kb"])
        b = plain_batch(mlm, rng, prose, args.batch) if prose and rng.random() < args.p_plain else \
            mquake_batch(mlm, ents, by_hops, rng, args.batch)
        loss, logs = mlm.loss(b, args.aux, keep_base=args.keep_base)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        hist.append(logs)
        if step % 20 == 0:
            keys = {k for h in hist for k in h}
            avg = {k: sum(h[k] for h in hist if k in h) / sum(k in h for h in hist) for k in sorted(keys)}
            print(f"step {step:5d}  " + "  ".join(f"{k} {v:.3f}" for k, v in avg.items())
                  + f"  {(time.time() - t0) / step:.2f}s/step", flush=True)
            hist = []
        if step % args.eval_every == 0 or step == args.steps:
            mlm.block.eval()
            res = dev_scores(mlm, kg, P, args.dev_n)
            mlm.block.train()
            print(f"dev @ {step}: {json.dumps(res)}", flush=True)
            mlm.save(ensure_parent(args.out.replace(".pt", "_last.pt")), dict(meta, step=step, dev=res))
            if res["per_question"] > best:
                best = res["per_question"]
                mlm.save(ensure_parent(args.out), dict(meta, step=step, dev=res))
    print("saved", args.out, "best dev per-question", round(best, 4))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", default="data/mquake")
    p.add_argument("--out", default='runs/mquake/prep.pt')
    p.add_argument("--dev_cases", type=int, default=300)
    for name in ("train", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--prep", default='runs/mquake/prep.pt')
        s.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
        s.add_argument("--dtype", default="float32")
        s.add_argument("--layer", type=int, default=None)
        s.add_argument("--cfg", default="{}", help="JSON overrides of the config (mquake_cfg() when training)")
        s.add_argument("--hold_out_rel", default="", help='comma-separated relation names excluded from training and reported separately at evaluation')
    t = sub.choices["train"]
    t.add_argument("--init", default=None)
    t.add_argument("--train_only", default=None)
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--aux", type=float, default=1.0)
    t.add_argument("--keep_base", type=float, default=0.0)
    t.add_argument("--p_plain", type=float, default=0.0, help='fraction of steps minimizing plain-text KL on WikiText-2 with the graph loaded')
    t.add_argument("--eval_every", type=int, default=500)
    t.add_argument("--dev_n", type=int, default=300)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--rewrites", default="none", choices=["none", "add", "only"],
                   help='use Remastered training rewrites: omit, add, or replace originals; keep single-hop questions')
    t.add_argument("--out", default='runs/mquake/adapter.pt')
    e = sub.choices["eval"]
    e.add_argument("--adapter", default=checkpoint_path("mquake.pt"))
    e.add_argument("--sets", default="all", help='comma-separated test sets, or all sets in the prep file')
    e.add_argument("--k", default="1,100,1000,all")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--unedited", action="store_true", help='also score original answers with graph edits disabled')
    e.add_argument("--generate", type=int, default=0)
    e.add_argument("--dump", default=None, help="write per-case predictions at k=all to DUMP.<set>.jsonl")
    args = ap.parse_args()
    {"prep": prep, "train": train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
