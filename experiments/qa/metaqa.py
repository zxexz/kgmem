"""Prepare, train, and evaluate MetaQA over the WikiMovies knowledge base.

  python -m experiments.qa.metaqa prep --data data/metaqa --out runs/metaqa/prep.pt
  python -m experiments.qa.metaqa train --prep runs/metaqa/prep.pt --init checkpoints/smollm2-v9.pt --out runs/metaqa/adapter.pt
  python -m experiments.qa.metaqa eval --prep runs/metaqa/prep.pt --adapter runs/metaqa/adapter.pt
  python -m experiments.qa.metaqa edit --prep runs/metaqa/prep.pt --adapter runs/metaqa/adapter.pt
  python -m experiments.qa.metaqa counterfactual --prep runs/metaqa/prep.pt --adapter runs/metaqa/adapter.pt

Read kb.txt and {1,2,3}-hop/vanilla/qa_{train,dev,test}.txt. Add inverse edges
and recover template-level relation paths from training question samples.
Paths supervise training; evaluation ranks retrieved entities without them.
Questions retain their topic mention but omit the release's brackets.

Hits@1 checks whether the highest-weight retrieved entity is a gold answer.
--generate additionally checks decoded entity names on a sample per hop count.
"""
import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kgmem import MemConfig, MemoryLM
from kgmem.paths import checkpoint_path, ensure_parent
from kgmem.graph import KnowledgeGraph
from kgmem.memlm import Batch
from kgmem.tasks import _find_last, _pack

# MetaQA relation -> (name read forwards, name of the inverse); keys and the relation-mention linker use the names
RELS = [("directed_by", "directed by", "director of"), ("written_by", "written by", "writer of"),
        ("starred_actors", "starred actors", "acted in"), ("release_year", "release year", "released in"),
        ("in_language", "in language", "language of"), ("has_tags", "has tags", "tag of"),
        ("has_genre", "has genre", "genre of"), ("has_imdb_votes", "imdb votes", "imdb votes of"),
        ("has_imdb_rating", "imdb rating", "imdb rating of")]
NREL = len(RELS)


def load_kb(path):
    ents, idx, triples = [], {}, set()
    fwd = {r: i for i, (r, _, _) in enumerate(RELS)}

    def eid(n):
        if n not in idx:
            idx[n] = len(ents)
            ents.append(n)
        return idx[n]
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) != 3 or parts[1] not in fwd:
            continue
        h, t, r = eid(parts[0]), eid(parts[2]), fwd[parts[1]]
        triples.add((h, r, t))
        triples.add((t, r + NREL, h))
    names = [f for _, f, _ in RELS] + [b for _, _, b in RELS]
    return ents, names, sorted(triples)


def load_qa(path, ent_index):
    out, skipped = [], 0
    for line in open(path, encoding="utf-8"):
        q, a = line.rstrip("\n").split("\t")
        m = re.search(r"\[([^\]]+)\]", q)
        answers = [ent_index[x] for x in a.split("|") if x in ent_index]
        if m is None or m.group(1) not in ent_index or not answers:
            skipped += 1
            continue
        template = q[:m.start()] + "NE" + q[m.end():]
        out.append((q.replace("[", "").replace("]", ""), ent_index[m.group(1)], answers, template))
    return out, skipped


def infer_paths(items, adj, out_rels, hops, per_template=20, seed=0):
    """Infer each template's relation path by votes over sampled training questions.

    A path matches when its result, excluding the topic, equals the gold answer
    set. Intermediate hops also exclude the topic. Return paths and their share
    of matching sampled questions.
    """
    rng = random.Random(seed)
    by_t = defaultdict(list)
    for it in items:
        by_t[it[3]].append(it)
    paths, agree = {}, {}
    for t, its in by_t.items():
        votes = Counter()
        sample = rng.sample(its, min(per_template, len(its)))
        for _, topic, answers, _ in sample:
            gold = set(answers)
            frontier = [((), {topic})]
            for step in range(hops):
                nxt = []
                for path, ents in frontier:
                    for r in {r for x in ents for r in out_rels[x]}:
                        res = set().union(*(adj.get((x, r), ()) for x in ents))
                        if step < hops - 1:
                            res = res - {topic}          # MetaQA's paths never pass back through the topic
                        if res:
                            nxt.append((path + (r,), res))
                frontier = nxt
            for path, res in frontier:
                if res - {topic} == gold:
                    votes[path] += 1
        if votes:
            best, n = votes.most_common(1)[0]
            paths[t], agree[t] = best, n / len(sample)
    return paths, agree


def prep(args):
    data = Path(args.data).expanduser()
    ents, rels, triples = load_kb(data / "kb" / "kb.txt" if (data / "kb").exists() else data / "kb.txt")
    ent_index = {n: i for i, n in enumerate(ents)}
    adj, out_rels = defaultdict(set), defaultdict(set)
    for h, r, t in triples:
        adj[(h, r)].add(t)
        out_rels[h].add(r)
    qa, report = {}, {"entities": len(ents), "relations": len(rels), "triples": len(triples)}
    for hops in (1, 2, 3):
        qa[hops] = {}
        for split in ("train", "dev", "test"):
            items, skipped = load_qa(data / f"{hops}-hop" / "vanilla" / f"qa_{split}.txt", ent_index)
            qa[hops][split] = items
            report[f"{hops}-hop {split}"] = dict(questions=len(items), skipped=skipped)
        t0 = time.time()
        paths, agree = infer_paths(qa[hops]["train"], adj, out_rels, hops)
        covered = sum(1 for it in qa[hops]["train"] if it[3] in paths)
        report[f"{hops}-hop paths"] = dict(templates=len({it[3] for it in qa[hops]['train']}), with_path=len(paths),
                                         train_covered=round(covered / len(qa[hops]["train"]), 4),
                                         mean_agreement=round(sum(agree.values()) / max(1, len(agree)), 3),
                                         seconds=round(time.time() - t0, 1))
        qa[hops]["paths"] = paths
        print(json.dumps({k: v for k, v in report.items() if k.startswith(str(hops))}), flush=True)
    torch.save(dict(entities=ents, relations=rels, triples=triples, qa=qa, report=report), ensure_parent(args.out))
    print(json.dumps(report))


def make_kg(P):
    return KnowledgeGraph(list(P["entities"]), list(P["relations"]), [tuple(t) for t in P["triples"]])


METAQA_CFG = dict(hops=3, use_scratchpad=False, rel_mention=True, rel_key_pool="last", relm_raw_cos=30.0,
                  relm_first=True, rel_halt=True, carry_share=0.2, end_gate=True, halt_rel_aux=1.0, halt_rel_same=True,
                  carry_commit=True, exclude_head=True, exact_link=0.9, k_cand=64, max_tails=64, carry_top=0.5)


def metaqa_cfg(**over):
    return MemConfig(**{**METAQA_CFG, **over})


def metaqa_batch(mlm, ents, pools, rng, B=8, n_qa=3, rel_sup=True):
    """Pack n_qa MetaQA questions per row with sampled gold-answer targets.

    Supervise the topic mention and recovered template relation path when present.
    """
    tok = mlm.tok
    nl = tok("\n", add_special_tokens=False).input_ids
    examples, spans, aux = [], [], []
    H = mlm.cfg.hops
    for b in range(B):
        ids, labels = list(mlm.bos), [-100] * len(mlm.bos)
        for _ in range(n_qa):
            hops = rng.choice((1, 2, 3))
            items, paths = pools[hops]
            q, topic, answers, template = rng.choice(items)
            p = tok(f"Q: {q}\nA:", add_special_tokens=False).input_ids
            a = mlm.tokenize_name(ents[rng.choice(answers)])
            mention = _find_last(p, mlm.tokenize_name(ents[topic]))
            base = len(ids)
            ids += p; labels += [-100] * len(p)
            path = list(paths.get(template, ())) if rel_sup else []        # answers only: no relation labels
            aux.append((b, len(ids) - 1, topic, path, hops, base + mention if mention >= 0 else -1))
            spans.append((b, len(ids), len(ids) + len(a), f"{hops}-hop"))
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
def hits_at_1(mlm, items, n_ent, B=64, limit=None):
    """Top entity the graph head retrieves at the answer position (aggregated over hops) is a gold answer."""
    tok, dev = mlm.tok, mlm.device
    was = mlm.block.training
    mlm.block.eval()
    items = items[:limit] if limit else items
    hit = 0
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        seqs = [mlm.bos + tok(f"Q: {q}\nA:", add_special_tokens=False).input_ids for q, *_ in chunk]
        T = max(map(len, seqs))
        pad = tok.pad_token_id or 0
        ids = torch.tensor([s + [pad] * (T - len(s)) for s in seqs], device=dev)
        attn = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], device=dev)
        mlm.forward(ids, attn, keep=1)
        last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        rows = torch.arange(len(chunk), device=dev)
        cid, cw = mlm.ctx.cand_ids[rows, last], mlm.ctx.cand_w[rows, last].float()
        dense = torch.zeros(len(chunk), n_ent, device=dev).scatter_add_(1, cid, cw)
        pred = dense.argmax(-1).tolist()
        hit += sum(p in set(it[2]) for p, it in zip(pred, chunk))
    mlm.block.train(was)
    return hit / max(1, len(items))


@torch.no_grad()
def generate_hits(mlm, items, ents, n, seed=0):
    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))
    ok = 0
    for q, _, answers, _ in sample:
        got = mlm.generate(f"Q: {q}\nA:", max_new_tokens=24).strip()
        ok += got in {ents[a] for a in answers}
    return ok / max(1, len(sample))


@torch.no_grad()
def edit_eval(args):
    """Test graph edits on single-answer, one-hop questions without retraining.

    Replace each selected fact and its inverse with a tail used elsewhere for
    that relation. Measure decoded edit success and unchanged predictions for
    questions about other topics.
    """
    P = torch.load(args.prep, weights_only=False)
    ents = P["entities"]
    kg = make_kg(P)
    sd = torch.load(args.adapter, map_location="cpu", weights_only=True)
    args.layer = sd["layer_idx"]
    mlm, _ = build(args, MemConfig(**sd["cfg"]))
    mlm.load(args.adapter)
    mlm.block.eval()
    mlm.load_graph(kg)
    rng = random.Random(args.seed)
    out_deg = defaultdict(list)
    for h, r, t in kg.triples:
        out_deg[(h, r)].append(t)
    tails_of_rel = defaultdict(set)
    for h, r, t in kg.triples:
        tails_of_rel[r].add(t)
    paths = P["qa"][1]["paths"]
    cands = [it for it in P["qa"][1]["test"] if it[3] in paths and len(out_deg[(it[1], paths[it[3]][0])]) == 1]
    rng.shuffle(cands)
    edits, seen = [], set()
    for q, topic, answers, t in cands:
        if topic in seen or len(edits) >= args.n:
            continue
        r = paths[t][0]
        new = rng.choice(sorted(tails_of_rel[r] - {answers[0], topic}))
        edits.append((q, topic, r, answers[0], new))
        seen.add(topic)
    others = [it for it in rng.sample(P["qa"][1]["test"], 4 * args.n) if it[1] not in seen][:args.n]
    ask = lambda q: mlm.generate(f"Q: {q}\nA:", max_new_tokens=24).strip()
    before = [ask(q) for q, *_ in others]
    pre_ok = sum(ask(q) == ents[old] for q, _, _, old, _ in edits)
    for _, topic, r, old, new in edits:                         # both directions: the fact and its inverse
        kg.set(ents[topic], kg.relations[r], ents[new])
        inv = r + NREL if r < NREL else r - NREL
        kg.triples = [x for x in kg.triples if x != (old, inv, topic)] + [(new, inv, topic)]
        kg._version += 1
    ok = sum(ask(q) == ents[new] for q, _, _, _, new in edits)
    after = [ask(q) for q, *_ in others]
    res = dict(edits=len(edits), answered_old_fact_before=round(pre_ok / len(edits), 4),
               edit_success=round(ok / len(edits), 4),
               locality_unchanged=round(sum(a == b for a, b in zip(before, after)) / len(others), 4))
    print(json.dumps(res))


def walk(adj, topic, path):
    """Walk a relation path, excluding the topic after each hop."""
    fr = {topic}
    for i, r in enumerate(path):
        fr = set().union(*(adj.get((x, r), ()) for x in fr))
        if i < len(path) - 1:
            fr = fr - {topic}
    return fr - {topic}


@torch.no_grad()
def counterfactual_eval(args):
    """Evaluate recomputed answers after shuffling tails within each relation.

    Preserve names and forward-relation tail counts, rebuild inverse edges, and
    recompute nonempty gold sets along recovered paths. The adapter is unchanged.
    """
    P = torch.load(args.prep, weights_only=False)
    ents, rels = P["entities"], P["relations"]
    rng = random.Random(args.seed)
    by_rel = defaultdict(list)
    for h, r, t in P["triples"]:
        if r < NREL:
            by_rel[r].append((h, t))
    triples = set()
    for r, ht in by_rel.items():
        tails = [t for _, t in ht]
        if not args.no_shuffle:
            rng.shuffle(tails)
        for (h, _), t in zip(ht, tails):
            triples.add((h, r, t))
            triples.add((t, r + NREL, h))
    triples = sorted(triples)
    adj = defaultdict(set)
    for h, r, t in triples:
        adj[(h, r)].add(t)
    kg = KnowledgeGraph(list(ents), list(rels), triples)
    sd = torch.load(args.adapter, map_location="cpu", weights_only=True)
    args.layer = sd["layer_idx"]
    mlm, _ = build(args, MemConfig(**sd["cfg"]))
    mlm.load(args.adapter)
    mlm.block.eval()
    mlm.load_graph(kg)
    res = {}
    for hops in (1, 2, 3):
        paths = P["qa"][hops]["paths"]
        items = []
        for q, topic, answers, t in P["qa"][hops][args.split]:
            if t in paths:
                gold = walk(adj, topic, paths[t])
                if gold:
                    items.append((q, topic, sorted(gold), t))
        items = random.Random(1).sample(items, min(args.n, len(items)))
        res[f"{hops}-hop"] = round(hits_at_1(mlm, items, len(ents)), 4)
        kind = "the real KB, recomputed answers" if args.no_shuffle else "the shuffled KB"
        print(f"{hops}-hop over {kind}: {res[f'{hops}-hop']} ({len(items)} questions)", flush=True)
    print(json.dumps(res))


def build(args, cfg):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 device_map=dev if dev != "cpu" else None).eval()
    return MemoryLM(model, tok, cfg, layer_idx=args.layer), tok


def train(args):
    P = torch.load(args.prep, weights_only=False)
    ents, kg = P["entities"], make_kg(P)
    init = torch.load(args.init, map_location="cpu", weights_only=True) if args.init else None
    layer = args.layer if args.layer is not None else (init["layer_idx"] if init else None)
    args.layer = layer
    cfg = metaqa_cfg(**json.loads(args.cfg))
    mlm, tok = build(args, cfg)
    if args.init:
        print("warm start:", mlm.load(args.init, strict=False))
    if args.train_only:
        keep = tuple(x for x in args.train_only.split(",") if x)
        for n, p in mlm.block.named_parameters():
            p.requires_grad_(n.startswith(keep))
    t0 = time.time()
    mlm.load_graph(kg)
    print(f"graph: {len(ents)} entities, {len(P['relations'])} relations, {len(P['triples'])} triples, keys in "
          f"{time.time() - t0:.0f}s", flush=True)
    pools = {h: (P["qa"][h]["train"], P["qa"][h]["paths"]) for h in (1, 2, 3)}
    dev_sets = {h: random.Random(1).sample(P["qa"][h]["dev"], min(args.dev_n, len(P["qa"][h]["dev"])))
                for h in (1, 2, 3)}
    params = mlm.trainable_parameters()
    opt = torch.optim.AdamW(params, args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    rng, hist, t0 = random.Random(args.seed), [], time.time()
    meta = {"args": vars(args), "parent": init.get("meta") if init else None}
    best = -1.0
    for step in range(1, args.steps + 1):
        b = metaqa_batch(mlm, ents, pools, rng, args.batch, rel_sup=not args.answers_only)
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
            res = {f"{h}-hop": round(hits_at_1(mlm, dev_sets[h], len(ents)), 4) for h in (1, 2, 3)}
            mean = sum(res.values()) / 3
            print(f"dev Hits@1 @ {step}: {json.dumps(res)}  mean {mean:.4f}", flush=True)
            mlm.save(ensure_parent(args.out.replace(".pt", "_last.pt")), dict(meta, step=step, dev=res))
            if mean > best:                       # Select checkpoints by dev accuracy.
                best = mean
                mlm.save(ensure_parent(args.out), dict(meta, step=step, dev=res))
    print("saved", args.out, "best dev mean", round(best, 4))


def evaluate(args):
    P = torch.load(args.prep, weights_only=False)
    ents, kg = P["entities"], make_kg(P)
    sd = torch.load(args.adapter, map_location="cpu", weights_only=True)
    args.layer = sd["layer_idx"]
    mlm, _ = build(args, MemConfig(**sd["cfg"]))
    mlm.load(args.adapter)
    mlm.block.eval()
    mlm.load_graph(kg)
    res = {}
    for h in (1, 2, 3):
        items = P["qa"][h][args.split]
        t0 = time.time()
        res[f"{h}-hop"] = round(hits_at_1(mlm, items, len(ents)), 4)
        if args.generate:
            res[f"{h}-hop generated (n={args.generate})"] = round(generate_hits(mlm, items, ents, args.generate), 4)
        print(f"{h}-hop {args.split}: {res[f'{h}-hop']} over {len(items)} questions ({time.time() - t0:.0f}s)", flush=True)
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", default="data/metaqa")
    p.add_argument("--out", default='runs/metaqa/prep.pt')
    for name in ("train", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--prep", default='runs/metaqa/prep.pt')
        s.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
        s.add_argument("--dtype", default="float32")
        s.add_argument("--layer", type=int, default=None)
    t = sub.choices["train"]
    t.add_argument("--init", default=None)
    t.add_argument("--cfg", default="{}", help="JSON overrides of metaqa_cfg(), e.g. '{\"k_cand\": 64}'")
    t.add_argument("--train_only", default=None)
    t.add_argument("--answers_only", action="store_true", help='omit per-hop relation labels; retain answer, topic, and hop-count supervision')
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--aux", type=float, default=1.0)
    t.add_argument("--keep_base", type=float, default=0.0)
    t.add_argument("--eval_every", type=int, default=500)
    t.add_argument("--dev_n", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default='runs/metaqa/adapter.pt')
    d = sub.add_parser("edit")
    d.add_argument("--prep", default='runs/metaqa/prep.pt')
    d.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    d.add_argument("--dtype", default="float32")
    d.add_argument("--layer", type=int, default=None)
    d.add_argument("--adapter", default=checkpoint_path("metaqa.pt"))
    d.add_argument("--n", type=int, default=500)
    d.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("counterfactual")
    for a_, d_ in (("--prep", 'runs/metaqa/prep.pt'), ("--model", "HuggingFaceTB/SmolLM2-135M"), ("--dtype", "float32"),
                   ("--adapter", checkpoint_path("metaqa.pt")), ("--split", "test")):
        c.add_argument(a_, default=d_)
    c.add_argument("--layer", type=int, default=None)
    c.add_argument("--n", type=int, default=2000)
    c.add_argument("--no_shuffle", action="store_true", help='use the original KB and recompute gold answer sets along recovered paths')
    c.add_argument("--seed", type=int, default=0)
    e = sub.choices["eval"]
    e.add_argument("--adapter", default=checkpoint_path("metaqa.pt"))
    e.add_argument("--split", default="test")
    e.add_argument("--generate", type=int, default=0)
    args = ap.parse_args()
    {"prep": prep, "train": train, "eval": evaluate, "edit": edit_eval,
     "counterfactual": counterfactual_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
