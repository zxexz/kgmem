"""Rerank kgmem's WebQSP candidates with an instruction model.

  python -m experiments.qa.webqsp_reader --prep runs/webqsp/prep.pt --dump runs/webqsp/predictions --model Qwen/Qwen2.5-7B-Instruct --ks 3,5,10 --out runs/webqsp/reader.json

Score candidate-label tokens after the question, supplied topic, and candidate
list. Optional --tta averages over seeded candidate permutations. Choose k,
view count, and whether to use the reader on dev, then score that choice on test.

--train_dumps fits LoRA on reader layers. Supply out-of-fold candidates from
webqsp.py train/eval --fold K:k; in-sample lists can overstate top-candidate
quality and teach a position shortcut. --labels letters uses letter labels;
the default uses digits. label_ids scores the final token of each label.

--evidence H lists every topic entity and, under each candidate, its shortest path of up to H hops from each topic
entity in the question's subgraph, so the reader can check a constraint ("... that uses the Euro") rather than guess it.
A reader trained without evidence should not be scored with it.
"""
import argparse
import collections
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kgmem.paths import ensure_parent

UNNAMED = re.compile(r"^[mg]\.[0-9a-z_]+$")
PROMPT = ("Answer the question using the candidate answers a knowledge graph search returned. Reply with the number "
          "of the best candidate only.\n\nQuestion: {q}\nTopic entity: {topic}\nCandidates:\n{cands}\nAnswer:")
PROMPT_EV = ("Answer the question using the candidate answers a knowledge graph search returned. Under each candidate "
             "are its paths from the topic entities in the graph. Reply with the number of the best candidate only."
             "\n\nQuestion: {q}\nTopic entities: {topic}\nCandidates:\n{cands}\nAnswer:")


def short_rel(r):
    """'location.country.currency_used' -> 'country currency used'."""
    return " ".join(r.split(".")[-2:]).replace("_", " ")


def evidence(P, it, names, hops):
    """Each named candidate's shortest path (up to hops) from each topic entity, as 'a -rel-> b' steps."""
    ents, rels = P["entities"], P.get("freebase_relations") or P["relations"]
    adj = collections.defaultdict(list)
    for h, r, t in it["tri"].tolist():
        adj[h].append((t, h, r, t))
        adj[t].append((h, h, r, t))
    want = set(names)
    show = lambda e: "(node)" if UNNAMED.match(ents[e]) else ents[e]
    out = collections.defaultdict(list)
    for src in it["topic"][:3]:
        prev, front, found = {src: None}, [src], {}
        for _ in range(hops):
            nxt = []
            for e in front:
                for n, h, r, t in adj[e]:
                    if n not in prev:
                        prev[n] = (e, h, r, t)
                        nxt.append(n)
                        if ents[n] in want and ents[n] not in found:
                            found[ents[n]] = n
            front = nxt
        for name, e in found.items():
            steps = []
            while prev[e] is not None:
                e0, h, r, t = prev[e]
                steps.append(f"{show(h)} -{short_rel(rels[r])}-> {show(t)}")
                e = e0
            out[name].append("; ".join(reversed(steps)))
    return dict(out)


def norm(s):
    from experiments.qa.webqsp_compare import normalize
    return normalize(s)


LETTERS = "ABCDEFGHIJKLMNOPQRST"


def label_ids(tok, k, labels):
    return [tok(f" {LETTERS[i] if labels == 'letters' else i + 1}", add_special_tokens=False).input_ids[-1]
            for i in range(k)]


def render(tok, it, k, labels):
    lab = (lambda j: LETTERS[j]) if labels == "letters" else (lambda j: str(j + 1))
    ev = it.get("ev")
    cands = "\n".join(f"{lab(j)}. {c}" + "".join(f"\n   {p}" for p in (ev.get(c, []) if ev is not None else []))
                      for j, c in enumerate(it["cands"][:k]))
    msg = (PROMPT if ev is None else PROMPT_EV).format(q=it["q"], topic=it["topic"], cands=cands)
    if labels == "letters":
        msg = msg.replace("Reply with the number", "Reply with the letter")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True) + "Answer:"


@torch.no_grad()
def choose(model, tok, items, k, bs=16, labels="digits", tta=1, seed=0, scores=None):
    """Pick candidates by label probability, averaged across tta list orders.

    Use the original order and tta - 1 seeded shuffles, mapping probabilities
    back to candidate IDs before averaging. Optionally append probabilities to
    scores for ensemble evaluation.
    """
    import random
    ids = label_ids(tok, k, labels)
    tok.padding_side = "left"
    rng = random.Random(seed)
    orders = []                                        # per item, per view: candidate index at each position
    for it in items:
        n = len(it["cands"][:k])
        views = [list(range(n))]
        for _ in range(tta - 1):
            v = list(range(n)); rng.shuffle(v); views.append(v)
        orders.append(views)
    flat = [(i, v) for i in range(len(items)) for v in range(len(orders[i]))]
    score = [torch.zeros(len(it["cands"][:k])) for it in items]
    for j in range(0, len(flat), bs):
        chunk = flat[j:j + bs]
        texts = [render(tok, dict(items[i], cands=[items[i]["cands"][c] for c in orders[i][v]]), k, labels)
                 for i, v in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1].float()
        for (i, v), lg in zip(chunk, logits):
            n = len(orders[i][v])
            if n:
                p = lg[ids[:n]].softmax(-1).cpu()
                for pos, c in enumerate(orders[i][v]):
                    score[i][c] += p[pos]
    if scores is not None:                             # each item's probability per candidate (for ensembles)
        scores.extend([(sc / max(1, tta)).tolist() for sc in score])
    return [it["cands"][int(sc.argmax())] if len(sc) else None for it, sc in zip(items, score)]


def train_reader(model, tok, items, k, labels, epochs, lr, r, bs=8, seed=0, accum=1):
    """Fit reader LoRA on the label of the first gold candidate in each training list."""
    import random
    from kgmem.lora import add_lora
    from kgmem.memlm import find_decoder_layers
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_decoder_layers(model)
    mods = add_lora(layers, len(layers) - 1, r, 2.0 * r)
    params = [p for m in mods.values() for p in (m.A, m.B)]
    opt = torch.optim.AdamW(params, lr, weight_decay=0.0)
    ids = torch.tensor(label_ids(tok, k, labels), device=model.device)
    data = [(it, next(j for j, c in enumerate(it["cands"][:k]) if norm(c) in it["gold"]))
            for it in items if any(norm(c) in it["gold"] for c in it["cands"][:k])]
    rng = random.Random(seed)
    tok.padding_side = "left"
    model.train()
    for ep in range(epochs):
        rng.shuffle(data)
        tot = n = 0
        for i in range(0, len(data), bs):
            chunk = data[i:i + bs]
            enc = tok([render(tok, it, k, labels) for it, _ in chunk], return_tensors="pt", padding=True).to(model.device)
            logits = model(**enc).logits[:, -1].float()
            sub = logits[:, ids]
            loss = torch.nn.functional.cross_entropy(sub, torch.tensor([j for _, j in chunk], device=model.device))
            (loss / accum).backward()
            if (i // bs + 1) % accum == 0 or i + bs >= len(data):
                opt.step(); opt.zero_grad()
            tot += float(loss) * len(chunk); n += len(chunk)
        print(f"reader epoch {ep + 1}: loss {tot / max(1, n):.4f} over {n} questions with a gold candidate in the top {k}",
              flush=True)
    model.eval()
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default='runs/webqsp/prep.pt')
    ap.add_argument("--dump", required=True, help='webqsp.py eval dump prefix; reads PREFIX.validation.jsonl and PREFIX.test.jsonl')
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ks", default="3,5,10")
    ap.add_argument("--overlap", default=None)
    ap.add_argument("--labels", default="digits", choices=["digits", "letters"])
    ap.add_argument("--train_dumps", default="", help='comma-separated dump prefixes containing out-of-fold training candidates')
    ap.add_argument("--train_k", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--train_max", type=int, default=0, help='training size target: retain all WebQSP questions and sample other sources to fill it')
    ap.add_argument("--bs", type=int, default=8, help='reader training batch size')
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--grad_ckpt", action="store_true", help='enable reader gradient checkpointing')
    ap.add_argument("--tta", type=int, default=1, help='inference views: original candidate order plus seeded shuffles')
    ap.add_argument("--save_lora", default="", help="save the trained reader's LoRA weights here")
    ap.add_argument("--load_lora", default="", help='load reader LoRA weights instead of training')
    ap.add_argument("--dump_probs", default="", help='write dev/test candidate probabilities using --train_k candidates')
    ap.add_argument("--evidence", type=int, default=0, help="show each candidate's paths of up to H hops from the topic entities")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = torch.load(a.prep, weights_only=False)
    tok = AutoTokenizer.from_pretrained(a.model)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if cuda else torch.float32,
                                                 device_map="cuda" if cuda else None).eval()
    res = {}
    data = {}
    sources = [("validation", [a.dump]), ("test", [a.dump])]
    if a.train_dumps:
        sources.append(("train", a.train_dumps.split(",")))
    for split, prefixes in sources:
        meta = {it["id"]: it for it in P["splits"][split]}
        rows = [json.loads(l) for pre in prefixes for l in open(f"{pre}.{split}.jsonl")]
        data[split] = [dict(id=r["id"], q=r["q"], kg=r["pred"],
                            topic=P["entities"][meta[r["id"]]["topic"][0]] if meta[r["id"]]["topic"] else "",
                            cands=[n for n, _ in r.get("top", []) if not UNNAMED.match(n)],
                            gold={norm(P["entities"][x]) for x in meta[r["id"]]["answers"]}) for r in rows]
    hit = lambda it, ans: ans is not None and norm(ans) in it["gold"]
    if a.train_dumps and a.train_max and len(data["train"]) > a.train_max:
        import random
        src = {it["id"]: it.get("source", "webqsp") for it in P["splits"]["train"]}
        own = [it for it in data["train"] if src.get(it["id"]) == "webqsp"]
        rest = [it for it in data["train"] if src.get(it["id"]) != "webqsp"]
        random.Random(0).shuffle(rest)
        data["train"] = own + rest[:max(0, a.train_max - len(own))]
        res["reader training questions"] = dict(webqsp=len(own), other=len(data["train"]) - len(own))
    if a.evidence:
        kmax = max(a.train_k, *map(int, a.ks.split(",")))
        for split, items in data.items():
            meta = {it["id"]: it for it in P["splits"][split]}
            for it in items:
                m = meta[it["id"]]
                it["topic"] = "; ".join(P["entities"][x] for x in m["topic"])
                it["ev"] = evidence(P, m, it["cands"][:kmax], a.evidence)
    if a.train_dumps:
        if a.grad_ckpt:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            model.enable_input_require_grads()
        mods = train_reader(model, tok, data["train"], a.train_k, a.labels, a.epochs, a.lr, a.lora_r, bs=a.bs,
                            accum=a.accum)
        res["reader trained on"] = len(data["train"])
        if a.save_lora:
            from kgmem.lora import lora_state
            torch.save(lora_state(mods), ensure_parent(a.save_lora))
    elif a.load_lora:
        from kgmem.lora import add_lora, load_lora
        from kgmem.memlm import find_decoder_layers
        layers = find_decoder_layers(model)
        mods = add_lora(layers, len(layers) - 1, a.lora_r, 2.0 * a.lora_r)
        load_lora(mods, torch.load(a.load_lora, map_location="cpu", weights_only=True))
        model.eval()
    dev = data["validation"]
    res["dev kgmem top-1"] = round(100 * sum(hit(it, it["kg"]) for it in dev) / len(dev), 2)
    for k in [int(x) for x in a.ks.split(",")]:
        for t in sorted({1, a.tta}):
            picks = choose(model, tok, dev, k, labels=a.labels, tta=t)
            res[f"dev reader k={k}" + (f" tta={t}" if t > 1 else "")] = round(
                100 * sum(hit(it, p) for it, p in zip(dev, picks)) / len(dev), 2)
        print(json.dumps(res), flush=True)
    cfgs = [(int(x), t) for x in a.ks.split(",") for t in sorted({1, a.tta})]
    key = lambda c: f"dev reader k={c[0]}" + (f" tta={c[1]}" if c[1] > 1 else "")
    best = max(cfgs, key=lambda c: res[key(c)])
    use_reader = res[key(best)] > res["dev kgmem top-1"]
    res["chosen on dev"] = key(best).replace("dev ", "") if use_reader else "kgmem top-1"
    test = data["test"]
    picks = choose(model, tok, test, best[0], labels=a.labels, tta=best[1]) if use_reader else [it["kg"] for it in test]
    overlap = set(json.load(open(a.overlap))) if a.overlap else set()
    for sub, keep in (("all", lambda it: True), ("CWQ-train overlap", lambda it: it["id"] in overlap),
                      ("no overlap", lambda it: it["id"] not in overlap)):
        sel = [(it, p) for it, p in zip(test, picks) if keep(it)]
        if sel:
            res[f"test [{sub}]"] = dict(n=len(sel), strict=round(100 * sum(hit(it, p) for it, p in sel) / len(sel), 2))
    if a.dump_probs:
        with open(ensure_parent(a.dump_probs), "w") as f:
            for split in ("validation", "test"):
                sc = []
                choose(model, tok, data[split], a.train_k, labels=a.labels, scores=sc)
                for it, p in zip(data[split], sc):
                    f.write(json.dumps(dict(split=split, id=it["id"], cands=it["cands"][:a.train_k], p=p,
                                            gold=sorted(it["gold"]))) + "\n")
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(ensure_parent(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
