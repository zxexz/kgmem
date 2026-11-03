"""Synthetic function-library retrieval with one- and two-call questions.

  python -m experiments.retrieval.funclib --n_funcs 50,200,1000,5000 --init checkpoints/mquake.pt --steps 3000 --out runs/retrieval/funclib/results.json

Each relation has a typed signature, name and description. Functions map numbers,
words or codes to random values; questions describe compositions using synonyms.
All functions are loaded, including distractors. rel_exists filters input types.
A name hash holds out a fifth of functions from training questions; their word parts
still occur in trained functions. --hold_phrasing reserves synonyms and templates."""
import argparse
import json
import random
import time
import zlib

import torch

from kgmem import MemConfig
from kgmem.paths import ensure_parent, checkpoint_path
from kgmem.graph import ExactScorer, GraphTensors, KnowledgeGraph
from kgmem.memlm import Batch, MemoryLM
from kgmem.tasks import SYL, _find_last, kg_batch
from experiments.qa.metaqa import build

# verb -> (output type, synonyms); first synonym is canonical; "same" preserves input type.
VERBS = {
    "count": ("number", ["count", "tally", "number"]), "sum": ("number", ["sum", "add up", "total"]),
    "max": ("same", ["take the largest", "find the biggest", "pick the maximum"]),
    "min": ("same", ["take the smallest", "find the least", "pick the minimum"]),
    "reverse": ("same", ["reverse", "flip", "invert the order of"]),
    "sort": ("same", ["sort", "order", "arrange"]), "rotate": ("same", ["rotate", "shift", "cycle"]),
    "hash": ("code", ["hash", "fingerprint", "digest"]), "encode": ("code", ["encode", "encipher", "scramble"]),
    "decode": ("word", ["decode", "decipher", "unscramble"]), "spell": ("word", ["spell", "write out", "name"]),
    "measure": ("number", ["measure", "gauge", "size up"]), "rank": ("number", ["rank", "place", "position"]),
    "merge": ("same", ["merge", "combine", "fuse"]), "split": ("same", ["split", "break up", "divide"]),
    "filter": ("same", ["filter", "keep only", "select"]), "drop": ("same", ["drop", "remove", "delete"]),
    "double": ("same", ["double", "duplicate", "repeat twice"]), "halve": ("same", ["halve", "cut in half", "split evenly"]),
    "invert": ("same", ["invert", "negate", "flip the sign of"]), "map": ("code", ["map", "translate", "convert"]),
    "label": ("word", ["label", "tag", "annotate"]), "score": ("number", ["score", "grade", "rate"]),
    "classify": ("word", ["classify", "categorise", "sort into a class"]),
}
# noun: (input type, synonyms)
NOUNS = {
    "vowels": ("word", ["vowels", "vowel letters", "vowel sounds"]),
    "consonants": ("word", ["consonants", "consonant letters", "non-vowels"]),
    "letters": ("word", ["letters", "characters", "glyphs"]), "syllables": ("word", ["syllables", "beats", "sound units"]),
    "prefix": ("word", ["prefix", "leading part", "start"]), "suffix": ("word", ["suffix", "ending", "tail end"]),
    "anagram": ("word", ["anagram", "letter shuffle", "rearrangement"]), "rhyme": ("word", ["rhyme", "rhyming word", "sound-alike"]),
    "initials": ("word", ["initials", "first letters", "leading letters"]), "stem": ("word", ["stem", "root", "base form"]),
    "plural": ("word", ["plural", "multiple form", "plural form"]), "accent": ("word", ["accent", "stress", "emphasis"]),
    "digits": ("number", ["digits", "numerals", "figures"]), "primes": ("number", ["primes", "prime factors", "prime divisors"]),
    "divisors": ("number", ["divisors", "factors", "even divisions"]), "square": ("number", ["square", "second power", "self-product"]),
    "root": ("number", ["square root", "radical", "root value"]), "parity": ("number", ["parity", "oddness", "evenness"]),
    "remainder": ("number", ["remainder", "modulus", "leftover"]), "successor": ("number", ["successor", "next number", "following value"]),
    "magnitude": ("number", ["magnitude", "absolute size", "scale"]), "binary": ("number", ["binary form", "base-two form", "bits"]),
    "checksum": ("code", ["checksum", "check digit", "control sum"]), "header": ("code", ["header", "leading block", "preamble"]),
    "payload": ("code", ["payload", "body", "content block"]), "opcode": ("code", ["opcode", "instruction code", "operation code"]),
    "register": ("code", ["register", "slot", "cell"]), "address": ("code", ["address", "location", "pointer"]),
    "parity bits": ("code", ["parity bits", "check bits", "error bits"]), "key": ("code", ["key", "secret", "cipher key"]),
    "token": ("code", ["token", "symbol", "unit"]), "flag": ("code", ["flag", "marker", "indicator"]),
    "checks": ("code", ["checks", "tests", "assertions"]), "segments": ("code", ["segments", "sections", "chunks"]),
    "moves": ("code", ["moves", "legal moves", "available moves"]), "chords": ("code", ["chords", "harmonies", "triads"]),
    "intervals": ("number", ["intervals", "gaps", "steps between"]), "fractions": ("number", ["fractions", "ratios", "proportions"]),
    "percentages": ("number", ["percentages", "percent values", "shares"]), "dates": ("number", ["dates", "days", "calendar values"]),
}
QUALS = {"": [""], "left": ["to the left", "leftward", "on the left side"], "right": ["to the right", "rightward",
         "on the right side"], "mod 12": ["modulo twelve", "mod 12", "wrapped at twelve"], "twice": ["two times", "twice",
         "a second time"], "weighted": ["with weights", "weighted", "by weight"], "unique": ["without repeats",
         "deduplicated", "unique"], "cumulative": ["cumulatively", "as a running total", "accumulated"]}
TYPES = ("number", "word", "code")
ONE = ["What do you get if you {p} of {h}?", "If you {p} of {h}, what is the result?", "{P} of {h}: what comes out?"]
TWO = ["What do you get if you {p1} of {h} and then {p2} of that?",
       "If you {p1} of {h}, then {p2} of the result, what do you get?"]


def phrase(f, rng, canonical=False, mode=None):
    """Choose one synonym per name part; return text and whether it uses held-out wording.

    mode="train" excludes each part's last synonym. The description uses the first."""
    novel, words = False, []
    for xs in (VERBS[f["verb"]][1], ["the"], NOUNS[f["noun"]][1], QUALS[f["qual"]]):
        if canonical:
            w = xs[0]
        else:
            w = rng.choice(xs[:-1] if mode == "train" and len(xs) > 1 else xs)
            novel |= len(xs) > 1 and w == xs[-1]
        words.append(w)
    text = " ".join(w for w in words if w)
    return text if canonical else (text, novel)


def library(n, seed=0):
    rng = random.Random(seed)
    combos = [(v, nn, q) for v in VERBS for nn in NOUNS for q in QUALS]
    rng.shuffle(combos)
    lib = []
    for v, nn, q in combos[:n]:
        it = NOUNS[nn][0]
        ot = it if VERBS[v][0] == "same" else VERBS[v][0]
        name = "_".join(x.replace(" ", "_") for x in (v, nn, q) if x)
        f = dict(name=name, verb=v, noun=nn, qual=q, in_type=it, out_type=ot,
                 held_out=zlib.crc32(name.encode()) % 5 == 0)
        f["key"] = f"{name}({it}) -> {ot}: {phrase(f, rng, canonical=True)}"
        lib.append(f)
    return lib


class FuncWorld:
    """Random maps over 30 values per type.

    side="train" uses trained functions; side="held_out" requires a held-out call."""

    def __init__(self, rng, lib, side="train", n_per=30, ask=None, phrasing=None):
        used, vals = set(), {}
        vals["number"] = [str(x) for x in rng.sample(range(10, 1000), n_per)]
        vals["word"] = [self._name(rng, used, 2, 3).capitalize() for _ in range(n_per)]
        vals["code"] = ["#" + self._name(rng, used, 2, 2) for _ in range(n_per)]
        ents = vals["number"] + vals["word"] + vals["code"]
        kg = KnowledgeGraph(ents, [f["key"] for f in lib])
        ix = kg.ent_index
        for r, f in enumerate(lib):
            for h in vals[f["in_type"]]:                 # Exclude the head from its own answers.
                kg.triples.append((ix[h], r, ix[rng.choice([v for v in vals[f["out_type"]] if v != h])]))
        kg._version += 1
        self.kg, self.lib, self.vals, self.side = kg, lib, vals, side
        self.ask = ask or len(lib)                 # Only the first ask functions occur in questions.
        self.phrasing = phrasing                   # None: any; "train": no held-out wording; "fresh": some held out
        self.facts = {(h, r): t for h, r, t in kg.triples}
        self.by_type = {t: [i for i, f in enumerate(lib[:self.ask]) if f["in_type"] == t] for t in TYPES}

    @staticmethod
    def _name(rng, used, lo, hi):
        while True:
            n = "".join(rng.choice(SYL) for _ in range(rng.randint(lo, hi)))
            if n not in used:
                used.add(n)
                return n

    def sample_question(self, rng, p_two_hop=0.4):
        lib, held = self.lib, self.side == "held_out"
        while True:
            hops = 2 if rng.random() < p_two_hop else 1
            r1 = rng.randrange(self.ask)
            chain = [r1]
            if hops == 2:
                nxt = self.by_type[lib[r1]["out_type"]]
                chain.append(rng.choice(nxt))
            if any(lib[r]["held_out"] for r in chain) != held:
                continue
            h = rng.choice(self.vals[lib[r1]["in_type"]])
            e = self.kg.ent_index[h]
            for r in chain:
                e = self.facts[(e, r)]
            mode = "train" if self.phrasing == "train" else None
            temps = ONE if hops == 1 else TWO
            t = rng.choice(temps[:-1] if mode == "train" else temps)
            parts = [phrase(lib[r], rng, mode=mode) for r in chain]
            if self.phrasing == "fresh" and not (t == temps[-1] or any(nv for _, nv in parts)):
                continue                           # Held-out template or synonym.
            if hops == 1:
                q = t.format(p=parts[0][0], P=parts[0][0].capitalize(), h=h)
            else:
                q = t.format(p1=parts[0][0], p2=parts[1][0], h=h)
            return q, self.kg.entities[e], h, [self.kg.relations[r] for r in chain]


def encode(mlm, names, pool):
    """Encode names in equal-length batches when the backbone disallows padding."""
    if not getattr(mlm, "unpadded", False):
        return mlm.encode_names(names, pool=pool)
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(len(mlm.tokenize_name(n)), []).append(i)
    out = [None] * len(names)
    for idx in groups.values():
        for j in range(0, len(idx), 64):
            part = idx[j:j + 64]
            for i, k in zip(part, mlm.encode_names([names[i] for i in part], bs=len(part), pool=pool)):
                out[i] = k
    return torch.stack(out)


class Keys:
    """Cache relation keys across worlds; each world encodes its 90 entity names."""

    def __init__(self, mlm, lib, first=True):
        self.mlm = mlm
        names = [f["key"] for f in lib]
        self.rel = encode(mlm, names, mlm.cfg.rel_key_pool)
        # RelationDecoder does not need first-token keys (0.5 GB for 44k tools on Gemma-4-31B).
        self.rel_first = encode(mlm, names, "first") if first else None
        self.rel_alias = mlm._alias(names)

    def load(self, kg):
        mlm, dev = self.mlm, self.mlm.device
        toks = [mlm.tokenize_name(n)[:12] for n in kg.entities]
        L = max(map(len, toks))
        name_tok = torch.full((len(toks), L), -1, dtype=torch.long)
        alias = {}
        for i, t in enumerate(toks):
            name_tok[i, :len(t)] = torch.tensor(t)
            alias[tuple(t)] = i if tuple(t) not in alias else -1
        mlm.ctx.graph = GraphTensors(
            ent_keys=encode(mlm, kg.entities, mlm.cfg.key_pool), rel_keys=self.rel,
            name_tok=name_tok.to(dev), name_len=torch.tensor(list(map(len, toks)), device=dev),
            scorer=ExactScorer(kg, mlm.cfg.max_tails, dev), kg=kg, rel_keys_first=self.rel_first,
            alias=alias, rel_alias=self.rel_alias)


def question_rows(mlm, world, rng, B=8):
    """Build unpadded question-only batches.

    Under the external loader's policy B, global attention would expose an appended answer at
    the question's final position. Each question therefore occupies its own row."""
    tok, kg, dev, H = mlm.tok, world.kg, mlm.device, mlm.cfg.hops
    out = []
    for _ in range(B):
        q, ans, head, rels = world.sample_question(rng, 0.4)
        p = tok(f"Q: {q}\nA:", add_special_tokens=False).input_ids
        ids = list(mlm.bos) + p
        mention = _find_last(p, mlm.tokenize_name(head))
        rel = torch.full((1, H), -100, dtype=torch.long)
        rel[0, :len(rels)] = torch.tensor([kg.rel_index[r] for r in rels])
        none = torch.full((1, H), -1, dtype=torch.long, device=dev)
        t = torch.tensor([ids], device=dev)
        out.append((Batch(t, torch.full_like(t, -100), torch.ones_like(t), [],
                          aux_idx=torch.tensor([len(ids) - 1], device=dev),
                          aux_ent=torch.tensor([kg.ent_index[head]], device=dev), aux_rel=rel.to(dev),
                          aux_hops=torch.tensor([len(rels) - 1], device=dev),
                          aux_mention=torch.tensor([len(mlm.bos) + mention if mention >= 0 else -1], device=dev),
                          aux_rel_mention=none, aux_rel_span_start=none), kg.ent_index[ans]))
    return out


def question_step(mlm, rows):
    """Average auxiliary loss, logs and Hits@1 over single-question rows."""
    total, logs, hits = 0.0, {}, 0
    for b, ans in rows:
        mlm.forward(b.ids, b.attn, aux_idx=b.aux_idx, aux_mention=b.aux_mention)
        loss, lg = mlm.aux_loss(b, 1.0)
        total = total + loss / len(rows)
        for k, v in lg.items():
            logs[k] = logs.get(k, 0.0) + v / len(rows)
        cid, cw = mlm.ctx.cand_ids[0, -1], mlm.ctx.cand_w[0, -1].float().detach()
        dense = torch.zeros(len(mlm.ctx.graph.kg.entities), device=cw.device).scatter_add_(0, cid, cw)
        hits += int(dense.argmax()) == ans
    return total, logs, hits


@torch.no_grad()
def evaluate(mlm, keys, lib, side, n, seed, ask=None, phrasing=None):
    """Measure graph-head Hits@1 and per-hop relation accuracy."""
    rng = random.Random(seed)
    was = mlm.block.training
    mlm.block.eval()
    hits = tot = 0
    accs = {}
    if getattr(mlm, "question_only", False):
        for w in range(n // 8):
            world = FuncWorld(rng, lib, side, ask=ask, phrasing=phrasing if side == "train" else None)
            keys.load(world.kg)
            _, logs, h = question_step(mlm, question_rows(mlm, world, rng, 8))
            hits, tot = hits + h, tot + 8
            for k, v in logs.items():
                if k.startswith("rel") and k.endswith("_acc"):
                    accs.setdefault(k, []).append(v)
        mlm.block.train(was)
        return dict(hits1=round(hits / max(1, tot), 4), n=tot, **{k: round(sum(v) / len(v), 4) for k, v in accs.items()})
    for w in range(n // 24):
        world = FuncWorld(rng, lib, side, ask=ask, phrasing=phrasing if side == "train" else None)
        keys.load(world.kg)
        b = kg_batch(mlm, world, rng, B=8, n_qa=3)
        _, logs = mlm.loss(b, 1.0)
        for k, v in logs.items():
            if k.startswith("rel") and k.endswith("_acc"):
                accs.setdefault(k, []).append(v)
        rows = b.aux_idx // b.ids.shape[1]
        pos = b.aux_idx % b.ids.shape[1]
        cid, cw = mlm.ctx.cand_ids[rows, pos], mlm.ctx.cand_w[rows, pos].float()
        dense = torch.zeros(len(rows), len(world.kg.entities), device=cw.device).scatter_add_(1, cid, cw)
        pred = dense.argmax(-1)
        ans = [b.ids[s, a:z].tolist() for s, a, z, _ in b.spans]
        names = [tuple(mlm.tokenize_name(world.kg.entities[p])) for p in pred.tolist()]
        hits += sum(tuple(x) == nm for x, nm in zip(ans, names))
        tot += len(ans)
    mlm.block.train(was)
    return dict(hits1=round(hits / max(1, tot), 4), n=tot, **{k: round(sum(v) / len(v), 4) for k, v in accs.items()})


@torch.no_grad()
def recall_eval(mlm, keys, lib, side, n, seed, ask=None, ks=(1, 3, 5, 10, 20), phrasing=None):
    """Measure first-function and whole-chain Recall@k.

    Rank by P(length) times hop probabilities. Each hop conditions on the decoder's
    previous choice; the decoder does not use an internal beam."""
    rng = random.Random(seed)
    was = mlm.block.training
    mlm.block.eval()
    hit = {f"{what}@{k}": 0 for what in ("function", "chain") for k in ks}
    tot = 0
    for _ in range(n // 8):
        world = FuncWorld(rng, lib, side, ask=ask, phrasing=phrasing if side == "train" else None)
        keys.load(world.kg)
        for b, _ in question_rows(mlm, world, rng, 8):
            mlm.forward(b.ids, b.attn, aux_idx=b.aux_idx, aux_mention=b.aux_mention)
            aux = mlm.ctx.kg_aux
            gold = [r for r in b.aux_rel[0].tolist() if r >= 0]
            p = [r[0].float().softmax(-1) for r in aux["rel"]]
            halt = aux["halt"][0].float().softmax(-1)
            first = p[0].topk(max(ks)).indices.tolist()
            K = max(ks)
            chains = [((r,), float(halt[0] * p[0][r])) for r in first]
            if len(p) > 1:
                p1 = p[1].topk(K)
                chains += [((r0, int(r1)), float(halt[1] * p[0][r0] * v1)) for r0 in first[:3]
                           for v1, r1 in zip(p1.values, p1.indices)]
            ranked = [c for c, _ in sorted(chains, key=lambda x: -x[1])]
            for k in ks:
                hit[f"function@{k}"] += gold[0] in first[:k]
                hit[f"chain@{k}"] += tuple(gold) in ranked[:k]
            tot += 1
    mlm.block.train(was)
    return {k: round(v / max(1, tot), 4) for k, v in hit.items()} | {"n": tot}


def bm25_recall(lib, side, n, seed, ask=None, ks=(1, 3, 5, 10, 20), k1=1.5, b=0.75, phrasing=None):
    """BM25 Recall@k on the same seeded questions as recall_eval.

    Index name, signature and description. Rank single functions and compatible
    pairs from the top 20 by summed score; both orders may compete."""
    import math
    import re
    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower().replace("_", " "))
    docs = [tok(f["key"]) for f in lib]
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    N, avg = len(docs), sum(map(len, docs)) / len(docs)
    idf = {w: math.log(1 + (N - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    tfs = [{w: d.count(w) for w in set(d)} for d in docs]

    def scores(q):
        qs = set(tok(q))
        return [sum(idf[w] * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg)) for w in qs if w in tf)
                for tf, d in zip(tfs, docs)]
    rng = random.Random(seed)
    hit = {f"{what}@{k}": 0 for what in ("function", "chain") for k in ks}
    tot = 0
    for _ in range(n // 8):
        world = FuncWorld(rng, lib, side, ask=ask, phrasing=phrasing if side == "train" else None)
        for _ in range(8):
            q, ans, head, rels = world.sample_question(rng, 0.4)
            gold = [world.kg.rel_index[r] for r in rels]
            sc = scores(q)
            order = sorted(range(N), key=lambda i: -sc[i])
            top = order[:20]
            chains = [((i,), sc[i]) for i in top] + [((i, j), sc[i] + sc[j]) for i in top for j in top
                                                     if i != j and lib[i]["out_type"] == lib[j]["in_type"]]
            ranked = [c for c, _ in sorted(chains, key=lambda x: -x[1])]
            for k in ks:
                hit[f"function@{k}"] += gold[0] in order[:k]
                hit[f"chain@{k}"] += tuple(gold) in ranked[:k]
            tot += 1
    return {k: round(v / max(1, tot), 4) for k, v in hit.items()} | {"n": tot}


def build_model(args, cfg):
    """Load through metaqa.build or a Python attention loader.

    The external loader exports register() -> a Transformers attention name and
    require_policy(text_config, policy). This path loads CUDA bf16 weights.
    Callers use unpadded rows and pass --policy (B by default) to the loader;
    the loader defines that policy's attention behavior.
    """
    if not args.backbone_loader:
        return build(args, cfg)[0]
    import importlib.util
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = importlib.util.spec_from_file_location("kgmem_backbone_loader", args.backbone_loader)
    fm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fm)
    name = fm.register()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation=name).eval()
    fm.require_policy(model.config.get_text_config(), args.policy)
    return MemoryLM(model, AutoTokenizer.from_pretrained(args.model), cfg, layer_idx=args.layer)


def run(args, n):
    torch.manual_seed(args.seed)
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=True)
        args.layer = init["layer_idx"]
        cfg = MemConfig(**{**init["cfg"], "rel_named": 0.0, **json.loads(args.cfg)})
    else:                                              # Fresh two-hop block using the MQuAKE configuration.
        from experiments.qa import mquake
        cfg = mquake.mquake_cfg(None, hops=2, rel_named=0.0, **json.loads(args.cfg))
    if args.question_only:
        cfg = MemConfig(**{**cfg.__dict__, "kg_gate_loss": 0.0})      # Question-only rows have no answer-token gate supervision.
    mlm = build_model(args, cfg)
    mlm.question_only, mlm.unpadded = args.question_only, bool(args.backbone_loader)
    if args.init:
        print(f"[{n}] warm start:", mlm.load(args.init, strict=False), flush=True)
    lib = library(n, args.lib_seed)
    t0 = time.time()
    if args.standardize:                               # Calibrate before encoding keys in the same space.
        crng = random.Random(1234)
        w = FuncWorld(crng, lib, "train")
        texts = [f"Q: {w.sample_question(crng)[0]}\nA:" for _ in range(200)] + [f["key"] for f in lib[:200]]
        mlm.calibrate(texts)
        print(f"[{n}] calibrated on {len(texts)} texts: scale median {float(mlm.block.in_scale.median()):.3g}, "
              f"max {float(mlm.block.in_scale.max()):.3g}", flush=True)
    keys = Keys(mlm, lib)
    print(f"[{n}] {len(lib)} functions ({sum(f['held_out'] for f in lib)} held out), keys in {time.time() - t0:.0f}s",
          flush=True)
    fresh = "fresh" if args.hold_phrasing else None
    out = {"before": {s: evaluate(mlm, keys, lib, s, args.eval_n, 7, phrasing=fresh) for s in ("train", "held_out")}}
    print(f"[{n}] before training: {json.dumps(out['before'])}", flush=True)
    params = mlm.trainable_parameters()
    opt = torch.optim.AdamW(params, args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    rng, t0 = random.Random(args.seed), time.time()
    for step in range(1, args.steps + 1):
        if step % args.swap_every == 1 or args.swap_every == 1:
            world = FuncWorld(rng, lib, "train", phrasing="train" if args.hold_phrasing else None)
            keys.load(world.kg)
        if args.question_only:
            loss, logs, _ = question_step(mlm, question_rows(mlm, world, rng, args.batch))
        else:
            b = kg_batch(mlm, world, rng, B=args.batch, n_qa=3)
            loss, logs = mlm.loss(b, 1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            print(f"[{n}] step {step} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(logs.items()))
                  + f" {(time.time() - t0) / step:.2f}s/step", flush=True)
    # Evaluate fresh worlds with trained functions and held-out-function chains.
    out["after"] = {s: evaluate(mlm, keys, lib, s, args.eval_n, 11, phrasing=fresh) for s in ("train", "held_out")}
    print(f"[{n}] after {args.steps} steps: {json.dumps(out['after'])}", flush=True)
    if args.save:
        mlm.save(ensure_parent(args.save.replace(".pt", f"_{n}.pt")), {"args": vars(args), "n_funcs": n, "eval": out["after"]})
    # Library prefixes share functions; extra entries are unseen distractors.
    out["distractors"] = {}
    for m in [int(x) for x in args.eval_libs.split(",") if x and int(x) > n]:
        big = library(m, args.lib_seed)
        out["distractors"][m] = {s: evaluate(mlm, Keys(mlm, big), big, s, args.eval_n, 11, ask=n, phrasing=fresh)
                                 for s in ("train", "held_out")}
        print(f"[{n}] with {m} functions loaded: {json.dumps(out['distractors'][m])}", flush=True)
    return out


def recall_main(args):
    """Evaluate adapter-library and --eval_libs sizes using --recall_eval."""
    sd = torch.load(args.recall_eval, map_location="cpu", weights_only=True)
    args.layer = sd["layer_idx"]
    mlm = build_model(args, MemConfig(**sd["cfg"]))
    mlm.load(args.recall_eval)                     # Restore calibration before encoding keys.
    mlm.question_only, mlm.unpadded = True, bool(args.backbone_loader)
    n = sd["meta"]["n_funcs"]
    fresh = "fresh" if sd["meta"]["args"].get("hold_phrasing") else None
    out = {}
    for m in [n] + [int(x) for x in args.eval_libs.split(",") if x and int(x) > n]:
        lib = library(m, args.lib_seed)
        keys = Keys(mlm, lib)
        out[m] = {s: recall_eval(mlm, keys, lib, s, args.eval_n, 11, ask=n, phrasing=fresh)
                  for s in ("train", "held_out")}
        print(f"[{n}] Recall@k with {m} functions loaded: {json.dumps(out[m])}", flush=True)
    json.dump(out, open(ensure_parent(args.out), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_funcs", default="50,200,1000,5000")
    ap.add_argument("--init", default=checkpoint_path("mquake.pt"), help="warm start ('' for a fresh block)")
    ap.add_argument("--layer", type=int, default=None, help="attach layer for a fresh block")
    ap.add_argument("--question_only", action="store_true", help="one unpadded question per row, no answer tokens")
    ap.add_argument("--backbone-loader", dest="backbone_loader", default="",
                    help='optional Python loader exporting register() -> attention name and require_policy(text_config, policy); uses CUDA bf16')
    ap.add_argument("--policy", default="B", help="the attention policy the model's config must carry")
    ap.add_argument("--standardize", action="store_true", help="calibrate the block's per-dimension input "
                    "standardisation (MemoryLM.calibrate) on sample questions and names before training")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--swap_every", type=int, default=1)
    ap.add_argument("--eval_n", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lib_seed", type=int, default=0)
    ap.add_argument("--eval_libs", default="", help="after training, evaluate with these bigger libraries loaded")
    ap.add_argument("--save", default="", help="save each trained adapter here (_<n> appended)")
    ap.add_argument("--out", default="runs/retrieval/funclib/results.json")
    ap.add_argument("--recall_eval", default="", help="no training: Recall@k of a saved question-only adapter")
    ap.add_argument("--bm25", action="store_true", help="no model: BM25's Recall@k on the same questions")
    ap.add_argument("--hold_phrasing", action="store_true", help="train without one synonym per word and one template "
                    "per hop count; trained functions are then evaluated only on wordings that use one of those")
    a = ap.parse_args()
    if a.bm25:
        out = {}
        for n in [int(x) for x in a.n_funcs.split(",")]:
            for m in [n] + [int(x) for x in a.eval_libs.split(",") if x and int(x) > n]:
                big = library(m, a.lib_seed)
                out[f"{n} asked, {m} loaded"] = {s: bm25_recall(big, s, a.eval_n, 11, ask=n,
                                                                phrasing="fresh" if a.hold_phrasing else None)
                                                 for s in ("train", "held_out")}
                print(f"BM25 {n} asked, {m} loaded: {json.dumps(out[f'{n} asked, {m} loaded'])}", flush=True)
        json.dump(out, open(ensure_parent(a.out), "w"), indent=1)
        return
    if a.recall_eval:
        return recall_main(a)
    res = {}
    for n in [int(x) for x in a.n_funcs.split(",")]:
        res[n] = run(a, n)
        json.dump(res, open(ensure_parent(a.out), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
