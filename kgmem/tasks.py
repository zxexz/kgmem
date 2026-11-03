"""Synthetic graph, scratchpad, and plain-text training batches.

Worlds resample facts and names. The deterministic name split keeps evaluation
entity, relation, and variable names out of training.
"""
from __future__ import annotations

import random
import zlib

import torch

from .graph import KnowledgeGraph
from .memlm import Batch

SYL = ["ka", "zor", "vel", "mi", "dra", "tun", "bel", "sha", "qui", "ron", "lo", "fen", "gar", "ix",
       "pel", "sor", "ta", "ven", "yul", "mar", "ok", "dee", "nar", "bri", "sto", "ul", "zen", "pha",
       "cor", "lin", "mu", "rak", "eth", "bo", "vir", "sal", "tor", "gwen", "ad", "ny"]

RELATIONS = ["capital", "currency", "located in", "lives in", "leader", "friend"]

ONE_HOP = {
    "capital": ["What is the capital of {h}?", "Name the capital city of {h}.", "Which city is the capital of {h}?"],
    "currency": ["What currency does {h} use?", "Which currency is used in {h}?", "What money do people use in {h}?"],
    "located in": ["In which country is {h}?", "{h} is a city in which country?", "Which country contains {h}?"],
    "lives in": ["Where does {h} live?", "In which city does {h} live?", "What city is {h} living in?"],
    "leader": ["Who leads {h}?", "Who is the leader of {h}?", "Who rules {h}?"],
    "friend": ["Who is {h}'s friend?", "Name the friend of {h}.", "Who is friends with {h}?"],
}
TWO_HOP = {
    ("lives in", "located in"): ["In which country does {h} live?", "What country is {h} living in?"],
    ("located in", "capital"): ["What is the capital of the country where {h} is?",
                                "Name the capital of the country containing {h}."],
    ("located in", "currency"): ["What currency is used in the city of {h}?",
                                 "Which money do they use in {h}'s country?"],
    ("leader", "lives in"): ["Where does the leader of {h} live?", "In which city does {h}'s leader live?"],
    ("friend", "lives in"): ["Where does {h}'s friend live?", "In which city does the friend of {h} live?"],
}

VALUES = ["red", "blue", "green", "black", "white", "gold", "silver", "pink", "cat", "dog", "fox", "owl",
          "bee", "cow", "pig", "rat", "apple", "pear", "plum", "lime", "rice", "salt", "tea", "milk"]

# Hold out one eighth of relation names. Arithmetic relations are excluded entirely
# so kgmem.strandedcpu evaluates on names absent from synthetic training.
REL_POOL = [
    "mentor", "rival", "neighbour", "sibling", "cousin", "partner", "ally", "enemy", "apprentice", "teacher", "student",
    "guardian", "heir", "deputy", "assistant", "bodyguard", "landlord", "tenant", "sponsor", "patron", "critic",
    "biographer", "editor", "publisher", "agent", "lawyer", "doctor", "coach", "captain", "pilot", "driver", "chef",
    "tailor", "barber", "banker", "accountant", "manager", "director", "producer", "twin", "godparent", "spouse",
    "roommate", "classmate", "colleague", "owner", "founder", "inventor", "designer", "architect", "builder", "creator",
    "keeper", "ruler", "governor", "mayor", "sheriff", "champion", "namesake", "sister city", "twin town", "port",
    "harbour", "river", "mountain", "lake", "forest", "island", "birthplace", "hometown", "workplace", "school",
    "alma mater", "favourite food", "favourite colour", "favourite book", "pet", "horse", "ship", "car", "house",
    "garden", "shop", "club", "team", "band", "record label", "studio", "newspaper", "emblem", "flag", "motto", "anthem",
    "mascot", "nickname", "codename", "weakness", "origin", "destination", "supplier", "customer", "competitor",
    "investor", "auditor", "insurer", "parent company", "subsidiary", "trading partner", "rival team", "home ground",
    "patron saint", "sworn enemy", "best friend", "first teacher", "music teacher", "personal trainer", "tutor",
    "translator", "secretary", "spokesperson", "ambassador", "envoy", "successor in office", "co-author", "muse",
    # Multi-word names require linking beyond the first token after "the".
    "older sister", "younger sister", "older brother", "younger brother", "first employer", "last employer",
    "head coach", "assistant coach", "senior partner", "junior partner", "previous owner", "next owner",
    "former partner", "current partner", "first novel", "last novel", "debut album", "latest album",
    "first language", "second language", "largest city", "smallest town", "closest ally", "oldest rival",
    "nearest airport", "train station", "lead singer", "head chef", "vice president", "chief engineer",
    "maternal grandmother", "paternal grandfather", "favourite song", "favourite film", "lucky number",
    "middle name", "stage name", "pen name", "home port", "main supplier", "biggest customer",
]
# Pair semantically similar names in half the worlds to train relation disambiguation.
CONTRAST_PAIRS = [
    ("older sister", "younger sister"), ("older brother", "younger brother"), ("first employer", "last employer"),
    ("head coach", "assistant coach"), ("senior partner", "junior partner"), ("previous owner", "next owner"),
    ("former partner", "current partner"), ("first novel", "last novel"), ("debut album", "latest album"),
    ("landlord", "tenant"), ("teacher", "student"), ("parent company", "subsidiary"), ("supplier", "customer"),
    ("mentor", "apprentice"), ("first language", "second language"), ("largest city", "smallest town"),
    ("closest ally", "oldest rival"), ("paternal grandfather", "maternal grandmother"),
]
REL_ONE_HOP = ["What is the {r} of {h}?", "Who is the {r} of {h}?", "Name the {r} of {h}.", "What is {h}'s {r}?"]
REL_TWO_HOP = ["What is the {r2} of the {r1} of {h}?", "Who is the {r2} of {h}'s {r1}?", "Name the {r2} of the {r1} of {h}."]


def is_heldout(name: str) -> bool:
    """Reserve one eighth of names for evaluation using crc32.

    Python's hash() is salted per process and would not give a stable split.
    """
    return zlib.crc32(name.lower().encode()) % 8 == 0


def make_name(rng: random.Random, used: set, lo=2, hi=3, cap=True, names: str | None = None) -> str:
    """names: "train" or "eval" restricts to that side of the held-out split; None draws from both."""
    while True:
        n = "".join(rng.choice(SYL) for _ in range(rng.randint(lo, hi)))
        n = n.capitalize() if cap else n
        if n not in used and (names is None or is_heldout(n) == (names == "eval")):
            used.add(n)
            return n


class World:
    def __init__(self, rng: random.Random, n_countries=8, cities_per=2, n_people=20, n_currencies=5,
                 names: str = "train"):
        used: set = set()
        countries = [make_name(rng, used, names=names) for _ in range(n_countries)]
        currencies = [make_name(rng, used, names=names) for _ in range(n_currencies)]
        cities = [make_name(rng, used, names=names) for _ in range(n_countries * cities_per)]
        people = [make_name(rng, used, names=names) for _ in range(n_people)]
        ents = countries + currencies + cities + people
        kg = KnowledgeGraph(ents, list(RELATIONS))
        self.kind = {**{c: "country" for c in countries}, **{c: "currency" for c in currencies},
                     **{c: "city" for c in cities}, **{p: "person" for p in people}}
        for i, c in enumerate(countries):
            cs = cities[i * cities_per:(i + 1) * cities_per]
            kg.add(c, "capital", cs[0])
            kg.add(c, "currency", rng.choice(currencies))
            kg.add(c, "leader", rng.choice(people))
            for city in cs:
                kg.add(city, "located in", c)
        for p in people:
            kg.add(p, "lives in", rng.choice(cities))
            kg.add(p, "friend", rng.choice([q for q in people if q != p]))
        self.kg = kg
        self.facts = {}
        for (h, r, t) in kg.triples:
            self.facts[(kg.entities[h], kg.relations[r])] = kg.entities[t]

    def follow(self, h, rels):
        for r in rels:
            h = self.facts.get((h, r))
            if h is None:
                return None
        return h

    def sample_question(self, rng: random.Random, p_two_hop=0.4):
        while True:
            if rng.random() < p_two_hop:
                rels = list(rng.choice(list(TWO_HOP)))
                template = rng.choice(TWO_HOP[tuple(rels)])
            else:
                rels = [rng.choice(RELATIONS)]
                template = rng.choice(ONE_HOP[rels[0]])
            heads = [e for (e, r) in self.facts if r == rels[0]]
            h = rng.choice(heads)
            ans = self.follow(h, rels)
            if ans is not None:
                return template.format(h=h), ans, h, rels


class RelWorld:
    """Sample relation names and random entity mappings for generic question templates.

    numeric=True uses integer entity names, as in kgmem.strandedcpu graphs.
    digits="balanced" targets equal groups from 0..9, 10..99, and 100..999,
    capped by each range's size. Uniform sampling from 0..999 supplies only 1%
    single-digit names.
    """

    def __init__(self, rng: random.Random, n_ent=40, n_rel=6, names: str = "train", relations: str = "train",
                 numeric: bool = False, edge_p: float = 0.8, digits: str = "uniform"):
        used: set = set()
        if numeric and digits == "balanced":
            k1 = min(10, n_ent // 3)
            k2 = min(90, (n_ent - k1) // 2)
            ents = [str(v) for v in rng.sample(range(10), k1) + rng.sample(range(10, 100), k2)
                    + rng.sample(range(100, 1000), n_ent - k1 - k2)]
        elif numeric:
            ents = [str(v) for v in rng.sample(range(1000), n_ent)]
        else:
            ents = [make_name(rng, used, names=names) for _ in range(n_ent)]
        side = relations == "eval"
        pool = [r for r in REL_POOL if is_heldout(r) == side]
        pairs = [p for p in CONTRAST_PAIRS if is_heldout(p[0]) == side and is_heldout(p[1]) == side]
        rels = list(rng.choice(pairs)) if pairs and rng.random() < 0.5 else []
        rels += rng.sample([r for r in pool if r not in rels], n_rel - len(rels))
        kg = KnowledgeGraph(ents, rels)
        for r in rels:
            for h in ents:
                if rng.random() < edge_p:
                    kg.add(h, r, rng.choice([e for e in ents if e != h]))
        self.kg = kg
        self.facts = {(kg.entities[h], kg.relations[r]): kg.entities[t] for (h, r, t) in kg.triples}

    def follow(self, h, rels):
        for r in rels:
            h = self.facts.get((h, r))
            if h is None:
                return None
        return h

    def sample_question(self, rng: random.Random, p_two_hop=0.4):
        rels_all = self.kg.relations
        while True:
            if rng.random() < p_two_hop:
                rels = rng.sample(rels_all, 2)
                template = rng.choice(REL_TWO_HOP).format(r1=rels[0], r2=rels[1], h="{h}")
            else:
                rels = [rng.choice(rels_all)]
                template = rng.choice(REL_ONE_HOP).format(r=rels[0], h="{h}")
            heads = [e for (e, r) in self.facts if r == rels[0]]
            h = rng.choice(heads)
            ans = self.follow(h, rels)
            if ans is not None:
                return template.format(h=h), ans, h, rels


_PROSE_THINGS = ["lamp", "chair", "garden", "window", "shirt", "car", "bicycle", "kitchen", "carpet", "bridge", "train",
                 "jacket", "roof", "fence", "mug", "boat", "pillow", "notebook", "street", "candle"]
_PROSE_QUALITIES = ["old", "new", "quiet", "bright", "dusty", "narrow", "warm", "cold", "heavy", "empty", "crowded",
                    "yellow", "purple", "orange", "grey", "striped", "wooden", "broken", "clean", "damp"]
_PROSE_DEVICES = ["thermostat", "radio", "heater", "timer", "fan", "volume", "speaker", "clock"]
_PROSE_TEMPLATES = [
    "The {t1} is {q1} and the {t2} is {q2}, so nobody minds.",
    "Set the {d} to {n} before you leave the house.",
    "Turn the {d} up to {n} and let it run for a while.",
    "My {t1} is {q1}, but my sister's {t1} is {q2}.",
    "The {t1} was {q1} this morning, and by evening the {t2} was {q2} too.",
    "If the {t1} is {q1}, put the {t2} by the door.",
    "Her favourite {t1} is the {q1} one; his is {q2}.",
    "We kept the {d} at {n} all winter.",
]


def assignment_prose(rng: random.Random, n: int) -> list[str]:
    """Generate ordinary assignment-like sentences for plain-text negatives.

    Templates and vocabulary are disjoint from eval_adapters.ASSIGN. These
    sentences should preserve the base LM distribution rather than trigger
    scratchpad recall.
    """
    out = []
    for _ in range(n):
        t1, t2 = rng.sample(_PROSE_THINGS, 2)
        q1, q2 = rng.sample(_PROSE_QUALITIES, 2)
        out.append(rng.choice(_PROSE_TEMPLATES).format(t1=t1, t2=t2, q1=q1, q2=q2, d=rng.choice(_PROSE_DEVICES),
                                                        n=rng.choice([3, 4, 12, 18, 25, 40, 55, 90])))
    return out


def plain_batch(mlm, rng: random.Random, texts: list, B=8) -> Batch:
    """Build unlabeled ordinary-text batches for KL to the base LM distribution."""
    picks = [list(mlm.bos) + rng.choice(texts) for _ in range(B)]
    ids, labels, attn = _pack([(p, [-100] * len(p), None) for p in picks], mlm.tok.pad_token_id or 0, mlm.device)
    return Batch(ids, labels, attn, [], plain=True)


def _find_last(seq, sub):
    """Index (in seq) of the last token of the last occurrence of sub, or -1."""
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i:i + len(sub)] == sub:
            return i + len(sub) - 1
    return -1


def _pack(examples, pad_id, device):
    T = max(len(e[0]) for e in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), T), -100, dtype=torch.long)
    attn = torch.zeros((len(examples), T), dtype=torch.long)
    for b, (i, l, _) in enumerate(examples):
        ids[b, :len(i)] = torch.tensor(i); labels[b, :len(l)] = torch.tensor(l); attn[b, :len(i)] = 1
    return ids.to(device), labels.to(device), attn.to(device)


def kg_batch(mlm, world: World, rng: random.Random, B=8, n_qa=3, p_two_hop=0.4) -> Batch:
    tok, kg = mlm.tok, world.kg
    nl = tok("\n", add_special_tokens=False).input_ids
    examples, spans, aux = [], [], []
    for b in range(B):
        ids, labels = list(mlm.bos), [-100] * len(mlm.bos)
        for _ in range(n_qa):
            q, ans, head, rels = world.sample_question(rng, p_two_hop)
            p = tok(f"Q: {q}\nA:", add_special_tokens=False).input_ids
            a = mlm.tokenize_name(ans)
            ids += p; labels += [-100] * len(p)
            mention = _find_last(p, mlm.tokenize_name(head))          # last token of the head mention
            base = len(ids) - len(p)
            names = [mlm.tokenize_name(r) for r in rels]
            named = [_find_last(p, n) for n in names]                 # last token naming each hop's relation
            starts = [base + n - len(nt) + 1 if n >= 0 else -1 for n, nt in zip(named, names)]
            aux.append((b, len(ids) - 1, kg.ent_index[head], [kg.rel_index[r] for r in rels],
                        base + mention if mention >= 0 else -1, [base + n if n >= 0 else -1 for n in named], starts))
            spans.append((b, len(ids), len(ids) + len(a), f"{len(rels)}-hop"))
            ids += a; labels += a
            ids += nl; labels += nl
        examples.append((ids, labels, None))
    ids, labels, attn = _pack(examples, tok.pad_token_id or 0, mlm.device)
    T, H = ids.shape[1], mlm.cfg.hops
    rel = torch.full((len(aux), H), -100, dtype=torch.long)
    named = torch.full((len(aux), H), -1, dtype=torch.long)
    starts = torch.full((len(aux), H), -1, dtype=torch.long)
    for i, (_, _, _, rs, _, ns, st) in enumerate(aux):
        rel[i, :len(rs)] = torch.tensor(rs)
        named[i, :len(ns)] = torch.tensor(ns)
        starts[i, :len(st)] = torch.tensor(st)
    dev = mlm.device
    return Batch(ids, labels, attn, spans,
                 aux_idx=torch.tensor([b * T + t for (b, t, *_) in aux], device=dev),
                 aux_ent=torch.tensor([e for (_, _, e, *_) in aux], device=dev),
                 aux_rel=rel.to(dev),
                 aux_hops=torch.tensor([len(rs) - 1 for (_, _, _, rs, *_) in aux], device=dev),
                 aux_mention=torch.tensor([m for (_, _, _, _, m, *_) in aux], device=dev),
                 aux_rel_mention=named.to(dev), aux_rel_span_start=starts.to(dev))


def scratchpad_batch(mlm, rng: random.Random, B=8, n_vars=4, n_assign=6, n_q=2, split_session=True,
                     names: str = "train") -> Batch:
    """Build variable assignments, overwrites, and recall questions.

    With split_session=True, writes and questions use separate batches. Only
    scratchpad state connects them; question tokens cannot attend to write
    text or its KV cache. Tokenizing pieces separately preserves positions for
    pointer and address supervision. With split_session=False, one Batch holds
    both writes and questions.
    """
    tok = mlm.tok
    enc = lambda t: tok(t, add_special_tokens=False).input_ids
    nl = enc("\n")
    writes, reads, spans, pw, pr, ap, an, sm = [], [], [], [], [], [], [], []
    for b in range(B):
        used: set = set()
        vars_ = [make_name(rng, used, 1, 2, cap=False, names=names) for _ in range(n_vars)]
        order = list(vars_) + [rng.choice(vars_) for _ in range(n_assign - n_vars)]
        state, last_val, all_val = {}, {}, {}
        w, bt = list(mlm.bos), [0] * len(mlm.bos)
        for i, v in enumerate(order):
            val = rng.choice(VALUES); state[v] = val
            for piece, role in [(("Set" if i == 0 else " Set"), None), (" " + v, "var"), (" to", None),
                                (" " + val, "val"), (".", None)]:
                ids = enc(piece)
                if role == "var":
                    var_pos = len(w) + len(ids) - 1
                if role == "val":
                    pw.append((b, len(w) + len(ids) - 1, var_pos))
                    last_val[v] = len(w) + len(ids) - 1
                    all_val.setdefault(v, []).append(last_val[v])
                w += ids; bt += [1 if role == "val" else 0] * len(ids)
        w += nl; bt += [0] * len(nl)
        lv = list(last_val.values())
        an += [(b, lv[i], lv[j]) for i in range(len(lv)) for j in range(i + 1, len(lv))]
        sm += [(b, p, last_val[v]) for v, ps in all_val.items() for p in ps[:-1]]     # Pair repeated writes of the same variable.
        ids, labels = (list(mlm.bos), [-100] * len(mlm.bos)) if split_session else (list(w), [-100] * len(w))
        for v in rng.sample(vars_, n_q):
            p1, p2, p3 = enc("Q: What is"), enc(" " + v), enc("?\nA:")
            var_pos = len(ids) + len(p1) + len(p2) - 1
            p = p1 + p2 + p3
            a = enc(" " + state[v])
            ids += p; labels += [-100] * len(p)
            pr.append((b, len(ids) - 1, var_pos))
            ap.append((b, len(ids) - 1, last_val[v]))
            spans.append((b, len(ids), len(ids) + len(a), "scratchpad"))
            ids += a; labels += a
            ids += nl; labels += nl
        writes.append((w, [-100] * len(w), bt)); reads.append((ids, labels, None))
    pad = tok.pad_token_id or 0
    ids, labels, attn = _pack(reads, pad, mlm.device)
    dev = mlm.device
    beta_r = torch.where(attn > 0, 0, -1)                     # Query sessions have zero write targets.
    ptr_w, addr_neg = torch.tensor(pw, device=dev), torch.tensor(an, device=dev)
    addr_same = torch.tensor(sm, dtype=torch.long, device=dev).view(-1, 3)
    prefix = None
    if split_session:
        wi, wl, wa = _pack(writes, pad, dev)
        bt = torch.full_like(wi, -1)
        for i, (_, _, t) in enumerate(writes):
            bt[i, :len(t)] = torch.tensor(t)
        prefix = Batch(wi, wl, wa, [], ptr_w=ptr_w, beta_tgt=bt, addr_neg=addr_neg, addr_same=addr_same)
    else:                                                     # Unsplit writes retain their original positions.
        for i, (_, _, t) in enumerate(writes):
            beta_r[i, :len(t)] = torch.tensor(t)
    return Batch(ids, labels, attn, spans, prefix=prefix, ptr_r=torch.tensor(pr, device=dev), beta_tgt=beta_r,
                 addr_pair=torch.tensor(ap, device=dev),
                 ptr_w=None if split_session else ptr_w, addr_neg=None if split_session else addr_neg,
                 addr_same=None if split_session else addr_same)
