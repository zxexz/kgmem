"""Knowledge graph storage and traversal scorers.

Scorers accept head/relation IDs and return tail IDs with probabilities. The
memory block links names using LM-derived keys; scorer embeddings stay private.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch


class Scorer(Protocol):
    def __call__(self, h: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h, r: LongTensor[P]. Returns tails LongTensor[P, M], probs FloatTensor[P, M] (rows sum <= 1)."""
        ...


@dataclass
class KnowledgeGraph:
    entities: list[str]                       # surface names; index = entity id
    relations: list[str]                      # surface names; index = relation id
    triples: list[tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self):
        self.ent_index = {n: i for i, n in enumerate(self.entities)}
        self.rel_index = {n: i for i, n in enumerate(self.relations)}
        self._version = 0

    def add(self, h: str, r: str, t: str):
        self.triples.append((self.ent_index[h], self.rel_index[r], self.ent_index[t]))
        self._version += 1

    def set(self, h: str, r: str, t: str):
        """Functional update: replace every (h, r, *) with (h, r, t)."""
        hi, ri = self.ent_index[h], self.rel_index[r]
        self.triples = [x for x in self.triples if not (x[0] == hi and x[1] == ri)]
        self.add(h, r, t)

    def tails(self, h: str, r: str) -> list[str]:
        hi, ri = self.ent_index[h], self.rel_index[r]
        return [self.entities[t] for (a, b, t) in self.triples if a == hi and b == ri]


class ExactScorer:
    """Look up stored edges using sorted head/relation keys.

    Storage is O(number of triples). Each pair returns at most max_tails edges,
    with uniform probability over the retained edges. Graph edits rebuild the
    index on the next lookup.
    """

    def __init__(self, kg: KnowledgeGraph, max_tails: int = 4, device: str | torch.device = "cpu"):
        self.kg, self.M, self.device = kg, max_tails, torch.device(device)
        self._built_version = -1
        self._build()

    def _build(self):
        R = len(self.kg.relations)
        t = torch.tensor(self.kg.triples, dtype=torch.long).view(-1, 3)
        keys = t[:, 0] * R + t[:, 1]
        order = torch.argsort(keys, stable=True)
        self.keys = keys[order].to(self.device)
        self.tails_flat = t[order, 2].to(self.device)
        self.R = R
        self._built_version = self.kg._version

    @torch.no_grad()
    def __call__(self, h: torch.Tensor, r: torch.Tensor):
        if self._built_version != self.kg._version:
            self._build()
        P, M = h.numel(), self.M
        if self.keys.numel() == 0:
            return (torch.zeros(P, M, dtype=torch.long, device=h.device),
                    torch.zeros(P, M, device=h.device))
        q = h * self.R + r
        start = torch.searchsorted(self.keys, q)
        end = torch.searchsorted(self.keys, q, right=True)
        cnt = (end - start).clamp(max=M)
        ar = torch.arange(M, device=h.device)
        idx = (start[:, None] + ar).clamp(max=self.keys.numel() - 1)
        mask = ar[None, :] < cnt[:, None]
        tails = torch.where(mask, self.tails_flat[idx], torch.zeros_like(idx))
        probs = mask.float() / cnt.clamp(min=1)[:, None].float()
        return tails, probs

    @torch.no_grad()
    def has_rel(self) -> torch.Tensor | None:
        """Return an (E, R) edge-existence mask, or None when E * R exceeds 2**28."""
        if self._built_version != self.kg._version:
            self._build()
        E, R = len(self.kg.entities), self.R
        if E * R > 1 << 28:
            return None
        if getattr(self, "_has_version", None) != self._built_version:
            has = torch.zeros(E * R, dtype=torch.bool, device=self.device)
            has[self.keys] = True
            self._has, self._has_version = has.view(E, R), self._built_version
        return self._has


class PyKEENScorer:
    """Score tails with a trained PyKEEN model.

    Entity and relation IDs must match the loaded graph; kg_from_pykeen preserves
    the factory's order. Unless chunk is supplied, row chunks target max_scores
    entries (256 MiB in fp32 by default). A single row still scores every entity.
    """

    def __init__(self, model, max_tails: int = 4, temperature: float = 1.0, chunk: int | None = None,
                 min_prob: float = 0.0, max_scores: int = 1 << 26):
        self.model, self.M, self.T, self.chunk, self.min_prob = model.eval(), max_tails, temperature, chunk, min_prob
        self.max_scores = max_scores

    @torch.no_grad()
    def __call__(self, h: torch.Tensor, r: torch.Tensor):
        dev = next(self.model.parameters()).device
        hr = torch.stack([h, r], 1).to(dev)
        chunk = self.chunk or max(1, self.max_scores // self.model.num_entities)
        tails, probs = [], []
        for i in range(0, hr.shape[0], chunk):
            s = self.model.score_t(hr_batch=hr[i:i + chunk]).div_(self.T)  # (P, E); avoid another full-width probability tensor.
            top = s.topk(self.M, dim=-1)
            p = (top.values - s.logsumexp(-1, keepdim=True)).exp()        # Full-softmax probabilities at the selected tails.
            tails.append(top.indices)
            probs.append(p * (p >= self.min_prob))
        return torch.cat(tails).to(h.device), torch.cat(probs).to(h.device)


def kg_from_pykeen(tf, entity_labels: dict[str, str] | None = None,
                   relation_labels: dict[str, str] | None = None) -> KnowledgeGraph:
    """Copy a PyKEEN TriplesFactory without changing its entity or relation IDs.

    Optional label dictionaries replace raw IDs with the names used for LM keys
    and copied answers.
    """
    ents = [k for k, _ in sorted(tf.entity_to_id.items(), key=lambda x: x[1])]
    rels = [k for k, _ in sorted(tf.relation_to_id.items(), key=lambda x: x[1])]
    if entity_labels:
        ents = [entity_labels.get(e, e) for e in ents]
    if relation_labels:
        rels = [relation_labels.get(r, r) for r in rels]
    triples = [tuple(x) for x in tf.mapped_triples.tolist()]
    return KnowledgeGraph(ents, rels, triples)


@dataclass
class GraphTensors:
    """Runtime graph tables built by MemoryLM.load_graph()."""
    ent_keys: torch.Tensor          # (E, d_model)   frozen LM's own representation of each name
    rel_keys: torch.Tensor          # (R, d_model)
    name_tok: torch.Tensor          # (E, Lmax) token ids of " " + name, padded with -1
    name_len: torch.Tensor          # (E,)
    scorer: Callable
    kg: KnowledgeGraph | None = None
    rel_keys_first: torch.Tensor | None = None   # (R, d_model) state at each relation name's first token
    alias: dict | None = None                    # name token tuple -> entity id (-1 if two share it), for exact_link
    rel_alias: dict | None = None                # the same for relation names
