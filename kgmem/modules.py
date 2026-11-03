"""Graph retrieval, delta-rule scratchpad, and entity-name copy head.

The block adds gated graph and scratchpad reads to the decoder residual.
Scratchpad reads also contribute a logit bias through the LM head. The copy
head mixes entity-name tokens with the LM distribution outside this module's
MemoryBlock.forward().
"""
from __future__ import annotations

import math
import weakref
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels


@dataclass
class MemConfig:
    d_q: int = 256  # Linker query/key dimension.
    d_val: int = 256  # Value dimension shared by graph and scratchpad reads.
    k_ent: int = 4  # Entity candidates per token.
    k_rel: int = 3  # Relation candidates per hop.
    k_cand: int = 8  # Entity candidates retained after each hop.
    max_tails: int = 4  # Maximum tails returned per head/relation pair.
    hops: int = 2  # Maximum graph chain length.
    use_kg: bool = True  # Construct the graph read head.
    use_scratchpad: bool = True  # Construct the delta-rule scratchpad.
    n_slots: int = 64  # Scratchpad address dimension.
    hard_addressing: bool = False  # Straight-through one-hot addresses.
    decay: bool = False  # Learn alpha; otherwise the recurrence uses alpha = 1.
    key_pool: str = "last"  # Entity key pooling: first, last, or mean.
    rel_key_pool: str = "mean"  # Mean pooling retains differences within multi-token relation names.
    k_mention: int = 2  # Mention positions retained per query.
    d_att: int = 64  # Mention/pointer attention dimension.
    read_norm: bool = False  # Normalize scratchpad reads before hidden and logit readouts.
    read_gate: float = 4.0  # With read_norm, attenuate reads below this norm.
    kg_gate_loss: float = 1.0  # Graph-gate supervision weight at answer and non-answer positions.
    rel_mention: bool = False  # Mix query-based relation scores with scores from relation-name tokens.
    relm_raw_cos: float = 0.0  # Weight of unprojected token/key cosine in relation-name scores.
    relm_first: bool = False  # Also score against first-token keys of relation names.
    rel_halt: bool = False  # Add named-relation confidence and attention overlap to halting logits.
    l2_keys: bool = False  # Unit-length keys make a full-strength write replace the value along its address.
    cont_keys: bool = False  # Unit-vector addresses instead of a softmax over slots.
    readout_gate: bool = False  # Supervised gate on the scratchpad logit readout.
    gate_pad_value: bool = False  # Apply readout_gate to the scratchpad residual contribution too.
    hop_order: float = 0.0  # Inference threshold for ordering named relations by distance from the head mention.
    carry_fresh: float = 1.0  # Weight of fresh candidates while a retrieved name is being emitted.
    carry_rel: bool = False  # Interpret carry_fresh relative to carried mass, not as an absolute multiplier.
    carry_share: float = 0.0  # Minimum share of total candidate mass for carry; zero uses weight > 0.05.
    carry_top: float = 0.0  # Positive values override carry_share: minimum fraction of the top weight.
    halt_rel_aux: float = 0.0  # Loss weight for the named-relation halting term alone.
    halt_rel_same: bool = False  # Include consecutive hops' relation-distribution overlap in halting features.
    carry_commit: bool = False  # Floor the copy gate at its value where the carried name started.
    exact_link: float = 0.0  # Prior mass assigned to an unambiguous exact token-name match.
    merge_cands: bool = False  # Sum weights of paths to the same entity before top-k pruning.
    rel_exists: bool = False  # At inference, weight relations by heads with an edge; requires scorer.has_rel().
    rel_decoder: int = 0  # Positive width replaces per-hop relation linkers and halting heads with a decoder.
    rel_exists_soft: float = 0.0  # With rel_exists, scale candidate mass by m / (m + this) to retain uncertainty.
    rel_named: float = 0.0  # Inference boost for relations named in full within the last 32 tokens.
    exclude_head: bool = False  # Exclude the linked head from answers and intermediate nodes after hop 1.
    end_gate: bool = False  # Learn to propose a newline after a retrieved name is complete.


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def merge_topk(ids, w, k):
    """Sum repeated entity IDs within each row, then retain the k largest weights.

    Returns zero-padded (N, k) IDs and weights. Merge before ranking so multiple
    paths to one entity contribute to its rank.
    """
    N, K = ids.shape
    E = int(ids.max().item()) + 1 if ids.numel() else 1
    key = torch.arange(N, device=ids.device)[:, None] * E + ids
    uniq, inv = torch.unique(key.flatten(), return_inverse=True)
    summed = torch.zeros(uniq.shape[0], device=w.device, dtype=w.dtype).scatter_add_(0, inv, w.flatten())
    row, ent = uniq // E, uniq % E
    order = torch.argsort(row.double() * 4.0 - (summed.double() / summed.double().max().clamp(min=1e-30)))
    row, ent, summed = row[order], ent[order], summed[order]          # rows grouped, heaviest first within a row
    first = torch.searchsorted(row, torch.arange(N, device=ids.device))
    rank = torch.arange(row.shape[0], device=ids.device) - first[row]
    keep = rank < k
    out_ids = torch.zeros(N, k, dtype=ids.dtype, device=ids.device)
    out_w = torch.zeros(N, k, dtype=w.dtype, device=w.device)
    out_ids[row[keep], rank[keep]] = ent[keep]
    out_w[row[keep], rank[keep]] = summed[keep]
    return out_ids, out_w


def covered_until(ids, alias, L):
    """Return the earliest exact-name end that covers each token, or 2**30.

    The (B, T) result excludes interior tokens from mention attention once their
    containing name has ended. This prevents a title such as "Sukiyaki Western
    Django" from linking through its interior "Western" token to a genre entity.
    """
    rows = ids.tolist()
    T = len(rows[0]) if rows else 0
    out = []
    for row in rows:
        cov = [1 << 30] * T
        for e in range(T):
            for l in range(2, min(L, e + 1) + 1):
                if alias.get(tuple(row[e - l + 1:e + 1]), -1) >= 0:
                    for s_ in range(e - l + 1, e):
                        cov[s_] = min(cov[s_], e)
        out.append(cov)
    return torch.tensor(out, dtype=torch.long)


def exact_mentions(ids, Tn, alias, L):
    """Return the longest exact-name match ending at each of the last Tn tokens.

    The result has shape (B, Tn), with -1 for no match or an ambiguous alias.
    Only tokens through each position count, matching incremental decoding.
    """
    rows = ids.tolist()
    out = []
    for row in rows:
        Th = len(row)
        hits = []
        for t in range(Th - Tn, Th):
            e = -1
            for l in range(1, min(L, t + 1) + 1):
                e = alias.get(tuple(row[t - l + 1:t + 1]), e)
            hits.append(e)
        out.append(hits)
    return torch.tensor(out, dtype=torch.long)


def _topk_chunked(q, K, k, max_scores=1 << 26):
    """Select top-k keys in row chunks sized from max_scores.

    The default targets 256 MiB of fp32 scores; each row still scores every key.
    A fixed 8192-row chunk would require about 32 GB against one million keys.
    """
    chunk = max(1, max_scores // max(1, K.shape[0]))
    with torch.no_grad():
        return torch.cat([(q[i:i + chunk] @ K.T).topk(k, -1).indices for i in range(0, q.shape[0], chunk)])


def _project_keys(mod, keys, chunk=1 << 16):
    """Return normalized projected keys, caching them when gradients are disabled.

    The cache tracks tensor identity and parameter versions. Inference upcasts
    one chunk at a time to avoid a full fp32 copy of a bf16 key table. Training
    recomputes the projection to retain its autograd graph.
    """
    if torch.is_grad_enabled():
        return F.normalize(mod.kp(mod.knorm(keys.float())), dim=-1)
    tag = [(p.data_ptr(), p._version) for p in (mod.kp.weight, mod.knorm.w)] + [keys._version]
    c = getattr(mod, "_kcache", None)
    if c is None or c[0]() is not keys or c[1] != tag:     # weakref: the cache must not keep a swapped-out table alive
        K = torch.cat([F.normalize(mod.kp(mod.knorm(keys[i:i + chunk].float())), dim=-1)
                       for i in range(0, keys.shape[0], chunk)])
        c = mod._kcache = (weakref.ref(keys), tag, K)
    return c[2]


class TopKLinker(nn.Module):
    """Select top-k keys by cosine similarity, then compute their softmax weights.

    Selection is chunked and outside autograd. Selected scores are recomputed
    with gradients, avoiding an N-by-E training score tensor.
    """

    def __init__(self, d_model: int, d_key: int, d_q: int, k: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_q, bias=False)
        self.knorm = RMSNorm(d_key)
        self.kp = nn.Linear(d_key, d_q, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.k = k

    def forward(self, h, keys, aux_idx=None):
        q = F.normalize(self.q(h), dim=-1)                          # (N, dq)
        K = _project_keys(self, keys)                               # (E, dq)
        idx = _topk_chunked(q, K, min(self.k, K.shape[0]))
        scale = self.log_scale.exp()
        w = ((q[:, None, :] * K[idx]).sum(-1) * scale).softmax(-1)
        full = (q[aux_idx] @ K.T) * scale if aux_idx is not None else None
        return idx, w, full


def _exact_prior(idx, w, ex, alpha):
    """Mix an exact-match prior of mass alpha into each candidate row.

    An absent entity (ex == -1) leaves the row unchanged. A new matched entity
    replaces the weakest candidate; remaining weights are renormalized.
    """
    has = ex >= 0
    present = idx == ex[:, None]
    slot = torch.where(present.any(-1), present.float().argmax(-1), w.argmin(-1))
    one = F.one_hot(slot, idx.shape[1]).to(w.dtype)
    kept = w * (1 - one * (~present.any(-1))[:, None].to(w.dtype))                   # drop a replaced slot's weight
    kept = kept / kept.sum(-1, keepdim=True).clamp(min=1e-9)
    idx = torch.where(has[:, None] & one.bool(), ex[:, None], idx)
    w = torch.where(has[:, None], (1 - alpha) * kept + alpha * one, w)
    return idx, w


class MentionLinker(nn.Module):
    """Link tokens to entities, then attend from each query to mention tokens.

    Each query inherits the entity candidates of its top mention positions.
    A per-position cache supports incremental decoding.
    """

    def __init__(self, d_model: int, d_key: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.m = nn.Linear(d_model, cfg.d_q, bias=False)
        self.knorm = RMSNorm(d_key)
        self.kp = nn.Linear(d_key, cfg.d_q, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.aq = nn.Linear(d_model, cfg.d_att, bias=False)
        self.ak = nn.Linear(d_model, cfg.d_att, bias=False)

    def forward(self, x, keys, cache=None, pad_mask=None, aux_idx=None, aux_mention=None, exact=None, cover=None):
        B, Tn, D = x.shape
        q = F.normalize(self.m(x), dim=-1).reshape(B * Tn, -1)
        K = _project_keys(self, keys)
        k = min(self.cfg.k_ent, K.shape[0])
        idx = _topk_chunked(q, K, k)
        scale = self.log_scale.exp()
        tok_w = ((q[:, None, :] * K[idx]).sum(-1) * scale).softmax(-1)                # (B*Tn, k)
        if exact is not None:
            idx, tok_w = _exact_prior(idx, tok_w, exact.reshape(-1), self.cfg.exact_link)
        ak = self.ak(x)
        valid = torch.ones(B, Tn, dtype=torch.bool, device=x.device) if pad_mask is None else pad_mask.bool()
        new = dict(ak=ak, ids=idx.view(B, Tn, k), w=tok_w.view(B, Tn, k), valid=valid)
        if cache is not None:                                                         # incremental decoding
            new = {n: torch.cat([cache[n], new[n]], 1) for n in new}
        T = new["ak"].shape[1]
        sc = torch.einsum("bqd,bsd->bqs", self.aq(x), new["ak"]) / math.sqrt(self.cfg.d_att)
        qpos = torch.arange(T - Tn, T, device=x.device)
        allowed = (torch.arange(T, device=x.device)[None, :] <= qpos[:, None])[None] & new["valid"][:, None, :]
        if cover is not None:                      # exact_link's longest match: not inside a name that has ended
            allowed = allowed & ~(cover[:, None, :] <= qpos[None, :, None])
        sc = sc.masked_fill(~allowed, -1e9)
        att = sc.softmax(-1)                                                          # (B,Tn,T)
        m = min(self.cfg.k_mention, T)
        with torch.no_grad():
            pos = att.topk(m, -1).indices                                             # (B,Tn,m)
        self.last_top = pos[..., 0]                                                   # read by hop ordering
        a = att.gather(-1, pos)
        a = a / a.sum(-1, keepdim=True).clamp(min=1e-9)
        bidx = torch.arange(B, device=x.device)[:, None, None]
        cand_ids = new["ids"][bidx, pos].reshape(B, Tn, m * k)
        cand_w = (a[..., None] * new["w"][bidx, pos]).reshape(B, Tn, m * k)
        aux = None
        if aux_idx is not None:
            tok_full = None
            if aux_mention is not None:
                ok = aux_mention >= 0
                mpos = (aux_idx // Tn) * Tn + aux_mention.clamp(min=0)
                tok_full = (q[mpos] @ K.T) * scale
            aux = dict(att=sc.reshape(B * Tn, T)[aux_idx], tok=tok_full,
                       q_ids=cand_ids.reshape(B * Tn, -1)[aux_idx], q_w=cand_w.reshape(B * Tn, -1)[aux_idx])
        cache_out = {n: v.detach() for n, v in new.items()}
        return cand_ids, cand_w, cache_out, aux


class RelationMentionLinker(nn.Module):
    """Link tokens to relations and attend to them separately for each hop.

    Relation distributions remain dense over R. Returns (B, Tn, R) probabilities
    and mention-attention logits for each hop, mixing gates, token logits, and a
    cache for incremental decoding.
    """

    def __init__(self, d_model: int, d_key: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.m = nn.Linear(d_model, cfg.d_q, bias=False)
        self.knorm = RMSNorm(d_key)
        self.kp = nn.Linear(d_key, cfg.d_q, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.aq = nn.ModuleList([nn.Linear(d_model, cfg.d_att, bias=False) for _ in range(cfg.hops)])
        self.ak = nn.Linear(d_model, cfg.d_att, bias=False)
        self.mix = nn.Linear(d_model, cfg.hops)                 # Per-hop mixing of mention-based and query-based relation scores.

    def _order_hops(self, scs, relness, mention_pos, q0):
        """The hop_order rule; see MemConfig. Rewrites a query's hop attentions only when two or more relation
        names are present and the trained attentions collapsed onto one of them."""
        scs = [sc.clone() for sc in scs]
        B, Tn, _ = scs[0].shape
        for b in range(B):
            for i in range(Tn):
                q, m = q0 + i, int(mention_pos[b, i])
                # Limit the scan near attended mentions to exclude relations in earlier questions.
                lo = max(1, min(int(sc[b, i].argmax()) for sc in scs) - 8)
                spans = []                                          # adjacent relation-like tokens = one name
                for s_ in range(lo, q + 1):                         # Exclude the position-zero attention sink.
                    if relness[b, s_] >= self.cfg.hop_order and s_ != m:
                        if spans and s_ == spans[-1][-1] + 1:
                            spans[-1].append(s_)
                        else:
                            spans.append([s_])
                if len(spans) < 2:
                    continue
                which = [next((k for k, sp in enumerate(spans) if int(sc[b, i].argmax()) in sp), -1) for sc in scs]
                if len(set(which)) == len(scs) and -1 not in which:
                    continue                                        # Preserve already distinct hop assignments.
                near = sorted(range(len(spans)), key=lambda k: abs(spans[k][-1] - m))
                for h, k in enumerate(near[:len(scs)]):
                    row = torch.full_like(scs[h][b, i], -1e9)
                    row[spans[k][-1]] = 0.0
                    scs[h][b, i] = row
        return scs

    def forward(self, x, keys, cache=None, pad_mask=None, keys_first=None, mention_pos=None, ent_keys=None,
                exact=None):
        B, Tn, _ = x.shape
        q = F.normalize(self.m(x), dim=-1)
        K = _project_keys(self, keys)
        tok_logits = q @ K.T                                       # (B,Tn,R)
        if self.cfg.relm_first and keys_first is not None:        # Include the first-token key for multi-token names.
            tok_logits = torch.maximum(tok_logits, q @ _project_keys(self, keys_first).T)
        tok_logits = tok_logits * self.log_scale.exp()
        relness = torch.zeros(B, Tn, device=x.device)
        if self.cfg.relm_raw_cos:
            xn = F.normalize(x, dim=-1)
            raw = xn @ F.normalize(keys.float(), dim=-1).T
            if keys_first is not None:      # Match either endpoint of a multi-token relation name.
                raw = torch.maximum(raw, xn @ F.normalize(keys_first.float(), dim=-1).T)
            tok_logits = tok_logits + self.cfg.relm_raw_cos * raw
            relness = raw.max(-1).values.detach()
            if self.cfg.hop_order and ent_keys is not None:
                # Entity-like tokens must not split a relation into extra hops.
                with torch.no_grad():
                    En = F.normalize(ent_keys.float(), dim=-1)
                    flat = xn.reshape(B * Tn, -1)
                    ent = torch.cat([(flat[i:i + 4096] @ En.T).max(-1).values for i in range(0, flat.shape[0], 4096)])
                relness = torch.where(relness > ent.view(B, Tn), relness, torch.zeros_like(relness))
        valid = torch.ones(B, Tn, dtype=torch.bool, device=x.device) if pad_mask is None else pad_mask.bool()
        p = tok_logits.softmax(-1)
        if exact is not None:                                      # exact_link: the name is spelled right here
            has = (exact >= 0)[..., None]
            one = F.one_hot(exact.clamp(min=0), p.shape[-1]).to(p.dtype)
            p = torch.where(has, (1 - self.cfg.exact_link) * p + self.cfg.exact_link * one, p)
        new = dict(ak=self.ak(x), p=p, valid=valid, rn=relness)
        if cache is not None:
            new = {n: torch.cat([cache[n], new[n]], 1) for n in new}
        T = new["ak"].shape[1]
        qpos = torch.arange(T - Tn, T, device=x.device)
        allowed = (torch.arange(T, device=x.device)[None, :] <= qpos[:, None])[None] & new["valid"][:, None, :]
        scs = []
        for aq in self.aq:
            sc = torch.einsum("bqd,bsd->bqs", aq(x), new["ak"]) / math.sqrt(self.cfg.d_att)
            scs.append(sc.masked_fill(~allowed, -1e9))
        if self.cfg.hop_order and self.cfg.relm_raw_cos and mention_pos is not None and not torch.is_grad_enabled():
            scs = self._order_hops(scs, new["rn"], mention_pos, T - Tn)
        hops = [(torch.einsum("bqs,bsr->bqr", sc.softmax(-1), new["p"]), sc) for sc in scs]
        rho = torch.sigmoid(self.mix(x))                          # (B,Tn,hops)
        return hops, rho, tok_logits, {n: v.detach() for n, v in new.items()}


class RelationDecoder(nn.Module):
    """Decode a relation chain with a recurrent state at each query position.

    Each hop attends causally to token states and consumes the previous hop's
    expected relation key. Returns relation logits (B, Tn, R) per hop, halting
    logits (B, Tn, hops), and cached token projections.
    """

    def __init__(self, d_model: int, d_key: int, cfg):
        super().__init__()
        d, self.heads, self.hops = cfg.rel_decoder, 4, cfg.hops
        self.norm = RMSNorm(d_model)
        self.k, self.v, self.q0 = (nn.Linear(d_model, d) for _ in range(3))
        self.qh = nn.Linear(d, d)
        self.cell = nn.GRUCell(2 * d, d)
        self.out = nn.Linear(d, d)
        self.knorm = RMSNorm(d_key)
        self.kp = nn.Linear(d_key, d, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.stop = nn.Linear(d, 1)

    def forward(self, x, rel_keys, cache=None, pad_mask=None):
        B, Tn, _ = x.shape
        xn = self.norm(x)
        valid = torch.ones(B, Tn, dtype=torch.bool, device=x.device) if pad_mask is None else pad_mask[:, -Tn:].bool()
        new = dict(k=self.k(xn), v=self.v(xn), valid=valid)
        if cache is not None:
            new = {n: torch.cat([cache[n], new[n]], 1) for n in new}
        T, H, dh = new["k"].shape[1], self.heads, new["k"].shape[-1] // self.heads
        qpos = torch.arange(T - Tn, T, device=x.device)
        allowed = (torch.arange(T, device=x.device)[None, :] <= qpos[:, None])[None] & new["valid"][:, None, :]
        K = F.normalize(self.kp(self.knorm(rel_keys.to(xn.dtype))), dim=-1)          # (R, d)
        kk, vv = new["k"].view(B, T, H, dh), new["v"].view(B, T, H, dh)
        s = torch.tanh(self.q0(xn)).reshape(B * Tn, -1)
        prev = torch.zeros_like(s)
        logits, stops = [], []
        for _ in range(self.hops):
            q = self.qh(s).view(B, Tn, H, dh)
            att = torch.einsum("bqhd,bkhd->bhqk", q, kk) / math.sqrt(dh)
            att = att.masked_fill(~allowed[:, None], -1e9).softmax(-1)
            read = torch.einsum("bhqk,bkhd->bqhd", att, vv).reshape(B * Tn, -1)
            s = self.cell(torch.cat([read, prev], -1), s)
            lg = (F.normalize(self.out(s), dim=-1) @ K.T) * self.log_scale.exp()      # (B*Tn, R)
            logits.append(lg.view(B, Tn, -1))
            stops.append(self.stop(s).view(B, Tn))
            prev = lg.softmax(-1) @ K                                                 # Expected relation key for the next recurrent step.
        return logits, torch.stack(stops, -1), {n: v.detach() for n, v in new.items()}


class KGReadHead(nn.Module):
    def __init__(self, d_model: int, d_key: int, cfg: MemConfig):
        super().__init__()
        self.cfg = cfg
        self.ent = MentionLinker(d_model, d_key, cfg)
        self.rel = nn.ModuleList([TopKLinker(d_model, d_key, cfg.d_q, cfg.k_rel) for _ in range(cfg.hops)])
        self.relm = RelationMentionLinker(d_model, d_key, cfg) if cfg.rel_mention else None
        # Per-hop confidence, adjacent attention overlap, and optional relation overlap.
        n_feat = 2 * cfg.hops - 1 + (cfg.hops - 1 if cfg.halt_rel_same else 0)
        self.halt_rel = nn.Linear(n_feat, cfg.hops) if cfg.rel_mention and cfg.rel_halt else None
        if self.halt_rel is not None:
            nn.init.zeros_(self.halt_rel.weight); nn.init.zeros_(self.halt_rel.bias)
        self.halt = nn.Linear(d_model, cfg.hops)
        self.dec = RelationDecoder(d_model, d_key, cfg) if getattr(cfg, "rel_decoder", 0) else None
        self.vnorm = RMSNorm(d_key)
        self.val = nn.Linear(d_key, cfg.d_val)

    def forward(self, x, g, ctx):
        B, T, D = x.shape
        aux_idx = ctx.aux_idx
        exact = exact_rel = cover = None
        if self.cfg.exact_link and getattr(g, "alias", None) and getattr(ctx, "tok_ids", None) is not None:
            exact = exact_mentions(ctx.tok_ids, T, g.alias, g.name_tok.shape[1]).to(x.device)
            cover = covered_until(ctx.tok_ids, g.alias, g.name_tok.shape[1]).to(x.device)
            if getattr(g, "rel_alias", None):
                exact_rel = exact_mentions(ctx.tok_ids, T, g.rel_alias, max(map(len, g.rel_alias))).to(x.device)
        ids, w, ctx.link_cache, ent_aux = self.ent(x, g.ent_keys, ctx.link_cache, ctx.write_mask,
                                                   aux_idx, ctx.aux_mention, exact=exact, cover=cover)
        h = x.reshape(B * T, D)
        N = B * T
        ids, w = ids.reshape(N, -1), w.reshape(N, -1)
        hop_ids, hop_w, rel_full = [], [], []
        R = g.rel_keys.shape[0]
        spelled = None                                                     # (N, R) relations the question names in full
        if self.cfg.rel_named and not torch.is_grad_enabled() and getattr(g, "rel_alias", None) \
                and getattr(ctx, "tok_ids", None) is not None:
            W = 32
            hits = exact_mentions(ctx.tok_ids, min(ctx.tok_ids.shape[1], T + W), g.rel_alias,
                                  max(map(len, g.rel_alias))).to(x.device)
            oh = F.one_hot(hits.clamp(min=0), R) * (hits >= 0)[..., None]
            cs = torch.cat([torch.zeros_like(oh[:, :1]), oh.cumsum(1)], 1)        # prefix sums over positions
            hi = torch.arange(hits.shape[1] - T, hits.shape[1], device=x.device)
            spelled = (cs[:, hi + 1] - cs[:, (hi - W).clamp(min=0)] > 0).reshape(B * T, R)
        head = ids.gather(1, w.argmax(-1, keepdim=True))                  # (N, 1) the linked entity, for exclude_head
        named = relm_aux = dec = None
        if self.dec is not None:
            dec, dec_halt, ctx.dec_cache = self.dec(x, g.rel_keys, getattr(ctx, "dec_cache", None), ctx.write_mask)
        elif self.relm is not None:
            named, rho, relm_tok, ctx.relm_cache = self.relm(x, g.rel_keys, ctx.relm_cache, ctx.write_mask,
                                                             keys_first=g.rel_keys_first, mention_pos=self.ent.last_top,
                                                             ent_keys=g.ent_keys, exact=exact_rel)
            rho = rho.reshape(N, -1)
            relm_aux = dict(p=[], att=[], tok=relm_tok.reshape(N, R), rho=rho[aux_idx]) if aux_idx is not None else None
        for hop, linker in enumerate(self.rel):
            if dec is not None:                       # Use recurrent relation logits instead of independent linkers.
                lg = dec[hop].reshape(N, R)
                p = lg.float().softmax(-1)
                r_w, r_ids = p.topk(min(self.cfg.k_rel, R), -1)
                r_w = r_w / r_w.sum(-1, keepdim=True).clamp(min=1e-9)
                rf = lg[aux_idx] if aux_idx is not None else None
            else:
                r_ids, r_w, rf = linker(h, g.rel_keys, aux_idx)
            if named is not None:                     # Mix relation-mention probabilities before top-k selection.
                dense = torch.zeros(N, R, device=h.device, dtype=r_w.dtype).scatter_add(1, r_ids, r_w)
                p = (1 - rho[:, hop:hop + 1]) * dense + rho[:, hop:hop + 1] * named[hop][0].reshape(N, R)
                r_w, r_ids = p.topk(min(self.cfg.k_rel, R), -1)
                r_w = r_w / r_w.sum(-1, keepdim=True).clamp(min=1e-9)
                if relm_aux is not None:
                    relm_aux["p"].append(p[aux_idx])
                    relm_aux["att"].append(named[hop][1].reshape(N, -1)[aux_idx])
            has = g.scorer.has_rel() if self.cfg.rel_exists and not torch.is_grad_enabled() \
                and hasattr(g.scorer, "has_rel") else None
            if has is not None or spelled is not None:
                dense = p.to(r_w.dtype) if (named is not None or dec is not None) else \
                    torch.zeros(N, R, device=h.device, dtype=r_w.dtype).scatter_add(1, r_ids, r_w)
                if spelled is not None:
                    dense = dense * (1 + self.cfg.rel_named * spelled.to(dense.dtype))
                    dense = dense / dense.sum(-1, keepdim=True).clamp(min=1e-9)
                if has is not None:
                    hw = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
                    ex = (hw[:, :, None] * has.to(h.device)[ids].to(hw.dtype)).sum(1)       # (N, R) head mass with r
                    q = dense * ex.to(dense.dtype)
                    tot = q.sum(-1, keepdim=True)
                    dense = torch.where(tot > 0, q / tot.clamp(min=1e-9), dense)
                r_w, r_ids = dense.topk(min(self.cfg.k_rel, R), -1)
                r_w = r_w / r_w.sum(-1, keepdim=True).clamp(min=1e-9)
                if has is not None and self.cfg.rel_exists_soft > 0:
                    r_w = r_w * (tot / (tot + self.cfg.rel_exists_soft)).to(r_w.dtype)
            kh, kr = ids.shape[1], r_ids.shape[1]
            ph = ids[:, :, None].expand(N, kh, kr).reshape(-1)
            pr = r_ids[:, None, :].expand(N, kh, kr).reshape(-1)
            pw = (w[:, :, None] * r_w[:, None, :]).reshape(N, kh * kr, 1)
            # Deduplicate repeated lookups; PyKEEN scores all E entities for each distinct pair.
            pair, inv = torch.unique(ph * R + pr, return_inverse=True)
            tails, tp = g.scorer(pair // R, pair % R)
            tails, tp = tails[inv], tp[inv]                               # (N*kh*kr, M)
            M = tails.shape[1]
            cw = (pw * tp.view(N, kh * kr, M)).view(N, -1)
            ct = tails.view(N, -1)
            if self.cfg.exclude_head and hop >= 1:   # neither an answer nor a step on the way back through the head
                cw = cw * (ct != head)
            if self.cfg.merge_cands:
                ids, w = merge_topk(ct, cw, min(self.cfg.k_cand, cw.shape[1]))
            else:
                top = cw.topk(min(self.cfg.k_cand, cw.shape[1]), -1)
                ids, w = ct.gather(1, top.indices), top.values           # becomes next hop's heads
            hop_ids.append(ids); hop_w.append(w); rel_full.append(rf)
        halt_logits, halt_rel = (dec_halt.reshape(N, -1) if dec is not None else self.halt(h)), None
        if self.halt_rel is not None and dec is None:
            atts = [hp[1].softmax(-1) for hp in named]
            feats = [hp[0].max(-1).values for hp in named] + [(a * b).sum(-1) for a, b in zip(atts, atts[1:])]
            if self.cfg.halt_rel_same:                                # Adjacent relation-distribution overlap.
                feats += [(a[0] * b[0]).sum(-1) for a, b in zip(named, named[1:])]
            halt_rel = self.halt_rel(torch.stack(feats, -1).reshape(N, -1))
            halt_logits = halt_logits + halt_rel
        halt = halt_logits.softmax(-1)
        cand_ids = torch.cat(hop_ids, 1)
        cand_w = torch.cat([halt[:, i:i + 1] * hw for i, hw in enumerate(hop_w)], 1)
        value = (cand_w[..., None] * self.val(self.vnorm(g.ent_keys[cand_ids].float()))).sum(1)
        aux = dict(ent=ent_aux, rel=rel_full, halt=halt_logits[aux_idx] if aux_idx is not None else None, relm=relm_aux,
                   halt_rel=halt_rel[aux_idx] if (aux_idx is not None and halt_rel is not None) else None)
        return value.view(B, T, -1), cand_ids.view(B, T, -1), cand_w.view(B, T, -1), aux


def _address(logits, hard: bool):
    soft = logits.softmax(-1)
    if not hard:
        return soft
    one_hot = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
    return one_hot - soft.detach() + soft               # straight-through


class PointerAttention(nn.Module):
    """Causally attend to token states used to form scratchpad addresses.

    Separate read and write pointers can select the same variable-name token;
    the shared address map then gives them the same address.
    """

    def __init__(self, d_model: int, d_att: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_att, bias=False)
        self.k = nn.Linear(d_model, d_att, bias=False)
        self.d_att = d_att

    def forward(self, x, cache=None, valid=None):
        B, Tn, _ = x.shape
        valid = torch.ones(B, Tn, dtype=torch.bool, device=x.device) if valid is None else valid.bool()
        new = dict(k=self.k(x), x=x, valid=valid)
        if cache is not None:
            new = {n: torch.cat([cache[n], new[n]], 1) for n in new}
        T = new["k"].shape[1]
        sc = torch.einsum("bqd,bsd->bqs", self.q(x), new["k"]) / math.sqrt(self.d_att)
        qpos = torch.arange(T - Tn, T, device=x.device)
        allowed = (torch.arange(T, device=x.device)[None, :] <= qpos[:, None])[None] & new["valid"][:, None, :]
        sc = sc.masked_fill(~allowed, -1e9)
        self.last_logits = sc                                     # (B,Tn,T) for pointer supervision
        return torch.einsum("bqs,bsd->bqd", sc.softmax(-1), new["x"]), {n: v.detach() for n, v in new.items()}


class DeltaScratchpad(nn.Module):
    """Read-before-write delta-rule memory S, shaped (B, n_slots, d_val).

        r_t = q_t^T S_{t-1}
        v_t = W_v h_t + W_r r_t
        S_t = a_t S_{t-1} + b_t k_t (v_t - k_t^T (a_t S_{t-1}))^T

    With one-hot k_t, b_t = 1, and a_t = 1, a write replaces one row exactly.
    Read and write pointers share an address map. When decay is disabled, a_t = 1.
    """

    def __init__(self, d_model: int, cfg: MemConfig):
        super().__init__()
        self.cfg = cfg
        self.ptr_w = PointerAttention(d_model, cfg.d_att)
        self.ptr_r = PointerAttention(d_model, cfg.d_att)
        self.addr = nn.Linear(d_model, cfg.n_slots)
        self.addr_scale = nn.Parameter(torch.tensor(math.log(2.0)))
        self.beta = nn.Linear(d_model, 1)
        nn.init.constant_(self.beta.bias, -2.0)             # Initially suppress writes.
        self.v = nn.Linear(d_model, cfg.d_val)
        self.v_r = nn.Linear(cfg.d_val, cfg.d_val, bias=False)
        nn.init.zeros_(self.v_r.weight)
        self.alpha = nn.Linear(d_model, 1) if cfg.decay else None
        if self.alpha is not None:
            nn.init.constant_(self.alpha.bias, 6.0)          # Initially retain state.

    @staticmethod
    def unit(k):
        """Keys as the delta rule needs them for exact overwrites (k^T k = 1); softmax addresses sum to 1 instead."""
        return k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def init_state(self, B, device, dtype=torch.float32):
        return torch.zeros(B, self.cfg.n_slots, self.cfg.d_val, device=device, dtype=dtype)

    def _addr(self, src):
        if self.cfg.cont_keys:
            return F.normalize(self.addr(src), dim=-1)
        z = F.normalize(self.addr(src), dim=-1) * self.addr_scale.exp() * math.sqrt(self.cfg.n_slots)
        return _address(z, self.cfg.hard_addressing)

    def forward(self, h, S, write_mask=None, cache=None):
        B, T, _ = h.shape
        cw, cr = (cache or {}).get("w"), (cache or {}).get("r")
        src_w, cw = self.ptr_w(h, cw, write_mask)
        src_r, cr = self.ptr_r(h, cr, write_mask)
        kw, kr = self._addr(src_w), self._addr(src_r)
        beta = torch.sigmoid(self.beta(h))                   # (B,T,1)
        if write_mask is not None:                           # padding must never write
            beta = beta * write_mask[..., None].to(beta.dtype)
        vh = self.v(h)
        alpha = torch.sigmoid(self.alpha(h)) if self.alpha is not None else None
        reads = []
        uw, ur = (self.unit(kw), self.unit(kr)) if self.cfg.l2_keys else (kw, kr)
        if alpha is None and kernels.delta_rule_ok(h, ur.shape[-1], vh.shape[-1], B, T, torch.is_grad_enabled()):
            # Fused implementation of the reference recurrence below.
            reads, S = kernels.delta_rule(ur, uw, vh, beta[..., 0], self.v_r.weight, S)
            self.last_aux = dict(ptr_w=self.ptr_w.last_logits, ptr_r=self.ptr_r.last_logits, beta=beta[..., 0],
                                 kw=kw, kr=kr)
            return reads, S, {"w": cw, "r": cr}
        for t in range(T):
            r = torch.einsum("bn,bnd->bd", ur[:, t], S)
            v = vh[:, t] + self.v_r(r)
            if alpha is not None:
                S = alpha[:, t, :, None] * S
            pred = torch.einsum("bn,bnd->bd", uw[:, t], S)
            S = S + beta[:, t, :, None] * uw[:, t, :, None] * (v - pred)[:, None, :]
            reads.append(r)
        self.last_aux = dict(ptr_w=self.ptr_w.last_logits, ptr_r=self.ptr_r.last_logits, beta=beta[..., 0],
                             kw=kw, kr=kr)
        return torch.stack(reads, 1), S, {"w": cw, "r": cr}


class MemoryBlock(nn.Module):
    def __init__(self, d_model: int, cfg: MemConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = RMSNorm(d_model)
        self.kg = KGReadHead(d_model, d_model, cfg) if cfg.use_kg else None
        self.pad = DeltaScratchpad(d_model, cfg) if cfg.use_scratchpad else None
        self.out_kg = nn.Linear(cfg.d_val, d_model, bias=False)
        self.out_pad = nn.Linear(cfg.d_val, d_model, bias=False)
        self.pad_readout = nn.Linear(cfg.d_val, d_model, bias=False)   # direct path: read -> LM unembedding
        # Stage-1 addresses can already be sharp; normalize reads before fitting the readout.
        self.read_norm = RMSNorm(cfg.d_val) if cfg.read_norm else None
        for lin in (self.out_kg, self.out_pad, self.pad_readout):
            nn.init.zeros_(lin.weight)                                  # Zero residual and logit contributions at initialization.
        self.gate = nn.Linear(d_model, 2)
        self.ro_gate = nn.Linear(d_model, 1) if cfg.readout_gate else None
        if self.ro_gate is not None:                                    # Preserve readout scale on warm starts.
            nn.init.zeros_(self.ro_gate.weight); nn.init.constant_(self.ro_gate.bias, 4.0)
        self.copy_gate = nn.Linear(d_model + 2, 1)
        nn.init.constant_(self.copy_gate.bias, -2.0)
        self.end_gate = nn.Linear(d_model, 1) if cfg.end_gate else None
        if self.end_gate is not None:                                   # sigmoid(-4) is about 0.018.
            nn.init.zeros_(self.end_gate.weight); nn.init.constant_(self.end_gate.bias, -4.0)
        self.end_id = -1                                                # the line-end token; MemoryLM sets it
        # Per-channel calibration prevents large shared channels from dominating key cosine.
        # Defaults are identity; MemoryLM.calibrate updates both queries and subsequent keys.
        self.register_buffer("in_shift", torch.zeros(d_model))
        self.register_buffer("in_scale", torch.ones(d_model))
        # Include persistent scratchpad state in state_dict checkpoints.
        if self.pad is not None:
            self.register_buffer("persistent_state", self.pad.init_state(1, "cpu"), persistent=True)

    def standardize(self, h):
        return (h.float() - self.in_shift) / self.in_scale

    def forward(self, h, ctx):
        B, T, D = h.shape
        x = self.norm(self.standardize(h))
        gates = torch.sigmoid(self.gate(x))
        delta = torch.zeros_like(x)
        if self.kg is not None and ctx.graph is not None:
            value, ctx.cand_ids, ctx.cand_w, ctx.kg_aux = self.kg(x, ctx.graph, ctx)
            if ctx.cand_hist_ids is None:                               # per-position history for carry-over
                ctx.cand_hist_ids, ctx.cand_hist_w = ctx.cand_ids, ctx.cand_w
            else:
                ctx.cand_hist_ids = torch.cat([ctx.cand_hist_ids, ctx.cand_ids], 1)
                ctx.cand_hist_w = torch.cat([ctx.cand_hist_w, ctx.cand_w.detach()], 1)
            delta = delta + gates[..., :1] * self.out_kg(value)
        if self.pad is not None:
            S0 = ctx.state if ctx.state is not None else self.pad.init_state(B, x.device)
            reads, S, ctx.pad_cache = self.pad(x, S0, ctx.write_mask, ctx.pad_cache)
            ctx.state = S if self.training else S.detach()
            reads = self.read_input(reads)
            ctx.pad_logit_dir = self.pad_readout(reads)                 # (B,T,D), mapped through lm_head later
            g_pad = gates[..., 1:]
            if self.ro_gate is not None:
                ctx.ro_gate = torch.sigmoid(self.ro_gate(x))
                ctx.pad_logit_dir = ctx.pad_logit_dir * ctx.ro_gate
                if self.cfg.gate_pad_value:
                    g_pad = g_pad * ctx.ro_gate
            delta = delta + g_pad * self.out_pad(reads)
        ctx.copy_x, ctx.gates = x, gates
        return h + delta.to(h.dtype)

    def read_input(self, reads):
        """Normalize reads and attenuate those below cfg.read_gate in norm.

        Without attenuation, RMSNorm amplifies near-empty reads into a logit bias.
        """
        if self.read_norm is None:
            return reads
        out = self.read_norm(reads)
        if self.cfg.read_gate > 0:
            out = out * (reads.norm(dim=-1, keepdim=True) / self.cfg.read_gate).clamp(max=1)
        return out


# Copy head: match generated prefixes against retrieved entity names.
def pointer_next_tokens(ids, cand_tok, cand_len):
    """Match candidate-name prefixes against the token history.

    ids is (B, Th); cand_tok is (B, Tn, C, L), padded with -1; cand_len is
    (B, Tn, C). At each of the last Tn positions, propose the token after the
    longest proper prefix. done marks names just emitted in full. Uses the CUDA
    kernel when supported.
    """
    if kernels.matching_ok(ids):
        return kernels.pointer_match(ids, cand_tok, cand_len)
    B, Th = ids.shape
    Tn, L = cand_tok.shape[1], cand_tok.shape[-1]          # Tn <= Th: candidates for the last Tn positions
    ext = torch.cat([torch.full((B, L), -2, dtype=ids.dtype, device=ids.device), ids], 1)
    pos = torch.arange(Th - Tn, Th, device=ids.device)
    back = pos[:, None] + L - torch.arange(L, device=ids.device)[None, :]
    hist = ext[:, back]                                   # (B,Tn,L): hist[...,m] = token m steps back
    best = torch.zeros(cand_tok.shape[:-1], dtype=torch.long, device=ids.device)
    done = torch.zeros_like(best, dtype=torch.bool)
    for j in range(1, L + 1):
        seg = hist[..., :j].flip(-1)                       # ids[i-j+1 .. i]
        ok = (cand_tok[..., :j] == seg[:, :, None, :]).all(-1)
        best = torch.where(ok & (j < cand_len), torch.full_like(best, j), best)
        done = done | (ok & (j == cand_len))
    nxt = cand_tok.gather(-1, best.clamp(max=L - 1)[..., None]).squeeze(-1)
    return nxt, done, (best > 0)


def carried_candidates(ids, hist_ids, hist_w, name_tok, name_len, Tn, share: float = 0.0, top: float = 0.0):
    """Continue names using candidates retrieved before their first emitted token.

    For each lag j, match the j emitted tokens against that position's candidate
    prefixes and propose token j. Eligibility uses top, then share, then the
    absolute 0.05 threshold. Returns proposals, weights, a partial-match mask,
    and a completion mask. With share or top enabled, completion must outweigh
    longer candidates with the completed name as a prefix.
    """
    if kernels.matching_ok(ids) and name_tok.shape[1] > 1:
        return kernels.carried_match(ids, hist_ids, hist_w, name_tok, name_len, Tn, share, top)
    B, Th = ids.shape
    L = name_tok.shape[1]
    ext = torch.cat([torch.full((B, L), -2, dtype=ids.dtype, device=ids.device), ids], 1)
    pos = torch.arange(Th - Tn, Th, device=ids.device)
    back = pos[:, None] + L - torch.arange(L, device=ids.device)[None, :]
    hist = ext[:, back]                                                # (B,Tn,L) tokens m steps back
    props, ws = [], []
    any_m = torch.zeros(B, Tn, dtype=torch.bool, device=ids.device)
    completed = torch.zeros_like(any_m)
    for j in range(1, L + 1):
        src = pos - j
        ok_src = (src >= 0)[None, :, None]
        cid = hist_ids[:, src.clamp(min=0)]                            # (B,Tn,C)
        cw = hist_w[:, src.clamp(min=0)] * ok_src
        tok, ln = name_tok[cid], name_len[cid]
        emitted = hist[..., :j].flip(-1)                               # tokens src+1 .. i
        if top:
            strong = (cw >= top * cw.max(-1, keepdim=True).values) & (cw > 0)
        else:
            strong = cw > share * cw.sum(-1, keepdim=True) if share else cw > 0.05
        full = (tok[..., :j] == emitted[:, :, None, :]).all(-1) & ok_src & strong
        if share or top:       # " 10" is finished only if no heavier retrieved name ("100") is still being spelled past it
            completed |= (cw * (full & (j == ln))).sum(-1) > (cw * (full & (j < ln))).sum(-1)
        else:
            completed |= (full & (j == ln)).any(-1)
        if j < L:
            match = full & (j < ln)
            props.append(tok[..., j].clamp(min=0)); ws.append(cw * match); any_m |= match.any(-1)
    return torch.cat(props, -1), torch.cat(ws, -1), any_m, completed


def copy_inputs(block, ctx, ids):
    g = ctx.graph
    cand_tok, cand_len = g.name_tok[ctx.cand_ids], g.name_len[ctx.cand_ids]
    nxt, done, partial = pointer_next_tokens(ids, cand_tok, cand_len)
    w = ctx.cand_w * (~done)
    carry_any, completed, c_w = torch.zeros_like(partial[..., 0]), None, None
    if getattr(block, "carry", True) and ctx.cand_hist_ids is not None and g.name_tok.shape[1] > 1:
        c_nxt, c_w, carry_any, completed = carried_candidates(ids, ctx.cand_hist_ids, ctx.cand_hist_w,
                                                              g.name_tok, g.name_len, nxt.shape[1],
                                                              block.cfg.carry_share, block.cfg.carry_top)
        if block.cfg.carry_rel:            # Scale fresh candidates relative to carried mass.
            scale = block.cfg.carry_fresh * c_w.sum(-1, keepdim=True) / w.sum(-1, keepdim=True).clamp(min=1e-12)
            w = torch.where(carry_any[..., None], w * scale, w)
        elif block.cfg.carry_fresh != 1.0:                             # Downweight fresh candidates during a carried name.
            w = w * torch.where(carry_any, block.cfg.carry_fresh, 1.0)[..., None]
        nxt, w = torch.cat([nxt, c_nxt], -1), torch.cat([w, c_w], -1)
        if getattr(block, "stop_after_complete", True):
            # Suppress further name proposals once a retrieved name has completed.
            w = w * (~completed)[..., None]
    mass = w.sum(-1)                                                   # (B,T)
    feats = torch.stack([mass, ((partial & ~done).any(-1) | carry_any).float()], -1)
    lam = torch.sigmoid(block.copy_gate(torch.cat([ctx.copy_x, feats], -1))).squeeze(-1)
    lam = lam * (mass > 1e-6)
    if block.cfg.carry_commit and c_w is not None:
        lam = _commit(ctx, ids, lam, c_w, g.name_tok.shape[1], mass)
    if getattr(block, "end_gate", None) is not None and completed is not None:
        # Completed names can propose a newline through the end gate.
        nxt = torch.cat([nxt, torch.full_like(nxt[..., :1], block.end_id)], -1)
        w = torch.cat([w, completed.to(w.dtype)[..., None]], -1)
        mass = w.sum(-1)
        lam = torch.where(completed, torch.sigmoid(block.end_gate(ctx.copy_x)).squeeze(-1).to(lam.dtype), lam)
    return nxt, w, mass, lam


def _commit(ctx, ids, lam, c_w, L, mass):
    """Floor each carried-name copy gate at its value where the name started.

    ctx.lam_hist stores raw gate values; incremental calls append to it.
    """
    B, Th = ids.shape
    Tn = lam.shape[1]
    prev = getattr(ctx, "lam_hist", None)
    if Tn == Th or prev is None:
        hist = torch.cat([lam.new_zeros(B, Th - Tn), lam], 1)
    else:
        hist = torch.cat([prev[:, :Th - Tn], lam.new_zeros(B, max(0, Th - Tn - prev.shape[1])), lam], 1)
    ctx.lam_hist = hist.detach()
    pos = torch.arange(Th - Tn, Th, device=ids.device)
    src = pos[:, None] - torch.arange(1, L, device=ids.device)[None, :]                # (Tn, L-1) start of a lag-j name
    lam_src = hist.detach()[:, src.clamp(min=0)] * (src >= 0)                           # (B, Tn, L-1)
    carried = (c_w.reshape(B, Tn, L - 1, -1) > 0).any(-1)                             # a name started there is carried
    return torch.maximum(lam, (lam_src * carried).amax(-1)) * (mass > 1e-6)


def copy_logprobs_for_targets(block, ctx, ids, lm_logits, targets, keep=None):
    """log p(target) under (1-λ) p_LM + λ p_ptr. Without keep: every position, lm_logits (B,T,V).
    With keep (1-D positions): lm_logits/targets are (B,K,·) for just those positions; the pointer side is
    cheap (no vocab dimension) so it is computed everywhere and then indexed."""
    lm_lp = lm_logits.float().log_softmax(-1).gather(-1, targets.clamp(min=0)[..., None]).squeeze(-1)
    if ctx.graph is None or ctx.cand_ids is None:
        return lm_lp
    outs = copy_inputs(block, ctx, ids)
    if isinstance(keep, tuple):                      # (rows, cols): one entry per answer token, (N, ·)
        outs = [t[keep[0], keep[1]] for t in outs]
    elif keep is not None:                           # 1-D columns shared by all rows, (B, K, ·)
        outs = [t[:, keep] for t in outs]
    nxt, w, mass, lam = (t.to(lm_lp.device) for t in outs)                       # device_map-safe
    ptr = (w * (nxt == targets[..., None])).sum(-1) / mass.clamp(min=1e-6)
    lam = lam.clamp(0, 1 - 1e-4)
    return torch.logaddexp(torch.log1p(-lam) + lm_lp, torch.log(lam + 1e-12) + torch.log(ptr + 1e-12))


def copy_mixture_probs(block, ctx, ids, lm_logits):
    """Return the mixed next-token distribution at the last position."""
    p = lm_logits[:, -1].float().softmax(-1)
    if ctx.graph is None or ctx.cand_ids is None:
        return p
    nxt, w, mass, lam = (t[:, -1].to(p.device) for t in copy_inputs(block, ctx, ids))
    ptr = torch.zeros_like(p).scatter_add_(-1, nxt, w) / mass.clamp(min=1e-6)[:, None]
    return (1 - lam[:, None]) * p + lam[:, None] * ptr
