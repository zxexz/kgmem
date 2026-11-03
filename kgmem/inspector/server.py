"""Local inspector for kgmem adapters.

  python -m kgmem.inspector
  python -m kgmem.inspector --adapters checkpoints/smollm2-v9.pt 'runs/*.pt' --port 8765

Open http://127.0.0.1:8765. The server binds to localhost; adapters must match --model.
Forward hooks record retrieval, gates and scratchpad writes. The response includes
replayed scratchpad states and a replay_ok comparison with the block's final state."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kgmem.inspector.graphs import demo_graph
from kgmem import KnowledgeGraph, MemConfig, MemoryLM
from kgmem.memlm import find_decoder_layers
from kgmem.modules import _project_keys, carried_candidates, copy_inputs, copy_mixture_probs
from kgmem.tasks import World
from kgmem.paths import checkpoint_path
from kgmem import strandedcpu

STATIC = Path(__file__).resolve().parent / "static"
MAX_CHARS = 1500


def r4(x) -> float | None:
    x = float(x)
    return round(x, 4) if math.isfinite(x) else None


def rows(t: torch.Tensor) -> list:
    return [[r4(v) for v in row] for row in t.tolist()]


def sparse_top(t: torch.Tensor, k: int, floor: float = 1e-3) -> list:
    """Per row: [[index, weight], ...] for the k largest entries above floor."""
    v, i = t.topk(min(k, t.shape[-1]), -1)
    return [[[int(j), r4(w)] for w, j in zip(vr, ir) if w > floor] for vr, ir in zip(v.tolist(), i.tolist())]


def aggregate(ids: torch.Tensor, w: torch.Tensor, k: int, floor: float = 1e-4) -> list:
    """Sum weights per id (candidate lists repeat ids) and keep the k heaviest: [[id, weight], ...]."""
    acc: dict[int, float] = {}
    for i, x in zip(ids.tolist(), w.tolist()):
        acc[i] = acc.get(i, 0.0) + x
    best = sorted(((x, i) for i, x in acc.items() if x > floor), reverse=True)[:k]
    return [[i, r4(x)] for x, i in best]


class Inspector:
    def __init__(self, model_id: str, dtype: str, adapter_globs: list[str]):
        self.model_id, self.globs = model_id, adapter_globs
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map="auto" if torch.cuda.is_available() else None).eval()
        self.tc = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") \
            else self.model.config
        self.lock = threading.Lock()
        self.mlm: MemoryLM | None = None
        self.kg = demo_graph()
        self.graph_label = "hand-written demo graph"
        self.scorer = None                          # None selects ExactScorer.
        self._adapter_cache: dict[str, tuple[float, dict | None]] = {}
        # Buffers and write logs are retained per adapter.
        self._buffers: dict[str, tuple[torch.Tensor, list]] = {}
        self.write_log: list = []
        found = self.adapters()
        if not found:
            raise SystemExit(f"no usable adapter for {model_id} in {adapter_globs}")
        self.use_adapter(found[0]["path"])

    # adapters
    def _describe(self, path: str) -> dict | None:
        """Check adapter width and layer; cache the result by file mtime."""
        mtime = os.path.getmtime(path)
        hit = self._adapter_cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        info = None
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            ok = isinstance(sd, dict) and "block" in sd and "layer_idx" in sd \
                and sd["block"]["norm.w"].shape[0] == self.tc.hidden_size \
                and sd["layer_idx"] < self.tc.num_hidden_layers
            if ok:
                meta = sd.get("meta") or {}
                info = dict(path=path, name=os.path.basename(path), layer_idx=sd["layer_idx"],
                            eval=meta.get("eval"), stage=meta.get("stage"), has_meta=bool(meta))
        except Exception:                                    # Cache shards are not adapters.
            info = None
        self._adapter_cache[path] = (mtime, info)
        return info

    def adapters(self) -> list[dict]:
        paths = []
        for g in self.globs:
            paths += sorted(glob.glob(os.path.expanduser(g)))
        seen, out = set(), []
        for p in paths:
            p = os.path.abspath(p)
            if p not in seen and p.endswith(".pt") and os.path.isfile(p):
                seen.add(p)
                d = self._describe(p)
                if d:
                    out.append(d)
        return out

    def use_adapter(self, path: str):
        if not any(a["path"] == path for a in self.adapters()):
            raise ValueError(f"not an adapter for {self.model_id}: {path}")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if self.mlm is not None:                             # Save the buffer before unwrapping the previous adapter.
            if self.mlm.block.pad is not None:
                self._buffers[self.adapter] = (self.mlm.block.persistent_state.clone(), self.write_log)
            layers = find_decoder_layers(self.model)
            layers[self.mlm.layer_idx] = layers[self.mlm.layer_idx].layer
        self.mlm = MemoryLM(self.model, self.tok, MemConfig(**sd["cfg"]), layer_idx=sd["layer_idx"])
        self.mlm.load(path)
        self.mlm.block.eval()
        self.adapter, self.meta = path, sd.get("meta")
        self.write_log = []
        if path in self._buffers and self.mlm.block.pad is not None:
            buf, self.write_log = self._buffers[path]
            self.mlm.block.persistent_state.copy_(buf)
        self.mlm.load_graph(self.kg, scorer=self.scorer)

    # views that don't need a prompt
    def graph_json(self) -> dict:
        kg = self.kg
        sc = self.scorer
        computed = {r: sc.programs[r] for r in kg.relations if sc is not None and r in sc.programs}
        return dict(label=self.graph_label, entities=kg.entities, relations=kg.relations,
                    triples=[list(t) for t in kg.triples], computed=computed)

    def load(self, kg, label, scorer=None):
        self.kg, self.graph_label, self.scorer = kg, label, scorer
        self.mlm.load_graph(kg, scorer=scorer)

    def strandedcpu_op(self, body: dict) -> dict:
        """Trace a relation program on the interpreter or transformer backend."""
        rel, x = str(body["relation"]), int(body["x"])
        backend = body.get("backend", "interpreter")
        if rel not in strandedcpu.PROGRAMS:
            raise ValueError(f"No program for relation {rel!r}.")
        if backend not in ("interpreter", "transformer"):
            raise ValueError("backend must be interpreter or transformer")
        if not strandedcpu.DOMAINS.get(rel, lambda v: True)(x):
            raise ValueError(f"{rel}({x}) is outside this program's 16-bit domain.")
        # Compare transformer state updates with the interpreter trace.
        res = strandedcpu.run_relation(strandedcpu.PROGRAMS[rel], x, backend=backend, trace=True)
        if backend == "transformer":
            ref = strandedcpu.run_relation(strandedcpu.PROGRAMS[rel], x, backend="interpreter", trace=True)
            same = len(ref["trace"]) == len(res["trace"]) and all(
                (a["pc"], a["op"], a["writes"]) == (b["pc"], b["op"], b["writes"]) for a, b in zip(ref["trace"], res["trace"]))
            res["agrees"] = res["y"] == ref["y"] and same
        names = self._op_names()
        steps = [dict(pc=s.get("pc"), op=names.get(s.get("op"), str(s.get("op"))), b=s.get("b"), c=s.get("c"),
                      writes={str(k): list(v) for k, v in (s.get("writes") or {}).items()})
                 for s in res.pop("trace")[:400]]
        return dict(res, relation=rel, x=x, source=strandedcpu.source(strandedcpu.PROGRAMS[rel]), trace=steps)

    @staticmethod
    def _op_names() -> dict:
        try:
            from strandedcpu import transformer
            return {v: k[3:] for k, v in vars(transformer).items() if k.startswith("OP_") and isinstance(v, int)}
        except ImportError:
            return {}

    def arch_json(self) -> dict:
        blk = self.mlm.block

        def tree(mod, name, depth):
            kids = [tree(c, n, depth + 1) for n, c in mod.named_children()] if depth < 2 else []
            return dict(name=name, type=type(mod).__name__, params=sum(p.numel() for p in mod.parameters()),
                        children=kids)

        n = self.tc.num_hidden_layers
        types = list(getattr(self.tc, "layer_types", None) or ["full_attention"] * n)
        return dict(model=self.model_id, n_layers=n, hidden=self.tc.hidden_size, vocab=self.tc.vocab_size,
                    layer_idx=self.mlm.layer_idx, layer_types=types,
                    frozen_params=sum(p.numel() for p in self.model.parameters()) - sum(
                        p.numel() for p in blk.parameters()),
                    block=tree(blk, "MemoryBlock", 0), cfg=dict(self.mlm.cfg.__dict__),
                    adapter=self.adapter, meta=self.meta)

    @torch.no_grad()
    def decode_slots(self, S: torch.Tensor, k: int = 3, labels: list | None = None) -> list:
        """Decode nonempty rows through the readout and unembedding.

        With continuous keys, S contains reads along variable keys and labels
        names those variables; raw state rows are key dimensions."""
        blk = self.mlm.block
        if not S.shape[0]:                                    # No recorded variables under continuous addressing.
            return []
        norms = S.norm(dim=-1)
        keep = (norms > max(1e-4, 1e-3 * float(norms.max()))).nonzero().flatten().tolist()
        if not keep:
            return []
        r = blk.read_input(S[keep])
        W = self.mlm.lm.get_output_embeddings().weight
        bias = (blk.pad_readout(r).to(W.device, W.dtype) @ W.T).float()
        v, i = bias.topk(k, -1)
        rows_ = [dict(slot=s, norm=r4(norms[s]), tokens=[[self.tok.decode([j]), r4(x)] for x, j in zip(vr, ir)],
                      **({"label": labels[s]} if labels else {}))
                 for s, vr, ir in zip(keep, v.tolist(), i.tolist())]
        return sorted(rows_, key=lambda d: -d["norm"])

    @staticmethod
    def var_keys(writes) -> tuple[list, torch.Tensor | None]:
        """Return each written variable's latest continuous key from (name, key) writes."""
        last = {}
        for name, k in writes:
            last.pop(name.strip(), None); last[name.strip()] = k
        if not last:
            return [], None
        return list(last), torch.stack([torch.as_tensor(k, dtype=torch.float32) for k in last.values()])

    def memory_json(self) -> dict:
        blk = self.mlm.block
        if blk.pad is None:
            return dict(enabled=False)
        S = blk.persistent_state[0].float()
        if blk.cfg.cont_keys:                               # State rows are key dimensions; display reads along variable keys.
            names, K = self.var_keys((w["via"], w["kw"]) for w in self.write_log)
            R = K @ S if K is not None else S[:0]
            return dict(enabled=True, cont=True, n_slots=len(names), labels=names,
                        norms=[r4(x) for x in R.norm(dim=-1).tolist()], slots=self.decode_slots(R, labels=names),
                        log=self.write_log)
        return dict(enabled=True, n_slots=S.shape[0], norms=[r4(x) for x in S.norm(dim=-1).tolist()],
                    slots=self.decode_slots(S), log=self.write_log)

    def state(self) -> dict:
        return dict(adapters=self.adapters(), adapter=self.adapter, arch=self.arch_json(), graph=self.graph_json(),
                    memory=self.memory_json())

    # generation + trace
    @torch.no_grad()
    def generate_ids(self, ids, max_new: int, stop: str, state, use_memory: bool):
        """Greedy KV-cached decode with the copy mixture; return ids and final state."""
        mlm, ctx = self.mlm, self.mlm.ctx
        was = ctx.enabled
        ctx.enabled = use_memory
        try:
            ctx.clear(); ctx.aux_idx = None; ctx.write_mask = None; ctx.aux_mention = None
            ctx.state = None if state is None else state.clone()
            ctx.tok_ids = ids
            out = mlm.lm_step(keep=1, input_ids=ids, use_cache=True)
            gen, stopped = [], False
            for _ in range(max_new):
                p = copy_mixture_probs(mlm.block, ctx, ids, out.logits) if use_memory else \
                    out.logits[:, -1].float().softmax(-1)
                nxt = p.argmax(-1, keepdim=True)
                gen.append(int(nxt)); ids = torch.cat([ids, nxt], 1)
                if stop and stop in self.tok.decode(gen):
                    stopped = True
                    break
                ctx.tok_ids = ids
                out = mlm.lm_step(keep=1, input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)
            return ids, gen, ctx.state
        finally:
            ctx.enabled = was

    @torch.no_grad()
    def trace(self, ids, state, use_memory: bool) -> dict:
        """Record a teacher-forced forward over ids, including base-model comparisons."""
        mlm, blk, ctx, tok = self.mlm, self.mlm.block, self.mlm.ctx, self.tok
        T = ids.shape[1]
        # Run memory-off first: forward() resets the context consumed by the hooks.
        was = ctx.enabled
        ctx.enabled = False
        try:
            p_base = mlm.forward(ids, None)[0].float().softmax(-1)
        finally:
            ctx.enabled = was
        cap = {}
        hooks = [blk.register_forward_hook(lambda m, a, o: cap.__setitem__("block", (a, o)))]
        if blk.kg is not None:
            hooks.append(blk.kg.register_forward_hook(lambda m, a, o: cap.__setitem__("kg", (a, o))))
        if blk.pad is not None:
            hooks.append(blk.pad.register_forward_hook(lambda m, a, o: cap.__setitem__("pad", (a, o))))
        was = ctx.enabled
        ctx.enabled = use_memory
        try:
            logits = mlm.forward(ids, None, aux_idx=torch.arange(T, device=ids.device), state=state)[0].float()
        finally:
            for h in hooks:
                h.remove()
            ctx.enabled = was
        id_list = ids[0].tolist()
        res: dict = dict(T=T, ids=id_list, tokens=[tok.decode([i]) for i in id_list], use_memory=use_memory)

        # Next-token distributions:
        #   base   the model with the memory off
        #   inj    the frozen upper layers after the block has changed the residual stream (no readout bias yet)
        #   gen    + the scratchpad's direct readout bias
        #   fin    + copy-head mixture
        W = mlm.lm.get_output_embeddings().weight
        bias = None
        if use_memory and ctx.pad_logit_dir is not None:
            bias = (ctx.pad_logit_dir[0].to(W.device, W.dtype) @ W.T).float().to(logits.device)
        p_gen = logits.softmax(-1)
        p_raw = (logits - bias).softmax(-1) if bias is not None else p_gen
        p_fin, ptr, lam, mass, carry, done = p_gen, None, None, None, None, None
        g = ctx.graph
        if use_memory and g is not None and ctx.cand_ids is not None:
            nxt, w, mass, lam = (t[0].to(logits.device) for t in copy_inputs(blk, ctx, ids))
            ptr = torch.zeros_like(p_gen).scatter_add_(-1, nxt, w) / mass.clamp(min=1e-6)[:, None]
            p_fin = (1 - lam[:, None]) * p_gen + lam[:, None] * ptr
            if g.name_tok.shape[1] > 1:
                _, _, carry, done = carried_candidates(ids, ctx.cand_hist_ids, ctx.cand_hist_w, g.name_tok,
                                                       g.name_len, T, blk.cfg.carry_share)
                carry, done = carry[0].tolist(), done[0].tolist()
        stages = (p_base, p_raw, p_gen, p_fin)
        nexttok = []
        for t in range(T):
            pick: list[int] = []                                 # Retain each stage's top 3 and the following token.
            for p in stages:
                pick += [j for j in p[t].topk(3).indices.tolist() if j not in pick]
            actual = id_list[t + 1] if t + 1 < T else None
            if actual is not None and actual not in pick:
                pick.append(actual)
            cands = sorted(pick, key=lambda j: -float(p_fin[t, j]))[:10]
            e = dict(cands=[[tok.decode([j])] + [r4(p[t, j]) for p in stages] + [j] for j in cands], actual=actual)
            if ptr is not None:
                pv, pi = ptr[t].topk(5)
                e.update(lam=r4(lam[t]), mass=r4(mass[t]),
                         pointer=[[tok.decode([j]), r4(v)] for v, j in zip(pv.tolist(), pi.tolist()) if v > 1e-4])
                if carry is not None:
                    e.update(carry=bool(carry[t]), completed=bool(done[t]))
            if bias is not None:
                bv, bi = bias[t].topk(5)
                e["readout"] = [[tok.decode([j]), r4(v)] for v, j in zip(bv.tolist(), bi.tolist())]
            nexttok.append(e)
        res["next"] = nexttok
        res["p_actual"] = [r4(p_fin[t, id_list[t + 1]]) for t in range(T - 1)] + [None]
        res["p_actual_base"] = [r4(p_base[t, id_list[t + 1]]) for t in range(T - 1)] + [None]

        if not use_memory or "block" not in cap:
            return res

        # Gates and residual update norms
        (h_in, _), _ = cap["block"]
        x = ctx.copy_x[0]
        gates = torch.sigmoid(blk.gate(x))
        hn = h_in[0].float().norm(dim=-1).clamp(min=1e-6)
        lanes = dict(g_kg=[r4(v) for v in gates[:, 0].tolist()], g_pad=[r4(v) for v in gates[:, 1].tolist()])
        if blk.ro_gate is not None:
            lanes["g_ro"] = [r4(v) for v in torch.sigmoid(blk.ro_gate(x))[:, 0].tolist()]

        # Graph head
        if "kg" in cap:
            (_, _, _), (value, cand_ids, cand_w, aux) = cap["kg"]
            lanes["d_kg"] = [r4(v) for v in ((gates[:, :1] * blk.out_kg(value[0])).norm(dim=-1) / hn).tolist()]
            hops = blk.cfg.hops
            ci, cw = cand_ids[0], cand_w[0]
            kc = ci.shape[-1] // hops
            halt = aux["halt"].softmax(-1)
            ent = aux["ent"]
            link_ids, link_w = ctx.link_cache["ids"][0], ctx.link_cache["w"][0]
            # Top-k weights sum to one even for weak matches. Expose raw cosine and mention usage.
            el = blk.kg.ent
            q = torch.nn.functional.normalize(el.m(x), dim=-1)
            link_cos = (q[:, None, :] * _project_keys(el, g.ent_keys)[link_ids]).sum(-1)
            att_full = ent["att"].softmax(-1)
            res["mention_strength"] = [r4(v) for v in att_full.max(0).values.tolist()]
            k_rel = [min(lk.k, g.rel_keys.shape[0]) for lk in blk.kg.rel]
            kgt = []
            for t in range(T):
                hop_out = []
                for h in range(hops):
                    if aux.get("relm") is not None:              # Include the relation-mention mixture.
                        rv, ri = aux["relm"]["p"][h][t].topk(k_rel[h])
                        rw = rv / rv.sum().clamp(min=1e-9)
                    else:
                        rv, ri = aux["rel"][h][t].topk(k_rel[h])
                        rw = rv.softmax(-1)
                    sl = slice(h * kc, (h + 1) * kc)
                    tail_w = cw[t, sl] / halt[t, h].clamp(min=1e-9)
                    hop_out.append(dict(rels=[[int(j), r4(v)] for v, j in zip(rw.tolist(), ri.tolist())],
                                        tails=aggregate(ci[t, sl], tail_w, 8)))
                    rm = aux.get("relm")
                    if rm is not None:
                        hop_out[-1]["named"] = sparse_top(rm["att"][h][t:t + 1].softmax(-1), 2, 0.05)[0]
                        hop_out[-1]["trust"] = r4(rm["rho"][t, h])
                        if hop_out[-1]["named"]:
                            s0 = hop_out[-1]["named"][0][0]
                            hop_out[-1]["reads_as"] = sparse_top(rm["tok"][s0:s0 + 1].softmax(-1), 3, 0.01)[0]
                    if isinstance(self.scorer, strandedcpu.StrandedCPUScorer):        # Attach the cached program result for the top pair.
                        src = (aggregate(ent["q_ids"][t], ent["q_w"][t], 1) if h == 0 else hop_out[0]["tails"][:1])
                        run = self.scorer.cache.get((src[0][0], int(ri[0]))) if src else None
                        if run is not None:
                            hop_out[-1]["strandedcpu"] = dict(relation=g.kg.relations[int(ri[0])], x=g.kg.entities[src[0][0]],
                                                       y=run["y"], steps=run["steps"], in_graph=run["tail"] is not None,
                                                       fault=run.get("fault"))
                kgt.append(dict(link=[[int(j), r4(v), r4(c)] for v, j, c in
                                      zip(link_w[t].tolist(), link_ids[t].tolist(), link_cos[t].tolist())],
                                heads=aggregate(ent["q_ids"][t], ent["q_w"][t], 8),
                                hops=hop_out, halt=[r4(v) for v in halt[t].tolist()],
                                final=aggregate(ci[t], cw[t], 8)))
            res["kg"] = kgt
            res["mention_att"] = sparse_top(att_full, 6)
            lanes["halt2"] = [r4(v) for v in halt[:, -1].tolist()]

        # Scratchpad
        if "pad" in cap:
            (xp, S0, _wm, _c), (reads, S_fin, _) = cap["pad"]
            pad, la = blk.pad, blk.pad.last_aux
            g_pad = gates[:, 1:] * (torch.sigmoid(blk.ro_gate(x)) if blk.cfg.gate_pad_value else 1)
            lanes["d_pad"] = [r4(v) for v in ((g_pad * blk.out_pad(blk.read_input(reads)[0])).norm(dim=-1) / hn).tolist()]
            lanes["beta"] = [r4(v) for v in la["beta"][0].tolist()]
            # Replay per-position states and compare with the block's final state.
            S, vh = S0.float().clone(), pad.v(xp)
            alpha = torch.sigmoid(pad.alpha(xp)) if pad.alpha is not None else None
            norms, states = [], []
            uw = pad.unit(la["kw"]) if blk.cfg.l2_keys else la["kw"]
            for t in range(T):
                v = vh[:, t] + pad.v_r(reads[:, t])
                if alpha is not None:
                    S = alpha[:, t, :, None] * S
                pred = torch.einsum("bn,bnd->bd", uw[:, t], S)
                S = S + la["beta"][:, t, None, None] * uw[:, t, :, None] * (v - pred)[:, None, :]
                norms.append(S[0].norm(dim=-1)); states.append(S[0])
            kw, kr = la["kw"][0], la["kr"][0]
            ptr_w = sparse_top(la["ptr_w"][0].softmax(-1), 6)
            res["pad"] = dict(
                n_slots=kw.shape[-1], replay_ok=bool(torch.allclose(S, S_fin.float(), atol=1e-4)),
                slot_norms=rows(torch.stack(norms)), start_norms=[r4(v) for v in S0[0].float().norm(dim=-1).tolist()],
                kw=rows(kw), kr=rows(kr), ptr_w=ptr_w, ptr_r=sparse_top(la["ptr_r"][0].softmax(-1), 6),
                final_slots=self.decode_slots(S_fin[0].float()))
            if blk.cfg.cont_keys:   # Display projections onto variable keys, not raw state rows.
                strong = [t for t in range(T) if float(la["beta"][0, t]) > 0.5]
                toks = res["tokens"]
                own = [(toks[ptr_w[t][0][0]] if ptr_w[t] else toks[t], kw[t]) for t in strong]
                prior = [(w["via"], w["kw"]) for w in self.write_log] if bool(S0.abs().sum() > 0) else []
                names, K = self.var_keys(prior + own)
                if K is not None:
                    reads_ = torch.einsum("vn,tnd->tvd", K.to(S.device), torch.stack(states))  # (position, variable)
                    res["pad"].update(cont=True, n_slots=len(names), labels=names,
                                      slot_norms=rows(reads_.norm(dim=-1)),
                                      start_norms=[r4(v) for v in (K.to(S0.device) @ S0[0].float()).norm(dim=-1).tolist()],
                                      final_slots=self.decode_slots(K.to(S_fin.device) @ S_fin[0].float(), labels=names))
                else:
                    res["pad"].update(cont=True, n_slots=0, labels=[], slot_norms=[[] for _ in range(T)], start_norms=[],
                                      final_slots=[])
        res["lanes"] = lanes
        return res

    # API actions
    def run(self, body: dict) -> dict:
        t0 = time.time()
        mode = body.get("mode", "ask")
        if mode not in ("ask", "remember", "trace"):
            raise ValueError(f"unknown mode: {mode}")
        raw = str(body.get("text", ""))
        text = raw[:MAX_CHARS]                                 # Tracing retains (T, vocab) distributions.
        if not text.strip():
            raise ValueError("The prompt is empty.")
        use_memory = bool(body.get("use_memory", True))
        persist = bool(body.get("persist", False))
        max_new = max(1, min(int(body.get("max_new_tokens", 24)), 64))
        stop = "\n" if body.get("stop_at_newline", True) else ""
        blk = self.mlm.block
        has_pad = blk.pad is not None
        if mode == "remember" and not (has_pad and use_memory):
            raise ValueError("Remember writes the scratchpad, so it needs the memory on"
                             + ("" if has_pad else " and an adapter with a scratchpad") + ".")
        prompt = torch.tensor([self.mlm.bos + self.tok(text, add_special_tokens=False).input_ids],
                              device=self.mlm.device)
        S0 = blk.persistent_state.clone() if (has_pad and (persist or mode == "remember")) else None
        # Only Remember commits to the persistent buffer.
        out: dict = dict(mode=mode, n_prompt=prompt.shape[1], committed=False, truncated=len(raw) > MAX_CHARS,
                         n_chars=len(raw), max_chars=MAX_CHARS,
                         buffer_empty=bool(has_pad and float(blk.persistent_state.norm()) < 1e-6))
        if mode == "ask":
            ids, gen, _ = self.generate_ids(prompt, max_new, stop, S0, use_memory)
            out["answer"] = self.tok.decode(gen).split(stop)[0] if stop else self.tok.decode(gen)
            if body.get("compare_base", True):
                _, base, _ = self.generate_ids(prompt, max_new, stop, None, False)
                out["base_answer"] = self.tok.decode(base).split(stop)[0] if stop else self.tok.decode(base)
                if S0 is not None and use_memory:
                    _, nob, _ = self.generate_ids(prompt, max_new, stop, None, True)
                    out["no_buffer_answer"] = self.tok.decode(nob).split(stop)[0] if stop else self.tok.decode(nob)
        else:
            ids = prompt
        out["trace"] = self.trace(ids, S0, use_memory)
        if mode == "remember":
            blk.persistent_state.copy_(self.mlm.ctx.state[:1]); out["committed"] = True
            self.log_writes(out["trace"])
        out["memory"] = self.memory_json()
        out["ms"] = int(1000 * (time.time() - t0))
        return out

    def log_writes(self, tr: dict):
        """Record strong Remember writes for later read/write overlap comparisons."""
        pad = tr.get("pad")
        if not pad:
            return
        for t, b in enumerate(tr["lanes"]["beta"]):
            if b is not None and b > 0.5:
                src = pad["ptr_w"][t][0][0] if pad["ptr_w"][t] else t
                self.write_log.append(dict(token=tr["tokens"][t], via=tr["tokens"][src], beta=b, kw=pad["kw"][t]))

    def graph_op(self, body: dict) -> dict:
        op, kg = body.get("op"), self.kg
        if op == "demo":
            self.load(demo_graph(), "hand-written demo graph")
        elif op == "random":
            seed, names = int(body.get("seed", 0)), body.get("names", "eval")
            self.load(World(random.Random(seed), names=names if names in ("train", "eval") else None).kg,
                      f"random world (seed {seed}, {names} names)")
        elif op == "strandedcpu":
            kg, sc = strandedcpu.strandedcpu_graph(range(int(body.get("size", 200))), list(strandedcpu.PROGRAMS))
            self.load(kg, "strandedcpu arithmetic", sc)
        elif op in ("set", "add", "remove"):
            h, r, t = str(body["h"]), str(body["r"]), str(body["t"])
            if self.scorer is not None and r in getattr(self.scorer, "programs", {}):
                raise ValueError(f"{r!r} is computed by a C program; its edges are read-only.")
            for n, idx in ((h, kg.ent_index), (r, kg.rel_index), (t, kg.ent_index)):
                if n not in idx:
                    raise ValueError(f"unknown name: {n}")
            if op == "set":
                kg.set(h, r, t)
            elif op == "add":
                kg.add(h, r, t)
            else:
                trip = (kg.ent_index[h], kg.rel_index[r], kg.ent_index[t])
                if not any(tuple(x) == trip for x in kg.triples):
                    raise ValueError(f"There is no edge {h} —{r}→ {t} to remove.")
                kg.triples = [x for x in kg.triples if tuple(x) != trip]
                kg._version += 1                              # Invalidate the scorer cache.
        elif op in ("add_entity", "add_relation"):
            name = str(body.get("name", "")).strip()[:64]
            if not name:
                raise ValueError("Type a name first.")
            ents, rels = list(kg.entities), list(kg.relations)
            (ents if op == "add_entity" else rels).append(name)
            if len(set(ents)) != len(ents) or len(set(rels)) != len(rels):
                raise ValueError(f"{name} is already in the graph.")
            if self.scorer is not None:
                raise ValueError("This graph uses numeric entities. Load another graph to add names.")
            self.load(KnowledgeGraph(ents, rels, list(kg.triples)), self.graph_label)   # Vocabulary changes require re-encoding.
        else:
            raise ValueError(f"unknown graph op: {op}")
        return dict(graph=self.graph_json())

    def memory_op(self, body: dict) -> dict:
        if body.get("op") == "reset" and self.mlm.block.pad is not None:
            self.mlm.reset_memory()
            self.write_log = []
        return dict(memory=self.memory_json())

    def set_adapter(self, body: dict) -> dict:
        self.use_adapter(str(body["path"]))
        return self.state()


class Handler(BaseHTTPRequestHandler):
    ins: Inspector
    files = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript"),
             "/style.css": ("style.css", "text/css")}

    def _send(self, code: int, body, ctype: str = "application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/state":
            with self.ins.lock:
                return self._send(200, self.ins.state())
        if path in self.files:
            name, ctype = self.files[path]
            return self._send(200, (STATIC / name).read_bytes(), ctype)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        routes = {"/api/run": "run", "/api/graph": "graph_op", "/api/memory": "memory_op",
                  "/api/adapter": "set_adapter", "/api/strandedcpu": "strandedcpu_op"}
        if self.path not in routes:
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
            with self.ins.lock:
                res = getattr(self.ins, routes[self.path])(body)
            self._send(200, res)
        except Exception as e:
            self._send(400, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--adapters", nargs="+", default=[checkpoint_path("smollm2-v9.pt")],
                    help="adapter files or globs, rescanned on page load")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if not any(Path(path).is_file() for pattern in args.adapters for path in glob.glob(os.path.expanduser(pattern))):
        ap.error("no adapter files found; pass --adapters PATH (checkpoints are included in the source checkout, not the installed package)")
    if args.threads:
        torch.set_num_threads(args.threads)
    Handler.ins = Inspector(args.model, args.dtype, args.adapters)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"kgmem inspector on http://127.0.0.1:{args.port}  (adapter: {Handler.ins.adapter})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
