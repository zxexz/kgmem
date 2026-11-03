"""Time the statistics-store hook's eager and CUDA-graph read paths.

  python -m experiments.statistics.hook_latency --model MODEL --backbone-loader backbone_loader.py --adapter owid_v2_null_clean.pt --store store.json --licence clean --questions held_keys.gated.jsonl --n 300 --out hook_latency.json

Input is [bos] + "Q: <question>\nA:" under the external attention loader's policy B. CUDA-synchronized
timings cover layers 0..L, the block and decoding. One graph per token length
captures the partial backbone forward after warm-up; static position buffers retain
the attention loader's document-check cache. Compare captured states, values, choices and abstain
scores with the eager path on each question. The eager readout timing excludes
option selection; graph_total includes it."""
from kgmem.paths import ensure_parent

import argparse
import importlib.util
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.statistics import owid_hook
from experiments.statistics.owid_mem import choose, decode, prompt


class GraphForward:
    """Cache one partial-forward CUDA graph per token length.

    Static buffers retain the attention loader's document-check cache after warm-up; captured
    graphs share one memory pool."""

    def __init__(self, mlm):
        self.mlm, self.graphs, self.pool = mlm, {}, torch.cuda.graph_pool_handle()
        lm = mlm.lm.model
        self.text = getattr(lm, "language_model", None) or lm

    def partial(self, ids, pos):
        """Run layers 0..L unwrapped, without capture hooks or an attention mask."""
        from collections import UserDict
        from kgmem.memlm import MemLayer
        t = self.text
        h = t.embed_tokens(ids)
        pe = {lt: t.rotary_emb(h, pos, lt) for lt in t.unique_layer_types}
        shared = UserDict()
        for i in range(self.mlm.layer_idx + 1):
            layer = t.layers[i]
            layer = layer.layer if isinstance(layer, MemLayer) else layer
            out = layer(h, None, shared_kv_states=shared, position_embeddings=pe[t.config.layer_types[i]],
                        attention_mask=None, position_ids=pos, past_key_values=None)
            h = out[0] if isinstance(out, tuple) else out
        return h.clone()                                # Keep a distinct captured output buffer.

    def __call__(self, ids):
        L = ids.shape[1]
        g = self.graphs.get(L)
        if g is None:
            buf = ids.clone()
            pos = torch.arange(L, device=ids.device)[None]      # Replays retain this buffer address.
            for _ in range(2):                              # Warm up allocations, kernels and the attention loader's document cache.
                self.partial(buf, pos)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = self.partial(buf, pos)
            g = self.graphs[L] = (graph, buf, out, pos)
        graph, buf, out, _ = g
        buf.copy_(ids)
        graph.replay()
        return out


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, 1000 * (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--model", "--adapter", "--store", "--questions", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--licence", default="all")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--backbone-loader", dest="backbone_loader", required=True,
                    help='Python loader exporting register() -> attention name and require_policy(text_config, policy); uses CUDA bf16')
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("kgmem_backbone_loader", a.backbone_loader)
    fm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fm)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation=fm.register()).eval()
    fm.require_policy(model.config.get_text_config(), "B")
    tok = AutoTokenizer.from_pretrained(a.model)
    owid_hook.load(a.adapter, a.store, a.licence)
    mlm = owid_hook.attach(model, tok)
    store, k = owid_hook._S["store"], owid_hook._S["k"]
    rows = [json.loads(l) for l in open(a.questions)][:a.n]
    enc = lambda q: torch.tensor([mlm.bos + tok(prompt(q), add_special_tokens=False).input_ids], device=mlm.device)
    gf = GraphForward(mlm)
    res = {"n": len(rows), "lengths": sorted({enc(r["question"]).shape[1] for r in rows})}
    with torch.no_grad():
        for r in rows[:5]:
            owid_hook.read(model, tok, r["question"], r["options"])
        t = {x: [] for x in ("forward", "block", "readout", "eager_total", "graph_forward", "graph_total")}
        same_h = same_read = 0
        max_dh = 0.0
        ids0 = enc(rows[0]["question"])                     # Compare direct and hooked partial forwards.
        h0 = mlm.hidden_at_layer(ids0, torch.ones_like(ids0))
        p0 = gf.partial(ids0, torch.arange(ids0.shape[1], device=ids0.device)[None])
        res["direct_path_max_abs_diff"] = float((p0.float() - h0.float()).abs().max())
        for r in rows:
            ids = enc(r["question"])
            h, ms = timed(lambda: mlm.hidden_at_layer(ids, torch.ones_like(ids)))
            t["forward"].append(ms)
            _, ms = timed(lambda: mlm.read(h, ids, k=k))
            t["block"].append(ms)
            d, ms = timed(lambda: dict(decode(mlm, store, k), choice=None))
            d["choice"] = choose(d["values"], r["options"]) if d["values"] else None
            t["readout"].append(ms)
            t["eager_total"].append(t["forward"][-1] + t["block"][-1] + t["readout"][-1])
            if ids.shape[1] not in gf.graphs:               # Exclude capture cost from replay timing.
                gf(ids)
            hg, ms = timed(lambda: gf(ids))
            t["graph_forward"].append(ms)
            dh = float((hg.float() - h.float()).abs().max())
            max_dh = max(max_dh, dh)
            same_h += dh == 0.0

            def graph_read():
                mlm.read(hg, ids, k=k)
                dd = decode(mlm, store, k)
                dd["choice"] = choose(dd["values"], r["options"]) if dd["values"] else None
                return dd
            dg, ms = timed(graph_read)
            t["graph_total"].append(t["graph_forward"][-1] + ms)
            same_read += (dg["choice"] == d["choice"] and abs(dg["abstain"] - d["abstain"]) < 1e-3
                          and [v["name"] for v in dg["values"]] == [v["name"] for v in d["values"]])
    res.update({f"{x}_ms_median": round(statistics.median(v), 2) for x, v in t.items()})
    res.update({f"{x}_ms_p90": round(sorted(v)[int(0.9 * len(v))], 2) for x, v in t.items()})
    res.update(graphs_captured=len(gf.graphs), states_bit_identical=same_h, max_abs_state_diff=max_dh,
               reads_identical=same_read)
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(ensure_parent(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
