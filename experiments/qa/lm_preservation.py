"""Compare next-token NLL on WikiText-2 test paragraphs.

Measure the released backbone, the attached block with MQuAKE's graph, and
the fully fine-tuned semantic parser from mquake_baselines.py. For kgmem,
report both LM-logit NLL and the decoded mixture including the copy head.

  python -m experiments.qa.lm_preservation --prep runs/mquake/prep.pt --adapter checkpoints/mquake.pt --parser runs/mquake/baselines/parser/
"""
import argparse
import json
import math
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.qa import mquake
from kgmem.graph import KnowledgeGraph
from kgmem.paths import checkpoint_path
from kgmem.modules import copy_inputs


def paragraphs(n):
    from datasets import load_dataset
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    out = [t.strip() for t in rows if len(t.strip()) >= 200 and not t.strip().startswith("=")]
    return out[:n]


@torch.no_grad()
def lm_nll(model, tok, texts, bs=16):
    dev = next(model.parameters()).device
    tot = cnt = 0.0
    for i in range(0, len(texts), bs):
        bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        enc = [bos + tok(t, add_special_tokens=False).input_ids[:256] for t in texts[i:i + bs]]
        T = max(map(len, enc))
        ids = torch.tensor([s + [0] * (T - len(s)) for s in enc], device=dev)
        attn = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in enc], device=dev)
        lp = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float().log_softmax(-1)
        lp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
        m = attn[:, 1:] > 0
        tot += float(-lp[m].sum()); cnt += float(m.sum())
    return tot / cnt


@torch.no_grad()
def kgmem_nll(mlm, texts, on):
    """(LM logits, decoded) NLL, one paragraph at a time; decoded mixes in the copy head as generation does."""
    lm = dec = n = 0.0
    was = mlm.ctx.enabled
    mlm.ctx.enabled = on
    try:
        for t in texts:
            ids = torch.tensor([mlm.bos + mlm.tok(t, add_special_tokens=False).input_ids[:256]], device=mlm.device)
            lp = mlm.forward(ids).float().log_softmax(-1)
            dp = lp
            if on and mlm.ctx.graph is not None and mlm.ctx.cand_ids is not None:
                nxt, w, mass, lam = (x.to(lp.device) for x in copy_inputs(mlm.block, mlm.ctx, ids))
                ptr = torch.zeros_like(lp).scatter_add_(-1, nxt, w) / mass.clamp(min=1e-6)[..., None]
                dp = ((1 - lam[..., None]) * lp.exp() + lam[..., None] * ptr + 1e-12).log()
            tgt = ids[:, 1:, None]
            lm += float(-lp[:, :-1].gather(-1, tgt).sum())
            dec += float(-dp[:, :-1].gather(-1, tgt).sum())
            n += tgt.numel()
    finally:
        mlm.ctx.enabled = was
    return lm / n, dec / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default='runs/mquake/prep.pt')
    ap.add_argument("--adapter", default=checkpoint_path("mquake.pt"))
    ap.add_argument("--parser", default=None)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cfg", default="{}", help='JSON overrides applied after saved config and MQUAKE_INFERENCE')
    a = ap.parse_args()
    texts = paragraphs(a.n)
    P = torch.load(a.prep, weights_only=False)
    args = types.SimpleNamespace(model=a.model, dtype="float32", layer=None, adapter=a.adapter, cfg=a.cfg)
    mlm = mquake.load_adapter(args)
    kg = KnowledgeGraph(list(P["entities"]), list(P["relations"]), [])
    mquake.set_graph(kg, P["train_kb"])
    mlm.load_graph(kg)
    tok = mlm.tok
    res = {"SmolLM2-135M as released": kgmem_nll(mlm, texts, False)[0]}
    lm, dec = kgmem_nll(mlm, texts, True)
    res["kgmem attached, MQuAKE graph loaded: the LM's logits"] = lm
    res["kgmem attached, MQuAKE graph loaded: what it decodes (copy head included)"] = dec
    base = AutoModelForCausalLM.from_pretrained(a.model).to(mlm.device).eval()
    res["SmolLM2-135M as released (cross-check, scored like the parser)"] = lm_nll(base, tok, texts)
    del base
    if a.parser:
        ptok = AutoTokenizer.from_pretrained(a.parser)
        pm = AutoModelForCausalLM.from_pretrained(a.parser).to(mlm.device).eval()
        res["fine-tuned as the parser"] = lm_nll(pm, ptok, texts)
    out = {k: dict(nll=round(v, 4), ppl=round(math.exp(v), 2)) for k, v in res.items()}
    out["paragraphs"] = len(texts)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
