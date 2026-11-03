"""Measure zero-shot alignment between question mentions and entity-name keys.

For synthetic worlds, compare each mention's last-token attach-layer state
with standalone name keys by cosine similarity. Report whether the nearest
key is the correct entity. --distractors adds names that never appear in the
questions, to measure linking as the key table grows.

  python -m experiments.synthetic.check_linking --model google/gemma-4-26B-A4B-it --dtype bfloat16
  python -m experiments.synthetic.check_linking --model google/gemma-4-26B-A4B-it --dtype bfloat16 --layer 23
"""
import argparse
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kgmem import MemConfig, MemoryLM
from kgmem.memlm import StopForward
from kgmem.tasks import SYL, World, kg_batch, make_name


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--worlds", type=int, default=6)
    ap.add_argument("--distractors", type=int, default=0, help="extra never-mentioned names in the key table")
    args = ap.parse_args()
    assert args.distractors < len(SYL) ** 2 + len(SYL) ** 3 - 1000, "more distractors than the name generator has"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 device_map="auto" if torch.cuda.is_available() else None).eval()
    mlm = MemoryLM(model, tok, MemConfig(), layer_idx=args.layer)
    rng = random.Random(12345)
    drng, used = random.Random(999), set()
    extra = [make_name(drng, used) for _ in range(args.distractors)]
    dkeys = mlm.encode_names(extra, pool="last").float() if extra else None
    ok = okk = n = total = 0
    k = mlm.cfg.k_ent
    for _ in range(args.worlds):
        w = World(rng, names="eval")
        keys = mlm.encode_names(w.kg.entities, pool="last").float()
        if dkeys is not None:                    # a distractor that is also a world name would be a forced tie
            keep = torch.tensor([d not in w.kg.ent_index for d in extra], device=dkeys.device)
            keys = torch.cat([keys, dkeys[keep]])
        b = kg_batch(mlm, w, rng, 8)
        mlm.ctx.capture, mlm.ctx.enabled = [], False
        try:
            model(input_ids=b.ids, attention_mask=b.attn, use_cache=False)
        except StopForward:
            pass
        h = mlm.ctx.capture[0].float(); mlm.ctx.capture, mlm.ctx.enabled = None, True
        T = b.ids.shape[1]
        sel = b.aux_mention >= 0
        rows = (b.aux_idx // T)[sel]
        x = h[rows, b.aux_mention[sel]]
        top = (F.normalize(x, dim=-1) @ F.normalize(keys.to(x.device), dim=-1).T).topk(k, -1).indices.cpu()
        hit = top == b.aux_ent[sel].cpu()[:, None]
        ok += hit[:, 0].sum().item(); okk += hit.any(-1).sum().item(); n += int(sel.sum()); total += sel.numel()
    print(f"layer {mlm.layer_idx}: zero-shot mention->key accuracy {ok / n:.3f}, top-{k} recall {okk / n:.3f} "
          f"(the linker keeps k_ent={k} candidates) over {n} mentions ({total - n} not string-matched, excluded), "
          f"{keys.shape[0]} keys (chance ~{1 / keys.shape[0]:.4f})")


if __name__ == "__main__":
    main()
