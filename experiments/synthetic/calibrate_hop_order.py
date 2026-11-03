"""Estimate a hop_order threshold from relation words and function words.

  python -m experiments.synthetic.calibrate_hop_order checkpoints/smollm2-v9.pt
  python -m experiments.synthetic.calibrate_hop_order runs/synthetic/g4.pt --model google/gemma-4-26B-A4B-it --dtype bfloat16

The score is the cosine between the normalized block input and the nearest
relation key, zeroed when an entity key is closer. The threshold is the midpoint
between function words' maximum and relation words' 10th percentile, if those
ranges separate. Questions use held-out graph names and strandedcpu relations.
The probe applies block.norm to raw attach-layer states, without the saved input
standardization. Its threshold is applicable only when that calibration is the identity.

v9 gives 0.39 (function maximum 0.367, relation 10th percentile 0.413); its saved
setting is 0.4. Retraining the v7 graph path raised "?" and " of" to 0.48 and 0.58,
which caused the old threshold to split one-hop questions.
"""
import argparse
import json
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kgmem import MemConfig, MemoryLM
from kgmem.strandedcpu import strandedcpu_graph
from kgmem.tasks import World

# Exclude "in": it occurs in relation names such as "lives in" and "located in".
FUNC = {"of", "the", "?", "'s", "a", "is", "what", "which", "who", "where", "does", "do", "name", "q", ":", "they",
        "people", "how", "many", "for"}
STRANDEDCPU_QS = ["What is the successor of the double of 7?", "What is the digit sum of the square of 12?",
           "What is the square of the predecessor of 9?", "What is the smallest prime factor of 91?"]


@torch.no_grad()
def measure(mlm, seeds=(12345, 1, 2), per_world=12, strandedcpu=True):
    tok, rows = mlm.tok, []
    lkg = strandedcpu_graph(range(200), ["successor", "predecessor", "double", "square", "digit sum",
                                  "smallest prime factor"])[0] if strandedcpu else None
    for seed in seeds:
        w, rng = World(random.Random(seed), names="eval"), random.Random(seed)
        qs = [(w.kg, w.sample_question(rng, p_two_hop=0.6)[0]) for _ in range(per_world)]
        qs += [(lkg, q) for q in STRANDEDCPU_QS] if strandedcpu else []
        for kg, q in qs:
            mlm.load_graph(kg)
            g = mlm.ctx.graph
            keys = torch.cat([g.rel_keys] + ([g.rel_keys_first] if g.rel_keys_first is not None else [])).float()
            ids = torch.tensor([mlm.bos + tok(f"Q: {q}\nA:", add_special_tokens=False).input_ids], device=mlm.device)
            h = mlm.hidden_at_layer(ids, torch.ones_like(ids))[0].float()
            xn = F.normalize(mlm.block.norm(h.to(mlm.device)), dim=-1)
            raw = (xn @ F.normalize(keys, dim=-1).T).max(-1).values
            ent = (xn @ F.normalize(g.ent_keys.float(), dim=-1).T).max(-1).values
            rel = torch.where(raw > ent, raw, torch.zeros_like(raw))
            words = {wd for r in kg.relations for wd in r.split()} - FUNC
            for t, i in enumerate(ids[0].tolist()[1:], 1):
                s = tok.decode([i]).strip()
                kind = "rel" if s in words else "func" if s.lower() in FUNC else None
                if kind:
                    rows.append((kind, s, float(rel[t])))
    relv = sorted(v for k, _, v in rows if k == "rel")
    fun = sorted(((v, s) for k, s, v in rows if k == "func"), reverse=True)
    p10 = relv[len(relv) // 10]
    return dict(relation_words=len(relv), rel_p10=round(p10, 3), rel_median=round(relv[len(relv) // 2], 3),
                function_words=len(fun), func_max=round(fun[0][0], 3), func_top=[(s, round(v, 3)) for v, s in fun[:5]],
                separable=p10 > fun[0][0], hop_order=round((p10 + fun[0][0]) / 2, 3) if p10 > fun[0][0] else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 device_map=dev if dev != "cpu" else None).eval()
    sd = torch.load(args.adapter, map_location="cpu", weights_only=True)
    mlm = MemoryLM(model, tok, MemConfig(**sd["cfg"]), layer_idx=sd["layer_idx"])
    mlm.load(args.adapter)
    mlm.block.eval()
    res = measure(mlm)
    print(json.dumps(res))
    if res["hop_order"] is None:
        print("relation words and function words overlap: hop_order would misfire on this adapter; leave it at 0")


if __name__ == "__main__":
    main()
