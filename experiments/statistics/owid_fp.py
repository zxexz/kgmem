"""Measure hook use and accuracy on held-out numeric questions outside the store task.

  python -m experiments.statistics.owid_fp --model MODEL --backbone-loader backbone_loader.py --adapter owid_v2_null_clean.pt --store store.json --licence clean --files heldout_other.jsonl:gsm8k_train_4,gsm8k_train_10 heldout_svamp.jsonl libeval_clean.jsonl

Read DI rows: state.question, falling back to the first question's instructions;
criteria sorted by key supply options, and expected supplies the gold key.
Count a row as used when a choice exists and abstain < --tau."""
from kgmem.paths import ensure_parent

import argparse
import importlib.util
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.statistics import owid_hook
def rows_of(spec):
    path, _, tracks = spec.partition(":")
    want = set(tracks.split(",")) if tracks else None
    for l in open(path):
        r = json.loads(l)
        if want and (r.get("metadata") or {}).get("track") not in want:
            continue
        qk, q = next(iter(r["questions"].items()))
        keys = sorted(q.get("criteria") or {})
        if not keys:
            continue
        text = (r.get("state") or {}).get("question") or q["instructions"]
        gold = r["expected"][qk]
        yield dict(id=r["id"], track=(r.get("metadata") or {}).get("track") or path.split("/")[-1], question=text,
                   options=[str(q["criteria"][k]) for k in keys], gold=keys.index(gold) if gold in keys else None)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--model", "--adapter", "--store"):
        ap.add_argument(k, required=True)
    ap.add_argument("--licence", default="all")
    ap.add_argument("--tau", type=float, default=0.18)
    ap.add_argument("--files", nargs="+", required=True, help="path[:track,track]")
    ap.add_argument("--out", default="")
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
    owid_hook.attach(model, tok)
    res, fired = {}, []
    for f in a.files:
        for r in rows_of(f):
            out = owid_hook.read(model, tok, r["question"], r["options"])
            s = res.setdefault(r["track"], dict(n=0, used=0, right=0, abstain_min=1.0))
            s["n"] += 1
            s["abstain_min"] = min(s["abstain_min"], round(out["abstain"], 4))
            if out["choice"] is not None and out["abstain"] < a.tau:
                s["used"] += 1
                s["right"] += out["choice"] == r["gold"]
                fired.append(dict(id=r["id"], track=r["track"], question=r["question"][:200], abstain=out["abstain"],
                                  choice=out["choice"], gold=r["gold"], series=out["series"][:1]))
    tot = dict(n=sum(s["n"] for s in res.values()), used=sum(s["used"] for s in res.values()),
               right=sum(s["right"] for s in res.values()))
    for s in list(res.values()) + [tot]:
        s["used_pct"] = round(100 * s["used"] / max(1, s["n"]), 3)
        s["precision_when_used"] = round(100 * s["right"] / s["used"], 1) if s["used"] else None
    res["all"] = tot
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        json.dump(dict(result=res, fired=fired, tau=a.tau, args=vars(a)), open(ensure_parent(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
