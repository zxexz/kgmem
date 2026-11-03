"""Compare kgmem and RoG WebQSP predictions under two answer-matching rules.

  python -m experiments.qa.webqsp_compare --prep runs/webqsp/prep.pt --kgmem runs/webqsp/predictions.test.jsonl --rog rog_predictions.jsonl

rog_rule accepts a normalized gold answer as a substring anywhere in joined
output lines. strict requires the first nonempty answer line to equal a gold
answer after normalization. For example, rog_rule accepts "Jamaican English"
for gold "English"; strict does not.

--overlap supplies test IDs to score separately, for example questions whose
source IDs also appear in CWQ training data.
"""
import argparse
import json
import re
import string

import torch


def normalize(s):
    """RoG's normalize(): lower case, no punctuation, no articles, single spaces."""
    s = "".join(c for c in s.lower() if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def rog_hit(lines, gold):
    text = normalize(" ".join(lines))
    return any(normalize(g) in text for g in gold)


def strict(lines, gold):
    first = next((x for x in lines if x.strip()), "")
    return normalize(first) in {normalize(g) for g in gold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default='runs/webqsp/prep.pt')
    ap.add_argument("--kgmem", required=True, help="webqsp.py eval --dump X writes X.test.jsonl")
    ap.add_argument("--rog", default=None, help="RoG's results/KGQA/.../predictions.jsonl")
    ap.add_argument("--overlap", default=None, help='JSON list of test IDs to score separately from the remaining questions')
    a = ap.parse_args()
    P = torch.load(a.prep, weights_only=False)
    gold = {it["id"]: [P["entities"][x] for x in it["answers"]] for it in P["splits"]["test"]}
    overlap = set(json.load(open(a.overlap))) if a.overlap else None
    subsets = {"all": lambda i: True}
    if overlap is not None:
        subsets["CWQ-train overlap"] = lambda i: i in overlap
        subsets["no overlap"] = lambda i: i not in overlap

    def score(name, preds):                       # preds: id -> list of answer lines
        for sub, keep in subsets.items():
            r = [(p, gold[i]) for i, p in preds.items() if keep(i)]
            out[f"{name} [{sub}]"] = dict(n=len(r), strict=sum(strict(p, g) for p, g in r) / max(1, len(r)),
                                          rog_rule=sum(rog_hit(p, g) for p, g in r) / max(1, len(r)))
    out = {}
    score("kgmem", {r["id"]: ([r["pred"]] if r["pred"] else []) for r in map(json.loads, open(a.kgmem))})
    if a.rog:
        lines = lambda p: p if isinstance(p, list) else str(p).split("\n")
        rog = {x["id"]: lines(x["prediction"]) for x in map(json.loads, open(a.rog))}
        score("RoG", rog)
        out["RoG answers per output"] = sum(len([x for x in p if x.strip()]) for p in rog.values()) / len(rog)
    print(json.dumps({s: ({m: round(100 * v, 2) if isinstance(v, float) else v for m, v in d.items()}
                          if isinstance(d, dict) else round(d, 2)) for s, d in out.items()}, indent=1))


if __name__ == "__main__":
    main()
