"""Compare store and native-engine answers on paired questions.

  python -m experiments.statistics.owid_pair --dump runs/owid_v2_null_dump --native native_v2/eng.jsonl

Rule 1 uses the store when abstain < tau. Rule 2 also requires native's top
probability < pi. Fit thresholds on dev and the even crc32(qid) half of
unknown_series, maximizing mean accuracy across those two sets. Apply them to
held_keys, held_series, fresh_wording and the odd unknown_series half."""
from kgmem.paths import ensure_parent

import argparse
import collections
import json
import os
import zlib

L = "ABCD"


def load(a):
    nat = {}
    for l in (l for path in a.native.split(",") for l in open(path)):
        r = json.loads(l)
        if r.get("ans") and r["ans"].get("q1"):
            q = r["ans"]["q1"]
            nat[r["id"].split("|", 1)[1]] = (q["choice"], r["expected"]["q1"], max(q["p"].values()))
    sets = {}
    for name in ("dev", "fresh_wording", "held_keys", "held_series", "unknown_series", "unknown_series_v2",
                 "unknown_series_v3"):
        path = f"{a.dump}_{name}.jsonl"
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        if a.questions:                                # Recompute choices using current numeric matching.
            from experiments.statistics.owid_mem import choose
            qf = os.path.join(a.questions, f"{name}.gated.jsonl")
            if not os.path.exists(qf):                 # Fully passed sets may have only a .clean file.
                qf = qf.replace(".gated.", ".clean.")
            opts = {}
            for q in map(json.loads, open(qf)):
                opts.setdefault(q["qid"], []).append(q.get("options"))
            for r in rows:
                o = opts.get(r["qid"], [None])
                r["choice"] = choose(r["values"], o[0]) if len(o) == 1 and o[0] else None
        seen = collections.Counter(r["qid"] for r in rows)        # Duplicate qids with different options cannot be paired.
        rows = [r for r in rows if r["qid"] in nat and seen[r["qid"]] == 1]
        if name.startswith("unknown_series"):
            half = lambda r: zlib.crc32(r["qid"].encode()) % 2
            tag = name[len("unknown_series"):]
            for part, h in (("fit", 0), ("report", 1)):
                sub = [r for r in rows if half(r) == h]
                if sub:
                    sets[f"unknown{tag}_{part}"] = sub
        else:
            if rows:
                sets[name] = rows
    return nat, sets


def used(rows, nat, tau, pi=1.01):
    return [r["choice"] is not None and r["abstain"] < tau and nat[r["qid"]][2] < pi for r in rows]


def acc(rows, nat, tau, pi=1.01):
    use = used(rows, nat, tau, pi)
    hit = sum((L[r["choice"]] if u else nat[r["qid"]][0]) == nat[r["qid"]][1] for r, u in zip(rows, use))
    return hit / max(1, len(rows)), sum(use) / max(1, len(rows))


def overrides(rows, nat, tau, pi=1.01, conf=0.9):
    """Measure disagreement with native at max P >= conf, and store accuracy there."""
    use = used(rows, nat, tau, pi)
    conf_rows = [(r, u) for r, u in zip(rows, use) if nat[r["qid"]][2] >= conf]
    dis = [r for r, u in conf_rows if u and L[r["choice"]] != nat[r["qid"]][0]]
    right = sum(L[r["choice"]] == nat[r["qid"]][1] for r in dis)
    return dict(native_confident=len(conf_rows), store_overrides=len(dis),
                store_right_on_overrides=round(100 * right / len(dis), 1) if dis else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--native", required=True, help="the engine's eng.jsonl files, comma-separated")
    ap.add_argument("--questions", default="", help="re-choose every item with owid_mem.choose from these sets' options")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    nat, sets = load(a)
    fit = [s for s in ("dev", "unknown_fit") if s in sets]          # Fit on v1 unknowns; reserve v2 for reporting.
    taus = [i / 100 for i in range(5, 101, 1)]
    pis = [i / 20 for i in range(4, 21)] + [1.01]
    score = lambda t, p=1.01: sum(acc(sets[s], nat, t, p)[0] for s in fit) / len(fit)
    tau = max(taus, key=score)
    tau2, pi2 = max(((t, p) for t in taus for p in pis), key=lambda x: score(*x))
    res = {"fit on": fit, "rule 1 tau": tau, "rule 2 (tau, pi)": [tau2, pi2]}
    for name, rows in sets.items():
        native = sum(nat[r["qid"]][0] == nat[r["qid"]][1] for r in rows) / len(rows)
        store = sum(r["choice"] is not None and L[r["choice"]] == nat[r["qid"]][1] for r in rows) / len(rows)
        a1, u1 = acc(rows, nat, tau)
        a2, u2 = acc(rows, nat, tau2, pi2)
        res[name] = dict(n=len(rows), native=round(100 * native, 2), store_alone=round(100 * store, 2),
                         rule1=round(100 * a1, 2), rule1_store_used=round(100 * u1, 1),
                         rule2=round(100 * a2, 2), rule2_store_used=round(100 * u2, 1),
                         rule1_vs_confident_native=overrides(rows, nat, tau),
                         role="fit" if name in fit else "report")
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(ensure_parent(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
