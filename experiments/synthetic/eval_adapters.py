"""Evaluate shipped adapters beyond train.evaluate's per-world scores.

  python -m experiments.synthetic.eval_adapters checkpoints/archive/kgmem_adapter_v8.pt checkpoints/smollm2-v9.pt
  python -m experiments.synthetic.eval_adapters checkpoints/smollm2-v9.pt --only recall,plain --threads 8

Sections:
  wide              Three seeds x eight held-out worlds, including answer stops.
  recall            Forty write/overwrite trials; each read starts a new session.
  plain             NLL change on 64 sampled completions and 20 assignment-like sentences.
  paths             NLL change with each block output path disabled in turn.
  strandedcpu       Six arithmetic relations, five compositions, 20 overwrite trials.
  strandedcpuwide   Every valid one-hop arithmetic query over 0..199 (911 queries).

NLL change is memory-on minus memory-off, in nats/token. Arithmetic sections need
strandedcpu.
"""
import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.synthetic import train
from kgmem import MemConfig, MemoryLM
from kgmem.memlm import find_decoder_layers
from kgmem.tasks import VALUES, World, make_name

SECTIONS = ("wide", "recall", "plain", "paths", "strandedcpu", "strandedcpuwide")

PROMPTS = ["My grandfather", "The museum", "On Tuesday", "Researchers in Japan", "The recipe calls for", "During the war,",
           "The new phone", "At the market,", "The weather", "Our garden", "The bridge", "After dinner,", "The teacher",
           "In the village", "The car", "The report says"]
ASSIGN = ["The walls are white and the floor is grey, so the room feels larger than it is.",
          "Set the oven to 200 degrees and leave the bread in for forty minutes.",
          "Her coat is red, her hat is blue, and her boots are the colour of mud.",
          "The password is long, but the hint is simple: it is the name of her first dog.",
          "In autumn the leaves turn gold and the river is slow and green.",
          "Set the alarm to six, then put the kettle on before anyone else is awake.",
          "His favourite food is rice, and his favourite drink is cold milk.",
          "The cat is black, the dog is brown, and the parrot is every colour at once.",
          "The sky was grey all morning, and by noon the sea was the same colour.",
          "Turn the dial to seven and wait for the light to go green.",
          "The tea is too hot, so let it sit for a minute before you drink it.",
          "The fox is quick, the hare is quicker, and the tortoise wins anyway.",
          "Change the setting to silent and the phone will stop buzzing in meetings.",
          "The door is locked, the key is under the mat, and the lights are off.",
          "Her name is Ada, and her brother's name is Tom.",
          "The answer is four, but the question was harder than it looked.",
          "The soup is salty, the bread is stale, and the wine is excellent.",
          "Put the gold coins in the left drawer and the silver ones in the right.",
          "The milk is in the fridge and the apples are in the bowl by the window.",
          "The bee is small, but the hive it belongs to is enormous."]
STRANDEDCPU_RELS = ["successor", "predecessor", "double", "square", "digit sum", "smallest prime factor"]
STRANDEDCPU_TWO = [("digit sum", "square"), ("successor", "double"), ("square", "predecessor"),
            ("double", "smallest prime factor"), ("smallest prime factor", "successor")]


def generated_texts(model, tok):
    """Sample 64 completions with seed 1 before attaching the adapter."""
    side, pad = tok.padding_side, tok.pad_token
    tok.padding_side, tok.pad_token = "left", tok.pad_token or tok.eos_token
    torch.manual_seed(1)
    out = []
    try:
        for i in range(0, 64, 16):
            enc = tok([PROMPTS[(i + j) % 16] for j in range(16)], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=40, do_sample=True, top_p=0.9, pad_token_id=tok.pad_token_id)
            out += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in g]
    finally:
        tok.padding_side, tok.pad_token = side, pad
    return [PROMPTS[i % 16] + t for i, t in enumerate(out)]


def harm(mlm, texts):
    return [train.text_nll(mlm, [t], True) - train.text_nll(mlm, [t], False) for t in texts]


def wide(mlm):
    res = [train.evaluate(mlm, s, n_worlds=8) for s in (12345, 1, 2)]
    return {k: round(sum(r[k] for r in res) / 3, 3) for k in res[0]}


def recall(mlm, n=40):
    """Check overwritten and untouched variables in separate sessions, without a graph.

    The first trial uses the demo names. Remaining trials use held-out names.
    no_overwrite omits the final write to A.
    """
    mlm.unload_graph()
    rng = random.Random(1)
    trials = [("kel", "vom", "fox", "tea", "rice")]
    while len(trials) < n:
        used = set()
        a, b = (make_name(rng, used, 1, 2, cap=False, names="eval") for _ in range(2))
        trials.append((a, b, *rng.sample(VALUES, 3)))

    def ask(state, v):
        mlm.block.persistent_state.copy_(state)
        return mlm.generate(f"Q: What is {v}?\nA:", persist=True, max_new_tokens=6).strip()

    ow = keep = keep2 = 0
    wrong = {"v1": 0, "v3": 0, "other": 0}
    for a, b, v1, v2, v3 in trials:
        mlm.reset_memory(); mlm.remember(f"Set {a} to {v1}. Set {b} to {v2}. Set {a} to {v3}.\n")
        s = mlm.block.persistent_state.clone()
        ow += ask(s, a) == v3
        got = ask(s, b)
        keep += got == v2
        if got != v2:
            wrong["v1" if got == v1 else "v3" if got == v3 else "other"] += 1
        mlm.reset_memory(); mlm.remember(f"Set {a} to {v1}. Set {b} to {v2}.\n")
        keep2 += ask(mlm.block.persistent_state.clone(), b) == v2
    mlm.reset_memory()
    return dict(overwritten=ow / n, untouched=keep / n, no_overwrite=keep2 / n, untouched_wrong=wrong)


def plain(mlm, gen):
    out = {}
    for name, texts in (("generated", gen), ("assignment-like", ASSIGN)):
        mlm.load_graph(World(random.Random(5), names="eval").kg)
        h = harm(mlm, texts)
        out[name] = dict(mean=round(sum(h) / len(h), 3), hurt=round(sum(x > 0.1 for x in h) / len(h), 2),
                         max=round(max(h), 2))
    return out


def paths(mlm):
    blk = mlm.block
    mlm.load_graph(World(random.Random(5), names="eval").kg)

    def mean_harm():                                         # per token over all 8 sentences, as train.evaluate
        return round(train.text_nll(mlm, train.PLAIN_EVAL, True) - train.text_nll(mlm, train.PLAIN_EVAL, False), 4)

    out = {"full": mean_harm()}
    for name, mod in (("no graph value", blk.out_kg), ("no scratchpad value", blk.out_pad),
                      ("no scratchpad readout", blk.pad_readout)):
        w = mod.weight.data.clone(); mod.weight.data.zero_()
        out[name] = mean_harm()
        mod.weight.data.copy_(w)
    b = blk.copy_gate.bias.data.clone(); blk.copy_gate.bias.data.fill_(-1e4)
    out["no copy head"] = mean_harm()
    blk.copy_gate.bias.data.copy_(b)
    return out


def strandedcpu(mlm):
    from kgmem.strandedcpu import strandedcpu_graph
    kg, sc = strandedcpu_graph(range(200), STRANDEDCPU_RELS)
    mlm.load_graph(kg, scorer=sc)
    rng, one = random.Random(0), {}
    for r in STRANDEDCPU_RELS:
        heads = [h for (h, rr, t) in kg.triples if kg.relations[rr] == r]
        ok = 0
        for i in range(10):
            x = kg.entities[rng.choice(heads)]
            q = ["What is the {r} of {x}?", "Name the {r} of {x}."][i % 2].format(r=r, x=x)
            ok += mlm.generate(f"Q: {q}\nA:").strip() == kg.tails(x, r)[0]
        one[r] = ok / 10
    two = {}
    for outer, inner in STRANDEDCPU_TWO:
        ok = n = 0
        for x in range(2, 200):
            mid = kg.tails(str(x), inner)
            end = kg.tails(mid[0], outer) if mid else []
            if not end:
                continue
            n += 1
            if n > 10:
                break
            ok += mlm.generate(f"Q: What is the {outer} of the {inner} of {x}?\nA:").strip() == end[0]
        two[f"{outer}∘{inner}"] = ok / min(n, 10)
    rng, ow, keep, N = random.Random(1), 0, 0, 20
    for _ in range(N):
        used = set()
        a, b = (make_name(rng, used, 1, 2, cap=False, names="eval") for _ in range(2))
        v1, v2, v3 = rng.sample(VALUES, 3)
        mlm.reset_memory(); mlm.remember(f"Set {a} to {v1}. Set {b} to {v2}. Set {a} to {v3}.\n")
        saved = mlm.block.persistent_state.clone()
        ow += mlm.generate(f"Q: What is {a}?\nA:", persist=True).strip() == v3
        mlm.block.persistent_state.copy_(saved)
        keep += mlm.generate(f"Q: What is {b}?\nA:", persist=True).strip() == v2
    mlm.reset_memory()
    return {"one-hop": round(sum(one.values()) / len(one), 2), "two-hop": round(sum(two.values()) / len(two), 2),
            "per relation": one, "per composition": two, "overwrite (20 trials)": dict(overwritten=ow / N, untouched=keep / N)}


def strandedcpuwide(mlm):
    from kgmem.strandedcpu import strandedcpu_graph
    kg, sc = strandedcpu_graph(range(200), STRANDEDCPU_RELS)
    mlm.load_graph(kg, scorer=sc)
    ok, n, miss = {}, {}, []
    for r in STRANDEDCPU_RELS:
        for x in range(200):
            gold = kg.tails(str(x), r)
            if not gold:
                continue
            got = mlm.generate(f"Q: What is the {r} of {x}?\nA:", max_new_tokens=8).strip()
            n[r] = n.get(r, 0) + 1
            ok[r] = ok.get(r, 0) + (got == gold[0])
            if got != gold[0]:
                miss.append(f"{r}({x})={gold[0]} got {got!r}")
    return {"one-hop": round(sum(ok.values()) / sum(n.values()), 3), "per relation": {r: f"{ok[r]}/{n[r]}" for r in n},
            "misses": miss[:30]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="+")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--only", default=",".join(SECTIONS), help=f"comma-separated subset of {', '.join(SECTIONS)}")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", help="bfloat16 for Gemma 4 26B (fp32 would be 104 GB)")
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    assert set(only) <= set(SECTIONS), f"unknown section in --only: {only}"
    if args.threads:
        torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 device_map=args.device if args.device != "cpu" else None).eval()
    gen = generated_texts(model, tok) if "plain" in only else None
    for path in args.adapters:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        mlm = MemoryLM(model, tok, MemConfig(**sd["cfg"]), layer_idx=sd["layer_idx"])
        mlm.load(path)
        mlm.block.eval()
        print(f"\n== {path}", flush=True)
        for s in (s for s in SECTIONS if s in only):
            res = {"wide": wide, "recall": recall, "paths": paths, "strandedcpu": strandedcpu, "strandedcpuwide": strandedcpuwide}[s](mlm) \
                if s != "plain" else plain(mlm, gen)
            print(f"  {s:<7} {res}", flush=True)
        layers = find_decoder_layers(model)                  # Remove this wrapper before attaching the next adapter.
        layers[mlm.layer_idx] = layers[mlm.layer_idx].layer


if __name__ == "__main__":
    main()
