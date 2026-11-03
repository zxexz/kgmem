"""Semantic-parsing baselines using mquake.py's graph and edit protocol.

  python -m experiments.qa.mquake_baselines llm --prep runs/mquake/prep.pt --model Qwen/Qwen2.5-7B-Instruct --out runs/mquake/baselines/qwen.json
  python -m experiments.qa.mquake_baselines parser --prep runs/mquake/prep.pt --model HuggingFaceTB/SmolLM2-135M --out runs/mquake/baselines/parser.json

llm prompts an instruction model with relation names and training examples.
parser fine-tunes all backbone weights on the training questions. Both emit
an entity name and relation chain, link those names to the graph vocabulary,
and traverse the edited graph. Unreachable walks return no answer. Ambiguous
entity names try each matching ID until a walk produces an answer.
"""
import argparse
import difflib
import json
import random
import re
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.qa import mquake
from kgmem.paths import ensure_parent


def key(s):
    return re.sub(r"[^0-9a-z]+", " ", mquake.norm(s)).strip()


def chain_line(entity, rel_names):
    return f"entity: {entity} | relations: {' > '.join(rel_names)}"


class Linker:
    def __init__(self, ents, rels):
        self.ents, self.rels = ents, rels
        self.by_key = defaultdict(list)
        for i, n in enumerate(ents):
            self.by_key[key(n)].append(i)
        self.max_words = max(len(k.split()) for k in self.by_key)
        self.rel_keys = {key(r): i for i, r in enumerate(rels)}

    def named_in(self, q):
        words = key(q).split()
        out = set()
        for i in range(len(words)):
            for j in range(i + 1, min(len(words), i + self.max_words) + 1):
                k = " ".join(words[i:j])
                if k in self.by_key:
                    out.add(k)
        return out

    def parse(self, text, q):
        """(entity ids to walk from, relation ids) from a model's line, or (None, None)."""
        m = re.search(r"entity:\s*(.*?)\s*\|\s*relations?:\s*(.*)", text, re.I)
        if not m:
            return None, None
        ent, rel_txt = key(m.group(1)), m.group(2).split("\n")[0]
        if ent in self.by_key:
            ids = self.by_key[ent]
        else:
            cands = self.named_in(q)
            if not cands:
                return None, None
            best = max(cands, key=lambda c: (difflib.SequenceMatcher(None, c, ent).ratio(), len(c)))
            ids = self.by_key[best]
        path = []
        for r in rel_txt.split(">"):
            r = key(r)
            if not r:
                continue
            if r in self.rel_keys:
                path.append(self.rel_keys[r])
            else:
                near = difflib.get_close_matches(r, list(self.rel_keys), n=1, cutoff=0.0)
                path.append(self.rel_keys[near[0]])
        return (ids, path) if path else (None, None)


def answer(kb, ids, path):
    for e in ids or []:
        a = mquake.walk(kb, e, path)
        if a is not None:
            return a
    return None


def evaluate(P, parsed, ks, unedited=True, H=frozenset()):
    """Evaluate parsed question-to-entity/path mappings under each test set and edit batch size."""
    ents = P["entities"]
    none = len(ents)
    ext = list(ents) + ["<no answer>"]
    res = {}
    for name, T in P["tests"].items():
        cases = T["cases"]
        for k in ks:
            preds = []
            if k == "masked":
                every = mquake.apply_edits({}, cases)
                for c in cases:
                    kb = mquake.masked_kb(T["kb"], every, c)
                    preds += [answer(kb, *parsed[q]) for q in c["qs"]]
            else:
                groups = [cases] if k == "all" else [cases[i:i + int(k)] for i in range(0, len(cases), int(k))]
                for g in groups:
                    kb = mquake.apply_edits(T["kb"], g)
                    preds += [answer(kb, *parsed[q]) for c in g for q in c["qs"]]
            preds = [none if p is None else p for p in preds]
            res[f"{name} k={k}"] = mquake.score_cases(cases, preds, ext)[0]
            if H:
                res[f"{name} k={k} by relation"] = mquake.subset_scores(cases, preds, ext, H)
        if unedited:
            preds = [answer(T["kb"], *parsed[q]) for c in cases for q in c["qs"]]
            res[f"{name} unedited"] = mquake.score_cases(cases, [none if p is None else p for p in preds], ext, "orig")[0]
        print(name, json.dumps({k: v["multi_hop"] if "multi_hop" in v else v for k, v in res.items()
                                if k.startswith(name + " ")}), flush=True)
    return res


def all_questions(P):
    qs = {q for T in P["tests"].values() for c in T["cases"] for q in c["qs"]}
    return sorted(qs | {q for c in P["dev"] for q in c["qs"]})


def shots(P, n=12, seed=0, H=frozenset()):
    """Select training examples across hop counts, favoring relation coverage."""
    rng = random.Random(seed)
    pool = [x for x in P["train"] if len(x[2]) >= 2 and not set(x[2]) & H]
    rng.shuffle(pool)
    out, seen = [], set()
    for hops in (2, 3, 4):
        picked = 0
        for x in sorted((x for x in pool if len(x[2]) == hops), key=lambda x: -len(set(x[2]) - seen)):
            if picked == n // 3:
                break
            if x[1] in {y[1] for y in out}:
                continue
            out.append(x)
            seen |= set(x[2])
            picked += 1
    return out


def llm(args):
    P = torch.load(args.prep, weights_only=False)
    ents, rels = P["entities"], P["relations"]
    link = Linker(ents, rels)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    H = mquake.held_out(P, args.hold_out_rel)
    ex = shots(P, args.shots, H=H)
    head = ("You turn questions into knowledge-graph queries. The graph's relations are: " + "; ".join(rels) + ".\n"
            "For a question, write the entity it starts from, as the question names it, and the relations to follow "
            "from that entity in order, each taken from the list. Answer with one line in the form "
            "\"entity: <name> | relations: <relation> > <relation>\".\n\n"
            + "".join(f"Question: {q}\n{chain_line(ents[t], [rels[r] for r in p])}\n\n" for q, t, p, _ in ex))
    qs = all_questions(P)
    prompts = [tok.apply_chat_template([{"role": "user", "content": head + f"Question: {q}"}], tokenize=False,
                                       add_generation_prompt=True) for q in qs]
    parsed, raw, t0 = {}, {}, time.time()
    if args.engine == "vllm":                     # same prompts and greedy decoding; the shared prefix is cached
        from vllm import LLM, SamplingParams
        engine = LLM(args.model, dtype="bfloat16", max_model_len=4096, enable_prefix_caching=True,
                     gpu_memory_utilization=args.gpu_mem)
        outs = engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=64))
        for q, o in zip(qs, outs):
            raw[q] = o.outputs[0].text.strip()
            parsed[q] = link.parse(raw[q], q)
        print(f"{len(qs)} questions in {time.time() - t0:.0f}s", flush=True)
        return finish(P, parsed, raw, args)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for i in range(0, len(qs), args.batch):
        enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
        for q, o in zip(qs[i:i + args.batch], out[:, enc.input_ids.shape[1]:]):
            raw[q] = tok.decode(o, skip_special_tokens=True).strip()
            parsed[q] = link.parse(raw[q], q)
        if i % (20 * args.batch) == 0:
            print(f"{i + len(prompts[i:i + args.batch])}/{len(qs)} questions, {time.time() - t0:.0f}s; e.g. {raw[qs[i]]!r}",
                  flush=True)
    finish(P, parsed, raw, args)


def parser(args):
    P = torch.load(args.prep, weights_only=False)
    ents, rels = P["entities"], P["relations"]
    link = Linker(ents, rels)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
    H = mquake.held_out(P, args.hold_out_rel)
    data = [(f"Q: {q}\nPath:", " " + chain_line(ents[t], [rels[r] for r in p]) + "\n") for q, t, p, _ in P["train"]
            if not set(p) & H]
    rng = random.Random(args.seed)
    steps = args.epochs * len(data) // args.batch
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    model.train()
    t0 = time.time()
    for step in range(steps):
        if step % (len(data) // args.batch) == 0:
            rng.shuffle(data)
        b = data[(step * args.batch) % len(data):][:args.batch]
        ids, labels = [], []
        for prompt, target in b:
            p = tok(prompt, add_special_tokens=False).input_ids
            t = tok(target, add_special_tokens=False).input_ids
            ids.append(p + t); labels.append([-100] * len(p) + t)
        L = max(map(len, ids))
        x = torch.tensor([[tok.pad_token_id] * (L - len(s)) + s for s in ids], device=dev)
        y = torch.tensor([[-100] * (L - len(s)) + s for s in labels], device=dev)
        attn = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in ids], device=dev)
        loss = model(input_ids=x, attention_mask=attn, labels=y).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    if args.save_model:
        model.save_pretrained(ensure_parent(args.save_model))
        tok.save_pretrained(ensure_parent(args.save_model))
    qs = all_questions(P)
    parsed, raw = {}, {}
    for i in range(0, len(qs), args.batch_eval):
        enc = tok([f"Q: {q}\nPath:" for q in qs[i:i + args.batch_eval]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
        for q, o in zip(qs[i:i + args.batch_eval], out[:, enc.input_ids.shape[1]:]):
            raw[q] = tok.decode(o, skip_special_tokens=True).strip()
            parsed[q] = link.parse(raw[q], q)
    finish(P, parsed, raw, args)


def finish(P, parsed, raw, args):
    H = mquake.held_out(P, args.hold_out_rel)
    dev_cases = [c for c in P["dev"] if not set(c["path"]) & H]
    edited = mquake.apply_edits(P["dev_kb"], dev_cases)
    gold = [dict(c, new_id=mquake.walk(edited, c["topic"], c["path"]), new=[]) for c in dev_cases]
    preds = [answer(edited, *parsed[q]) for c in dev_cases for q in c["qs"]]
    none = len(P["entities"])
    dev = mquake.score_cases(gold, [none if p is None else p for p in preds], list(P["entities"]) + ["<none>"])[0]
    print("dev", json.dumps(dev), flush=True)
    res = evaluate(P, parsed, args.k.split(","), H=H)
    res["dev"] = dev
    unparsed = sum(1 for v in parsed.values() if v[0] is None)
    res["unparsed_questions"] = unparsed
    json.dump(dict(results=res, raw=raw, args=vars(args)), open(ensure_parent(args.out), "w"))
    print(json.dumps({k: v["multi_hop"] if "multi_hop" in v else v for k, v in res.items() if isinstance(v, dict)}),
          "unparsed", unparsed)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("llm", "parser"):
        s = sub.add_parser(name)
        s.add_argument("--prep", default='runs/mquake/prep.pt')
        s.add_argument("--model", required=True)
        s.add_argument("--k", default="1,100,1000,all,masked")
        s.add_argument("--out", required=True)
        s.add_argument("--hold_out_rel", default="", help="relation names kept out of training / worked examples")
    l = sub.choices["llm"]
    l.add_argument("--shots", type=int, default=12)
    l.add_argument("--batch", type=int, default=32)
    l.add_argument("--engine", default="hf", choices=["hf", "vllm"])
    l.add_argument("--gpu_mem", type=float, default=0.85, help='fraction of GPU memory reserved by vLLM')
    p = sub.choices["parser"]
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--batch_eval", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model", default=None, help='output directory for the fine-tuned parser')
    args = ap.parse_args()
    {"llm": llm, "parser": parser}[args.cmd](args)


if __name__ == "__main__":
    main()
