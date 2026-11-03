"""Train a MemoryBlock on a frozen decoder with synthetic tasks.

  python -m experiments.synthetic.train --model HuggingFaceTB/SmolLM2-135M --steps 300
  python -m experiments.synthetic.train --model google/gemma-4-26B-A4B-it --dtype bfloat16 --steps 1500 --batch 8 --out runs/synthetic/gemma4.pt

--swap_every controls graph resampling. Optional tasks train scratchpad recall,
held-out relation linking, and KL preservation of base-model predictions.
check_linking.py measures name-key alignment before training a new backbone.
"""
import argparse
import json
import random
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kgmem import MemConfig, MemoryLM
from kgmem.tasks import RelWorld, World, assignment_prose, kg_batch, plain_batch, scratchpad_batch
from kgmem.paths import ensure_parent

# Fixed evaluation prose, independent of the generated plain-text training pool.
PLAIN_EVAL = [
    "The river runs past the old mill and under a stone bridge before it reaches the sea.",
    "In spring the water is high and brown, and children are told to stay away from the banks.",
    "She opened the letter slowly, read it twice, and then put it back in the drawer without a word.",
    "Most of the town's shops close early on Sundays, so plan to buy what you need on Saturday.",
    "The committee will publish its findings next month, after the last round of interviews.",
    "He learned to cook from his grandmother, who never measured anything and never burned a meal.",
    "A good map shows you not only where things are but also what the mapmaker thought was important.",
    "The train was late again, which gave us time to finish the crossword and argue about the answers.",
]
# Prompts for base-model completions used in plain-text training.
PLAIN_PROMPTS = ["The", "In the morning", "She said that", "Scientists have found", "The history of", "When I was young",
                 "Our team", "The river", "It is important to", "Last year,", "The best way to", "Many people believe",
                 "According to the report,", "He walked", "The city of", "In 1998,"]


@torch.no_grad()
def plain_pool(mlm, n=192, length=48, seed=0, path=None):
    """Ordinary text for the plain-text task: base-model completions (memory off) of PLAIN_PROMPTS, as token ids."""
    import os
    if path and os.path.exists(path):
        return torch.load(path)
    tok, was = mlm.tok, mlm.ctx.enabled
    side, pad = tok.padding_side, tok.pad_token
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mlm.ctx.enabled = False
    torch.manual_seed(seed)
    texts = []
    try:
        for i in range(0, n, 16):
            prompts = [PLAIN_PROMPTS[(i + j) % len(PLAIN_PROMPTS)] for j in range(16)]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(mlm.device)
            gen = mlm.lm.generate(**enc, do_sample=True, top_p=0.95, temperature=0.9, max_new_tokens=length,
                                  pad_token_id=tok.pad_token_id)
            for row in tok.batch_decode(gen, skip_special_tokens=True):
                ids = tok(row.strip(), add_special_tokens=False).input_ids
                if len(ids) >= 12:
                    texts.append(ids[:length + 8])
    finally:
        mlm.ctx.enabled, tok.padding_side, tok.pad_token = was, side, pad
    if path:
        torch.save(texts, ensure_parent(path))
    return texts


def evaluate(mlm, seed, n_worlds=4, B=8, use_memory=True):
    """Evaluate fresh worlds with names reserved by tasks.is_heldout.

    Relation worlds also use held-out relation names; half use integer entities.
    Restore the caller's graph and enabled flag after evaluation, then leave the
    block in training mode.
    """
    rng = random.Random(seed)
    graph, enabled = mlm.ctx.graph, mlm.ctx.enabled
    mlm.block.eval(); mlm.ctx.enabled = use_memory
    acc = defaultdict(list)
    for i in range(n_worlds):
        w = World(rng, names="eval")
        mlm.load_graph(w.kg)
        for kind, b in [("kg", kg_batch(mlm, w, rng, B)), ("pad", scratchpad_batch(mlm, rng, B, names="eval"))]:
            if not mlm.cfg.use_scratchpad and kind == "pad":
                continue
            for k, ok, stop in mlm.exact_match(b, stops=True):
                acc[k].append(ok); acc[f"{k}+stop"].append(stop)
            if kind == "kg" and use_memory:
                acc["lm_harm"].append(prompt_nll(mlm, b, True) - prompt_nll(mlm, b, False))
        if use_memory:
            acc["plain_harm"].append(text_nll(mlm, PLAIN_EVAL, True) - text_nll(mlm, PLAIN_EVAL, False))
        rw = RelWorld(rng, names="eval", relations="eval", numeric=i % 2 == 1)
        mlm.load_graph(rw.kg)
        for k, ok, stop in mlm.exact_match(kg_batch(mlm, rw, rng, B), stops=True):
            acc[f"rel-{k}"].append(ok); acc[f"rel-{k}+stop"].append(stop)
    mlm.block.train(); mlm.ctx.graph, mlm.ctx.enabled = graph, enabled
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}


@torch.no_grad()
def text_nll(mlm, texts, use_memory: bool) -> float:
    """Mean next-token NLL over plain sentences; plain_harm = with memory minus without."""
    tok = mlm.tok
    seqs = [mlm.bos + tok(t, add_special_tokens=False).input_ids for t in texts]
    T = max(map(len, seqs))
    ids = torch.tensor([s + [0] * (T - len(s)) for s in seqs], device=mlm.device)
    attn = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], device=mlm.device)
    was = mlm.ctx.enabled
    mlm.ctx.enabled = use_memory
    try:
        logits = mlm.forward(ids, attn)
    finally:
        mlm.ctx.enabled = was
    lp = logits[:, :-1].float().log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return float(-lp[attn[:, 1:] > 0].mean())


@torch.no_grad()
def prompt_nll(mlm, b, use_memory: bool) -> float:
    """Return mean next-token NLL at unlabeled, non-padding question positions.

    lm_harm is this value with memory enabled minus the value with it disabled.
    """
    was = mlm.ctx.enabled
    mlm.ctx.enabled = use_memory
    try:
        logits = mlm.forward(b.ids, b.attn)
    finally:
        mlm.ctx.enabled = was
    lp = logits[:, :-1].float().log_softmax(-1).gather(-1, b.ids[:, 1:, None]).squeeze(-1)
    m = (b.labels[:, 1:] == -100) & (b.attn[:, 1:] > 0)
    return float(-lp[m].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--layer", type=int, default=None, help="decoder layer to attach after (default ~2/3 depth)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--aux", type=float, default=1.0, help="weight of linking/relation/halting supervision")
    ap.add_argument("--swap_every", type=int, default=1)
    ap.add_argument("--p_pad", type=float, default=0.3, help="fraction of steps on the scratchpad task")
    ap.add_argument("--p_rel", type=float, default=0.0, help='fraction of steps on random relation-name worlds')
    ap.add_argument("--p_plain", type=float, default=0.0, help='fraction of steps minimizing KL to base-model predictions on plain text')
    ap.add_argument("--plain_cache", default=None, help='path to cache generated plain-text training sequences')
    ap.add_argument("--plain_assign", type=int, default=0, help='number of assignment-like plain sentences to add as scratchpad negatives')
    ap.add_argument("--rel_mention", action="store_true", help='mix query-based relation scores with relation-name token scores')
    ap.add_argument("--keep_base", type=float, default=0.0, help="weight of KL to the base model on question text")
    ap.add_argument("--relm_raw_cos", type=float, default=0.0, help="raw-cosine prior in the relation-mention linker")
    ap.add_argument("--relm_first", action="store_true", help='also score relation names against their first-token keys')
    ap.add_argument("--rel_halt", action="store_true", help='condition halting on relation-mention confidence and attention overlap')
    ap.add_argument("--l2_keys", action="store_true", help="scratchpad delta rule with L2-normalised keys")
    ap.add_argument("--cont_keys", action="store_true", help="unit-vector scratchpad keys instead of a slot softmax")
    ap.add_argument("--readout_gate", action="store_true", help='learn a supervised gate on the scratchpad logit readout')
    ap.add_argument("--gate_pad_value", action="store_true", help='apply the readout gate to out_pad as well')
    ap.add_argument("--only_pad", action="store_true", help='train scratchpad and readout parameters; freeze the graph path and shared norm')
    ap.add_argument("--only_kg_value", action="store_true",
                    help='train out_kg, kg.val, and kg.vnorm while keeping retrieval fixed')
    ap.add_argument("--stop_weight", type=float, default=1.0, help="loss weight of the line end after each answer")
    ap.add_argument("--carry_share", type=float, default=0.0, help='minimum candidate-mass share for carrying a name; use the intended inference value')
    ap.add_argument("--end_gate", action="store_true", help='learn a newline proposal after a retrieved name completes')
    ap.add_argument("--hop_order", type=float, default=0.0, help='relation-order inference threshold used during evaluation')
    ap.add_argument("--carry_fresh", type=float, default=1.0, help='fresh-candidate weight while emitting a carried name')
    ap.add_argument("--halt_rel_aux", type=float, default=0.0, help='auxiliary loss weight on the named-relation halting term')
    ap.add_argument("--carry_commit", action="store_true", help='keep the copy gate at least as open as when a carried name started')
    ap.add_argument("--numeric_digits", default="uniform", choices=["uniform", "balanced"],
                    help='integer-name sampling distribution for relation-world training')
    ap.add_argument("--halt_rel_same", action="store_true", help='add consecutive relation-distribution overlap to halting features')
    ap.add_argument("--train_only", default=None, help='comma-separated block parameter prefixes to train, e.g. end_gate.,kg.halt_rel.')
    ap.add_argument("--no_scratchpad", action="store_true")
    ap.add_argument("--hard", action="store_true", help="straight-through one-hot addressing")
    ap.add_argument("--no_read_norm", action="store_true", help='disable scratchpad read normalization')
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default='runs/synthetic/adapter.pt')
    ap.add_argument("--init", default=None, help="warm-start from an adapter (shape-matching params only)")
    ap.add_argument("--seed", type=int, default=0, help='random seed for parameter initialization and synthetic data')
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 device_map="auto" if torch.cuda.is_available() else None)
    model.eval()                                       # Disable backbone dropout; MemoryLM freezes its parameters.
    cfg = MemConfig(use_scratchpad=not args.no_scratchpad, hard_addressing=args.hard, read_norm=not args.no_read_norm,
                    rel_mention=args.rel_mention, rel_key_pool="last" if args.rel_mention else "mean",
                    relm_raw_cos=args.relm_raw_cos, relm_first=args.relm_first, rel_halt=args.rel_halt,
                    l2_keys=args.l2_keys, cont_keys=args.cont_keys,
                    readout_gate=args.readout_gate, gate_pad_value=args.gate_pad_value, carry_share=args.carry_share,
                    end_gate=args.end_gate, halt_rel_aux=args.halt_rel_aux, halt_rel_same=args.halt_rel_same,
                    hop_order=args.hop_order, carry_commit=args.carry_commit,
                    carry_fresh=args.carry_fresh)
    mlm = MemoryLM(model, tok, cfg, layer_idx=args.layer)
    if args.init:
        print("warm start:", mlm.load(args.init, strict=False))
    if args.only_pad:   # Keep the graph path and shared norm frozen.
        for n, p in mlm.block.named_parameters():
            p.requires_grad_(n.split(".")[0] in ("pad", "out_pad", "pad_readout", "read_norm", "ro_gate"))
    if args.train_only:
        keep = tuple(x for x in args.train_only.split(",") if x)
        for n, p in mlm.block.named_parameters():
            p.requires_grad_(n.startswith(keep))
        assert any(p.requires_grad for p in mlm.block.parameters()), f"--train_only {keep} matches no parameter"
    if args.only_kg_value:  # Fit graph values while keeping retrieval fixed.
        for n, p in mlm.block.named_parameters():
            p.requires_grad_(n.startswith(("out_kg.", "kg.val.", "kg.vnorm.")))
    # Keep training arguments and parent metadata with each checkpoint.
    meta = {"args": vars(args), "parent": torch.load(args.init, map_location="cpu").get("meta") if args.init else None}
    n_tr = sum(p.numel() for p in mlm.trainable_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"attached after layer {mlm.layer_idx}; trainable {n_tr/1e6:.2f}M / {n_all/1e6:.1f}M frozen")

    print("baseline (memory off):", evaluate(mlm, 12345, use_memory=False))
    opt = torch.optim.AdamW(mlm.trainable_parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    plain = plain_pool(mlm, path=args.plain_cache) if args.p_plain > 0 else None
    if plain is not None and args.plain_assign:
        plain = plain + [tok(t, add_special_tokens=False).input_ids
                         for t in assignment_prose(random.Random(args.seed + 7), args.plain_assign)]

    def use(kg):                                                   # Reuse encoded keys while the graph object is unchanged.
        if mlm.ctx.graph is None or mlm.ctx.graph.kg is not kg:
            mlm.load_graph(kg)

    rng = random.Random(args.seed)
    world, t0, hist = None, time.time(), []
    for step in range(1, args.steps + 1):
        if world is None or step % args.swap_every == 0:
            world = World(rng)
        u = rng.random()
        if cfg.use_scratchpad and u < args.p_pad:
            use(world.kg)
            b, kind = scratchpad_batch(mlm, rng, args.batch), "pad"
        elif u < args.p_pad + args.p_plain:
            use(world.kg)
            b, kind = plain_batch(mlm, rng, plain, args.batch), "plain"
        elif u < args.p_pad + args.p_plain + args.p_rel:
            rw = RelWorld(rng, numeric=rng.random() < 0.5, digits=args.numeric_digits)
            use(rw.kg)
            b, kind = kg_batch(mlm, rw, rng, args.batch, p_two_hop=0.5), "rel"
        else:
            use(world.kg)
            b, kind = kg_batch(mlm, world, rng, args.batch), "kg"
        loss, logs = mlm.loss(b, args.aux, keep_base=args.keep_base, stop_weight=args.stop_weight)
        if "nll" in logs:
            logs[f"nll_{kind}"] = logs.pop("nll")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mlm.trainable_parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(logs)
        if step % 10 == 0:
            keys = {k for h in hist for k in h}
            avg = {k: sum(h[k] for h in hist if k in h) / sum(k in h for h in hist) for k in sorted(keys)}
            print(f"step {step:5d}  " + "  ".join(f"{k} {v:.3f}" for k, v in avg.items())
                  + f"  {(time.time()-t0)/step:.2f}s/step", flush=True)
            hist = []
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(mlm, 12345)
            print(f"eval @ {step} (held-out worlds):", json.dumps(res), flush=True)
            mlm.save(ensure_parent(args.out), dict(meta, step=step, eval=res))
    print("saved", args.out)


if __name__ == "__main__":
    main()
