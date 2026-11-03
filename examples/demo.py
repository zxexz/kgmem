"""Query a graph, edit it, replace it, then save and read scratchpad state.

  python examples/demo.py
  python examples/demo.py --adapter checkpoints/archive/kgmem_adapter.pt
  python examples/demo.py --model google/gemma-4-26B-A4B-it --dtype bfloat16 --adapter checkpoints/gemma4.pt
  python examples/demo.py --strandedcpu

The default is the shipped SmolLM2 v9 adapter. Each adapter requires its original
base model. The arithmetic example also needs the strandedcpu package.
"""
import argparse
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kgmem import MemConfig, MemoryLM
from kgmem.inspector.graphs import demo_graph
from kgmem.paths import checkpoint_path
from kgmem.tasks import World

def ask(mlm, q):
    mem = mlm.generate(f"Q: {q}\nA:")
    base = mlm.generate(f"Q: {q}\nA:", use_memory=False)
    print(f"  Q: {q:<48} memory:{mem!s:<14} base LM:{base}")


def strandedcpu_demo(mlm):
    """Query arithmetic relations absent from adapter training."""
    from kgmem.strandedcpu import strandedcpu_graph
    kg, scorer = strandedcpu_graph(range(121), ["successor", "double", "square", "digit sum", "smallest prime factor"])
    mlm.load_graph(kg, scorer=scorer)
    mlm.block.eval()
    print(f"\n[strandedcpu arithmetic: {len(kg.entities)} numbers, {len(kg.relations)} relations, every edge a C program]")
    for q, gold in [("What is the successor of 41?", "42"), ("What is the double of 23?", "46"),
                    ("What is the square of 9?", "81"), ("What is the digit sum of 87?", "15"),
                    ("What is the smallest prime factor of 91?", "7"),
                    ("What is the digit sum of the square of 9?", "9")]:
        out = mlm.generate(f"Q: {q}\nA:").strip()
        print(f"  Q: {q:<48} memory: {out:<6} (gold {gold}{'' if out == gold else ', wrong'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default=checkpoint_path("smollm2-v9.pt"))
    ap.add_argument("--dtype", default="float32", help="bfloat16 for Gemma 4 26B-A4B")
    ap.add_argument("--strandedcpu", action="store_true",
                    help="run only the arithmetic queries (requires strandedcpu)")
    ap.add_argument("--state", default="runs/memory_state.pt", help="scratchpad state output")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 device_map="auto" if torch.cuda.is_available() else None).eval()
    sd = torch.load(args.adapter, map_location="cpu")
    mlm = MemoryLM(model, tok, MemConfig(**sd["cfg"]), layer_idx=sd["layer_idx"])
    mlm.load(args.adapter)
    if args.strandedcpu:
        return strandedcpu_demo(mlm)
    mlm.block.eval()

    kg = demo_graph()
    mlm.load_graph(kg)
    print("\n[hand-written graph]")
    for q in ["What is the capital of Velmora?", "Which currency is used in Quistan?", "Where does Talia live?",
              "In which country does Talia live?", "Where does the leader of Velmora live?",
              "Where does Talia's friend live?"]:
        ask(mlm, q)

    # The vocabulary is unchanged, so only the edge table needs rebuilding.
    kg.set("Velmora", "capital", "Pellin")
    kg.set("Talia", "lives in", "Sorvik")
    print("\n[after editing two edges at inference time]")
    for q in ["What is the capital of Velmora?", "Where does Talia live?", "In which country does Talia live?"]:
        ask(mlm, q)

    # Draw names from the held-out pool as well as replacing the facts.
    w = World(random.Random(4242), names="eval")
    mlm.load_graph(w.kg)
    print("\n[swapped to a fresh random world]")
    rng = random.Random(7)
    for _ in range(5):
        q, a, _, _ = w.sample_question(rng)
        print(f"  (gold: {a})", end="")
        ask(mlm, q)

    # Only the saved buffer carries information between sessions.
    if mlm.block.pad is not None:
        mlm.reset_memory()
        mlm.remember("Set kel to fox. Set vom to tea. Set kel to rice.\n")
        Path(args.state).parent.mkdir(parents=True, exist_ok=True)
        torch.save(mlm.block.persistent_state, args.state)
        mlm.block.persistent_state.copy_(torch.load(args.state, weights_only=True))
        print(f"\n[new session, no shared context; state loaded from {args.state}]")
        for v, gold in [("kel", "rice"), ("vom", "tea")]:
            mlm.block.persistent_state.copy_(torch.load(args.state, weights_only=True))   # Restore the same state before each independent read.
            out = mlm.generate(f"Q: What is {v}?\nA:", persist=True)
            print(f"  Q: What is {v}?   memory:{out}   (gold:{gold})")


if __name__ == "__main__":
    main()
