"""Read the statistics store using an attached model and tokenizer.

    from experiments.statistics import owid_hook
    owid_hook.load("owid_v2_null.pt", "store.json")
    owid_hook.attach(model, tok)
    out = owid_hook.read(model, tok, question, options)
    use_store = out["choice"] is not None and out["abstain"] < TAU

Use the attention loader and policy the adapter trained with (policy B in the recorded runs).
The read runs layers 0..L on [bos] + "Q: <question>\nA:", without options or a chat
template, then the memory block. attach() leaves the hook disabled for normal
engine forwards. Results include choice, abstain, values, series and years."""
import torch

from kgmem import MemConfig
from kgmem.memlm import MemoryLM
from experiments.statistics.owid_mem import Store, choose, decode, prompt

_S = {}


def load(adapter_path, store_path, licence="all", soft=0.1, k=5):
    """Load adapter tensors and the CPU graph; attach() encodes its keys."""
    _S.update(sd=torch.load(adapter_path, map_location="cpu", weights_only=True), store=Store(store_path, licence),
              soft=soft, k=k, mlm=None, model=None)


def attach(model, tok):
    if _S.get("model") is model:
        return _S["mlm"]
    sd = _S["sd"]
    mlm = MemoryLM(model, tok, MemConfig(**{**sd["cfg"], "rel_exists_soft": _S["soft"]}), layer_idx=sd["layer_idx"])
    mlm.question_only, mlm.unpadded = True, True
    mlm.block.load_state_dict(sd["block"])
    mlm.block.eval()
    _S["store"].load(mlm)
    mlm.ctx.enabled = False
    _S.update(mlm=mlm, model=model)
    return mlm


@torch.no_grad()
def read(model, tok, question, options=None):
    mlm = attach(model, tok)
    ids = torch.tensor([mlm.bos + tok(prompt(question), add_special_tokens=False).input_ids], device=mlm.device)
    h = mlm.hidden_at_layer(ids, torch.ones_like(ids))
    mlm.read(h, ids, k=_S["k"])
    d = decode(mlm, _S["store"], _S["k"])
    d["choice"] = choose(d["values"], options) if options and d["values"] else None
    return d


def _bench():
    """Compare hook outputs with --dump and report CUDA-synchronized latency.

    Require equal choices and abstain scores within 1e-3; use the adapter's --backbone-loader configuration."""
    import argparse
    import importlib.util
    import json
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    for k in ("--model", "--adapter", "--store", "--questions", "--dump"):
        ap.add_argument(k, required=True)
    ap.add_argument("--licence", default="all")
    ap.add_argument("--n", type=int, default=300)
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
    load(a.adapter, a.store, a.licence)
    t0 = time.time()
    attach(model, tok)
    print(f"attach {time.time() - t0:.1f}s", flush=True)
    want = {json.loads(l)["qid"]: json.loads(l) for l in open(a.dump)}
    rows = [json.loads(l) for l in open(a.questions)]
    rows = [r for r in rows if r["qid"] in want][:a.n]
    for r in rows[:5]:
        read(model, tok, r["question"], r.get("options"))
    ms, same, close, diff = [], 0, 0, []
    for r in rows:
        torch.cuda.synchronize(); t = time.time()
        out = read(model, tok, r["question"], r.get("options"))
        torch.cuda.synchronize(); ms.append(1000 * (time.time() - t))
        w = want[r["qid"]]
        same += out["choice"] == w["choice"]
        if out["choice"] != w["choice"]:
            diff.append(dict(qid=r["qid"], hook=out["choice"], dump=w["choice"], gold=r.get("gold"),
                             hook_values=[(v["name"], round(v["weight"], 4)) for v in out["values"][:3]],
                             dump_values=[(v["name"], round(v["weight"], 4)) for v in w["values"][:3]],
                             options=r.get("options")))
        close += abs(out["abstain"] - w["abstain"]) < 1e-3
    ms.sort()
    print(json.dumps(dict(n=len(rows), same_choice=same, abstain_within_1e3=close,
                          ms_median=round(ms[len(ms) // 2], 1), ms_p90=round(ms[int(0.9 * len(ms))], 1),
                          ms_max=round(ms[-1], 1))), flush=True)
    for d in diff:
        print("choice differs:", json.dumps(d), flush=True)


if __name__ == "__main__":
    _bench()
