"""Run the upstream RAE baseline on mquake.py's prepared graphs.

  python -m experiments.qa.mquake_rae --rae /path/to/RAE --prep runs/mquake/prep.pt --set cf3k-v2 --shard 0 --nshards 4 --out runs/mquake/rae/v2_0.jsonl
  python -m experiments.qa.mquake_rae --score runs/mquake/rae/v2_*.jsonl

Use RAE's extraction, pruning, examples, answering, and answer checks. Replace
its graph and live Wikidata name lookup with the prepared graph, applying all
edits together. Disable correctConflict, which restores original facts needed
by the test case's chain. SparseExtract also retains paths at dead ends so
sparse graphs survive RAE's fixed-depth search.

Report upstream QA_func's alias check and alias_fixed, which checks every alias
rather than stopping at the first. A case succeeds if any question succeeds.
"""
import argparse
import difflib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from kgmem.paths import ensure_parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rae", help="path to the upstream RAE checkout (required unless --score)")
    ap.add_argument("--prep", default='runs/mquake/prep.pt')
    ap.add_argument("--set", default="cf3k-v2")
    ap.add_argument("--data_json", default=None, help="the MQuAKE file the set came from (default: by set name)")
    ap.add_argument("--mquake_dir", default="data/mquake")
    ap.add_argument("--model", default="EleutherAI/gpt-j-6B")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="a seeded random sample of this many cases (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--score", nargs="*", default=None)
    a = ap.parse_args()
    if a.score is not None:
        return score(a.score)
    if not a.rae or not a.out:
        ap.error("run mode requires --rae and --out")
    run(a)


def score(paths):
    rows = [json.loads(l) for p in paths for l in open(p)]
    n = len(rows)
    out = {k: round(100 * sum(r[k] for r in rows) / max(1, n), 2) for k in ("pruned", "raw", "pruned_alias_fixed")}
    out["cases"] = n
    print(json.dumps(out))
    return out


def SparseExtract(rae_model):
    """Retain dead-end paths during RAE's fixed-depth beam search.

    RAE otherwise drops paths that end before gold-chain length + 2. Sparse
    MQuAKE graphs often terminate earlier. Retaining those paths lets upstream
    entropy pruning choose where the answer chain ends.
    """
    import torch

    class Extract(rae_model.Extract):
        def multi_hop_search(self, input_text, input_entity, iteration, fact_needed):
            text_list, entity_list = [input_text], [input_entity]
            entity_eid_list, text_prob_list = [rae_model.entity2id(input_entity)], [1.0]
            for i in range(iteration):
                new_text_list, next_entity_list, next_entites_eid_list, new_text_prob_list = [], [], [], []
                for j, text in enumerate(text_list):
                    text_prob = text_prob_list[j]
                    facts, next_entites, next_entites_eid, top_prob = self.retr_fact_KG_sole_prob(
                        input_text, text, entity_list[j], entity_eid_list[j], fact_needed)
                    if facts is None:                               # Retain a completed path when no outgoing edge remains.
                        new_text_list.append(text); next_entity_list.append(entity_list[j])
                        next_entites_eid_list.append(entity_eid_list[j]); new_text_prob_list.append(text_prob)
                        continue
                    for k, fact in enumerate(facts):
                        new_text_list.append(text + '\n' + fact)
                        next_entity_list.append(next_entites[k])
                        next_entites_eid_list.append(next_entites_eid[k])
                        new_text_prob_list.append(top_prob[k] * text_prob)
                text_list, entity_list = new_text_list, next_entity_list
                entity_eid_list, text_prob_list = next_entites_eid_list, new_text_prob_list
            text_prob = torch.tensor(text_prob_list)
            final_score = text_prob if self.args.loss == "prob_div" else text_prob * torch.log2(text_prob)
            _, index = torch.topk(final_score, 1)
            return text_list[index][len(input_text + ' '):]
    return Extract


def run(a):
    import torch
    from experiments.qa import mquake
    # Upstream RAE uses paths relative to its checkout after chdir.
    rae = str(Path(a.rae).expanduser().resolve())
    output_path = Path(a.out).expanduser().resolve()
    data_json = Path(a.data_json or Path(a.mquake_dir).expanduser() / mquake.TESTS[a.set]).expanduser().resolve()
    P = torch.load(os.path.expanduser(a.prep), weights_only=False)
    T = P["tests"][a.set]
    ents, qids, pids = P["entities"], P["qids"], P["pids"]
    kb = mquake.apply_edits(T["kb"], T["cases"])
    triplets = defaultdict(list)
    for (s, r), t in kb.items():
        triplets[qids[s]].append(f"{qids[s]}\t{pids[r]}\t{qids[t]}")
    label = {q: ents[i] for i, q in enumerate(qids)}
    by_name = defaultdict(list)
    for i, n in enumerate(ents):
        by_name[n.lower()].append(qids[i])
    names = list(by_name)

    tag = "gptj" if "gpt-j" in a.model.lower() else a.model.split("/")[-1].lower()
    sys.argv = [sys.argv[0], "--model", tag, "--dataset", a.set]       # RAE's modules parse argv when imported
    sys.path.insert(0, rae)
    os.chdir(rae)
    import utils_func
    import model as rae_model
    import main as rae_main
    from wiki_api import wikidata

    question = {"q": ""}

    def entity2id(q):
        if q.lower() in by_name:
            return by_name[q.lower()][0]
        ql = question["q"].lower()
        inq = [n for n in names if n in ql and q.lower() in n]
        if inq:
            return by_name[max(inq, key=len)][0]
        near = difflib.get_close_matches(q.lower(), names, n=1, cutoff=0.6)
        return by_name[near[0]][0] if near else "not applicable"

    def id2entity(q):
        return label.get(q, q)
    for mod in (wikidata, rae_model, utils_func):
        for name, fn in (("entity2id", entity2id), ("id2entity", id2entity)):
            if hasattr(mod, name):
                setattr(mod, name, fn)

    args = utils_func.args
    args.correctConflict = False
    args.template, args.NatureL = True, True
    args.device = "cuda"
    model = utils_func.AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to("cuda").eval()
    tok = utils_func.AutoTokenizer.from_pretrained(a.model)
    lines = mquake.load_cases(data_json)
    train = utils_func.load_dataset("data/MQuAKE-CF.json")
    NL_dict = utils_func.load_dataset("data/cloze_templates_NL.json")
    relation_dict, reverse_dict = utils_func.build_relation(utils_func.load_dataset(args.relation_path), NL_dict)
    rae_main.args, rae_main.train, rae_main.NL_dict = args, train, NL_dict
    rae_main.tuple_list = utils_func.load_train_question("data/train_question_tuple.txt")
    extractor = SparseExtract(rae_model)(model, tok, triplets, relation_dict, reverse_dict, {}, args)
    pruner = rae_model.Prune(model, tok, args)

    simple = lambda s: re.sub(r"[^a-zA-Z ]+", "", s).lower()
    todo = list(enumerate(lines))
    if a.sample:                                       # Random sampling avoids bias from files ordered by hop count.
        import random
        todo = [todo[i] for i in sorted(random.Random(a.seed).sample(range(len(todo)), a.sample))]
    todo = todo[a.shard::a.nshards]
    if a.limit:
        todo = todo[:a.limit]
    with open(ensure_parent(output_path), "w") as f:
        for i, line in todo:
            truth = {" ".join([ll[0], NL_dict[line["orig"]["new_triples"][l][1]], ll[2]])
                     for l, ll in enumerate(line["orig"]["new_triples_labeled"])}
            res = dict(case=i, case_id=line["case_id"], raw=0, pruned=0, pruned_alias_fixed=0, answers=[])
            for q in line["questions"]:
                question["q"] = q
                extract_prompt = rae_main.tempalate_extractor(q, True, "prob")
                ans_prompt = rae_main.tempalate_extractor(q, True, "ans")
                try:
                    raw_entity = utils_func.ner_entity(q)
                    retrieved = extractor.multi_hop_search(extract_prompt + "Question: " + q + "\nAnswer:", raw_entity,
                                                           len(truth) + 2, [])
                    pruned = pruner.prune_fact(q, retrieved, ans_prompt)
                except Exception as e:                       # main.py logs "Fail to extract" and goes on
                    print(f"!!!! case {i}: fail to extract: {type(e).__name__}: {e}", flush=True)
                    retrieved = pruned = ""
                raw_ans, raw_ok = utils_func.QA_func(model, tok, line, retrieved, q, ans_prompt, "Raw")
                ans, ok = utils_func.QA_func(model, tok, line, pruned, q, ans_prompt, "Pruned")
                fixed = any(simple(x) and simple(x) in ans for x in [line["new_answer"], *line["new_answer_alias"]])
                res["answers"].append(dict(q=q, pruned_facts=pruned, answer=ans))
                res["raw"] |= raw_ok
                res["pruned"] |= ok
                res["pruned_alias_fixed"] |= int(fixed)
                if ok:
                    break                                   # main.py stops at the first question answered right
            f.write(json.dumps(res) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
