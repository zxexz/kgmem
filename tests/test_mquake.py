"""MQuAKE preparation, graph edits, and scoring on local fixtures."""
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.qa import mquake


def case(cid, chain, edits, new_chain, questions):
    """chain / new_chain: [(subject, relation, object)] with "Qn:label" entities and "Pn:label" relations."""
    split = lambda x: x.split(":")
    ids = lambda tr: [[split(s)[0], split(r)[0], split(o)[0]] for s, r, o in tr]
    labels = lambda tr: [[split(s)[1], split(r)[1], split(o)[1]] for s, r, o in tr]
    hop = lambda s, r, o: dict(question=f"What is the {split(r)[1]} of {split(s)[1]}?", answer=split(o)[1],
                               answer_alias=[], cloze="")
    return dict(case_id=cid, requested_rewrite=[], questions=questions,
                answer=split(chain[-1][2])[1], answer_alias=[], new_answer=split(new_chain[-1][2])[1],
                new_answer_alias=[f"The {split(new_chain[-1][2])[1]}"],
                single_hops=[hop(*t) for t in chain], new_single_hops=[hop(*t) for t in new_chain],
                orig=dict(triples=ids(chain), triples_labeled=labels(chain), new_triples=ids(new_chain),
                          new_triples_labeled=labels(new_chain), edit_triples=ids(edits)))


A = case(1, [("Q1:Fernando Santos", "P27:country of citizenship", "Q2:Portugal"),
             ("Q2:Portugal", "P35:head of state", "Q3:Marcelo")],
         [("Q1:Fernando Santos", "P27:country of citizenship", "Q4:United Kingdom")],
         [("Q1:Fernando Santos", "P27:country of citizenship", "Q4:United Kingdom"),
          ("Q4:United Kingdom", "P35:head of state", "Q5:Elizabeth II")],
         ["Who is the head of state of the country Fernando Santos is a citizen of?",
          "Who heads the state whose citizen Fernando Santos is?", "Fernando Santos's country is led by whom?"])
B = case(2, [("Q6:Kismet", "P495:country of origin", "Q2:Portugal"), ("Q2:Portugal", "P36:capital", "Q7:Lisbon")],
         [("Q2:Portugal", "P36:capital", "Q8:Porto")],
         [("Q6:Kismet", "P495:country of origin", "Q2:Portugal"), ("Q2:Portugal", "P36:capital", "Q8:Porto")],
         ["What is the capital of the country Kismet comes from?", "Kismet's country has which capital?",
          "Name the capital of Kismet's country of origin."])
C = case(3, [("Q9:Juarez", "P495:country of origin", "Q4:United Kingdom"),
             ("Q4:United Kingdom", "P35:head of state", "Q5:Elizabeth II")],
         [("Q4:United Kingdom", "P35:head of state", "Q3:Marcelo")],
         [("Q9:Juarez", "P495:country of origin", "Q4:United Kingdom"),
          ("Q4:United Kingdom", "P35:head of state", "Q3:Marcelo")],
         ["Who is the head of state of the country Juarez comes from?", "Juarez's country is headed by whom?",
          "Who leads the country of origin of Juarez?"])


@pytest.fixture()
def prepped(tmp_path):
    (tmp_path / "MQuAKE-CF.json").write_text(json.dumps([A, B, C]))
    (tmp_path / "MQuAKE-CF-3k-v2.json").write_text(json.dumps([A, C]))
    out = tmp_path / "runs" / "mquake" / "prep.pt"
    mquake.prep(SimpleNamespace(data=str(tmp_path), out=str(out), dev_cases=0))
    return torch.load(out, weights_only=False)


def test_prep_keeps_test_cases_out_of_training(prepped):
    P = prepped
    questions = {q for q, *_ in P["train"]}
    assert set(B["questions"]) <= questions
    assert not questions & set(A["questions"] + C["questions"])
    assert P["relations"][P["pids"].index("P27")] == "country of citizenship"


def test_graph_holds_unedited_facts_and_edits_reach_the_new_answer(prepped):
    P = prepped
    ents, e = P["entities"], {q: i for i, q in enumerate(P["qids"])}
    r = {p: i for i, p in enumerate(P["pids"])}
    T = P["tests"]["cf3k-v2"]
    kb = T["kb"]
    assert kb[(e["Q1"], r["P27"])] == e["Q2"]               # before the edit
    assert kb[(e["Q4"], r["P35"])] == e["Q5"]               # the edited chain's unedited fact is in the graph
    a, c = T["cases"]
    assert ents[mquake.walk(kb, a["topic"], a["path"])] == "Marcelo" and "marcelo" in a["orig"]
    one = mquake.apply_edits(kb, [a])
    assert ents[mquake.walk(one, a["topic"], a["path"])] == "Elizabeth II" and "the elizabeth ii" in a["new"]
    # with both cases' edits the UK's head of state is Marcelo, which A's labelled answer doesn't know about:
    # MQuAKE-CF-3k's conflicts, and why its k = all ceiling is below 100%
    both = mquake.apply_edits(kb, [a, c])
    assert ents[mquake.walk(both, a["topic"], a["path"])] == "Marcelo"
    # masked (MQuAKE-Remastered's protocol): A keeps the unedited fact its own chain passes through, C still gets
    # its edit
    every = mquake.apply_edits({}, [a, c])
    assert ents[mquake.walk(mquake.masked_kb(kb, every, a), a["topic"], a["path"])] == "Elizabeth II"
    assert ents[mquake.walk(mquake.masked_kb(kb, every, c), c["topic"], c["path"])] == "Marcelo"


def test_score_cases_counts_a_case_solved_if_any_question_is(prepped):
    P = prepped
    a, c = P["tests"]["cf3k-v2"]["cases"]
    ents = P["entities"]
    eliz, marcelo = ents.index("Elizabeth II"), ents.index("Marcelo")
    res, per_case = mquake.score_cases([a, c], [marcelo, eliz, marcelo] + [eliz] * 3, ents)
    assert per_case == [[False, True, False], [False, False, False]]
    assert res["multi_hop"] == 0.5 and res["all_three"] == 0.0 and res["per_question"] == round(1 / 6, 4)


def test_grow_hops_extends_a_three_hop_adapter():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from kgmem import MemoryLM
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2)
    model = LlamaForCausalLM(cfg).eval()
    small = MemoryLM(model, tok, mquake.metaqa_cfg(d_q=32, d_val=32, n_slots=8))
    big = MemoryLM(model, tok, mquake.mquake_cfg(d_q=32, d_val=32, n_slots=8, rel_decoder=0))
    sd = mquake.grow_hops(small.block.state_dict(), 4)
    big.block.load_state_dict(sd, strict=True)                           # every key, every shape
    old, new = small.block.state_dict(), big.block.state_dict()
    for k in ("kg.rel.2.q.weight", "kg.relm.aq.2.weight"):
        assert torch.equal(new[k], old[k]) and torch.equal(new[k.replace(".2.", ".3.")], old[k])
    assert torch.equal(new["kg.halt.weight"][:3], old["kg.halt.weight"])
    assert new["kg.halt_rel.weight"].shape == (4, 10)
    assert torch.equal(new["kg.halt_rel.weight"][:3, :3], old["kg.halt_rel.weight"][:, :3])


def test_batches_carry_the_path_and_mention(prepped):
    pytest.importorskip("transformers")
    import random
    from collections import defaultdict
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from kgmem import MemoryLM
    from kgmem.graph import KnowledgeGraph
    P = prepped
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2)
    mlm = MemoryLM(LlamaForCausalLM(cfg).eval(), tok, mquake.mquake_cfg(d_q=32, d_val=32, n_slots=8))
    by_hops = defaultdict(list)
    for x in P["train"]:
        by_hops[len(x[2])].append(x)
    b = mquake.mquake_batch(mlm, P["entities"], by_hops, random.Random(0), B=3)
    known = (b.aux_rel != -100).sum(-1).tolist()
    assert known == [h + 1 for h in b.aux_hops.tolist()] and (b.aux_mention >= 0).all()
    kg = KnowledgeGraph(list(P["entities"]), list(P["relations"]), [])
    mquake.set_graph(kg, P["train_kb"])
    mlm.load_graph(kg)
    loss, _ = mlm.loss(b)
    assert torch.isfinite(loss)


def test_baseline_linker_and_walks_score_like_mquake(prepped):
    from experiments.qa import mquake_baselines as mb
    P = prepped
    ents, rels = P["entities"], P["relations"]
    link = mb.Linker(ents, rels)
    a, c = P["tests"]["cf3k-v2"]["cases"]
    ids, path = link.parse("entity: Santos | relations: citizenship > head of state", a["qs"][0])
    assert [ents[i] for i in ids] == ["Fernando Santos"] and [rels[r] for r in path] == ["country of citizenship",
                                                                                         "head of state"]
    assert link.parse("no line here", a["qs"][0]) == (None, None)
    gold = {q: ([x["topic"]], x["path"]) for x in (a, c) for q in x["qs"]}
    res = mb.evaluate(P, gold, ["1", "all", "masked"])
    assert res["cf3k-v2 k=1"]["multi_hop"] == 1.0 and res["cf3k-v2 k=masked"]["multi_hop"] == 1.0
    assert res["cf3k-v2 k=all"]["multi_hop"] == 0.5               # the two cases' edits conflict (see above)
    none = {q: (None, None) for q in gold}
    assert mb.evaluate(P, none, ["all"])["cf3k-v2 k=all"]["multi_hop"] == 0.0
