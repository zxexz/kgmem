"""MetaQA preparation and graph rules on a local WikiMovies-format fixture."""
import random
from collections import defaultdict

import pytest
import torch

from experiments.qa import metaqa

KB = """Kismet|directed_by|William Dieterle
Kismet|starred_actors|Marlene Dietrich
Kismet|release_year|1944
Juarez|directed_by|William Dieterle
Juarez|starred_actors|Bette Davis
Dodsworth|directed_by|William Wyler
Dodsworth|starred_actors|Mary Astor
Jezebel|directed_by|William Wyler
Jezebel|starred_actors|Bette Davis
"""


@pytest.fixture()
def data(tmp_path):
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "kb.txt").write_text(KB)
    qs = {1: "what movies did [William Wyler] direct\tDodsworth|Jezebel\n"
             "who directed [Kismet]\tWilliam Dieterle\n",
          2: "who starred in the movies directed by [William Wyler]\tMary Astor|Bette Davis\n"
             "which person directed the movies starred by [Bette Davis]\tWilliam Dieterle|William Wyler\n",
          3: "what years were the movies directed by the directors of [Juarez] released\t1944\n"}
    for h, text in qs.items():
        d = tmp_path / f"{h}-hop" / "vanilla"
        d.mkdir(parents=True)
        for split in ("train", "dev", "test"):
            (d / f"qa_{split}.txt").write_text(text)
    return tmp_path


def test_kb_gets_an_inverse_of_every_relation(data):
    ents, rels, triples = metaqa.load_kb(data / "kb" / "kb.txt")
    idx = {n: i for i, n in enumerate(ents)}
    assert len(rels) == 2 * metaqa.NREL and rels[0] == "directed by" and rels[metaqa.NREL] == "director of"
    assert (idx["Kismet"], 0, idx["William Dieterle"]) in triples
    assert (idx["William Dieterle"], metaqa.NREL, idx["Kismet"]) in triples


def test_questions_parse_and_paths_are_recovered(data):
    ents, rels, triples = metaqa.load_kb(data / "kb" / "kb.txt")
    idx = {n: i for i, n in enumerate(ents)}
    adj, out = defaultdict(set), defaultdict(set)
    for h, r, t in triples:
        adj[(h, r)].add(t)
        out[h].add(r)
    one, skipped = metaqa.load_qa(data / "1-hop" / "vanilla" / "qa_train.txt", idx)
    assert skipped == 0 and one[0][0] == "what movies did William Wyler direct"       # brackets removed
    assert one[0][1] == idx["William Wyler"] and set(one[0][2]) == {idx["Dodsworth"], idx["Jezebel"]}
    assert one[0][3] == "what movies did NE direct"
    paths, agree = metaqa.infer_paths(one, adj, out, 1)
    assert paths["what movies did NE direct"] == (metaqa.NREL + 0,)                  # director of
    assert paths["who directed NE"] == (0,)
    three, _ = metaqa.load_qa(data / "3-hop" / "vanilla" / "qa_train.txt", idx)
    p3, _ = metaqa.infer_paths(three, adj, out, 3)
    # Juarez -> William Dieterle -> {Kismet, Juarez} -> {1944}: the topic is revisited on the way
    assert p3[three[0][3]] == (0, metaqa.NREL + 0, 3)


def test_batches_carry_the_path_hops_and_mention(data):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from kgmem import MemoryLM
    ents, rels, triples = metaqa.load_kb(data / "kb" / "kb.txt")
    idx = {n: i for i, n in enumerate(ents)}
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2)
    mlm = MemoryLM(LlamaForCausalLM(cfg).eval(), tok, metaqa.metaqa_cfg(d_q=32, d_val=32, n_slots=8))
    pools = {}
    for h in (1, 2, 3):
        items, _ = metaqa.load_qa(data / f"{h}-hop" / "vanilla" / "qa_train.txt", idx)
        adj, out = defaultdict(set), defaultdict(set)
        for a, r, t in triples:
            adj[(a, r)].add(t)
            out[a].add(r)
        pools[h] = (items, metaqa.infer_paths(items, adj, out, h)[0])
    b = metaqa.metaqa_batch(mlm, ents, pools, random.Random(0), B=4)
    hops = b.aux_hops.tolist()
    known = (b.aux_rel != -100).sum(-1).tolist()
    assert all(k == h + 1 for k, h in zip(known, hops))                              # one relation per hop
    assert (b.aux_mention >= 0).all()                                                 # every topic was found
    for (bi, s, e, kind), m in zip(b.spans, b.aux_mention.tolist()):
        assert kind == f"{int(kind[0])}-hop" and m < s
    mlm.load_graph(metaqa.make_kg(dict(entities=ents, relations=rels, triples=triples)))
    loss, logs = mlm.loss(b)
    assert torch.isfinite(loss)


def test_exclude_head_keeps_the_topic_off_hops_two_and_on():
    from kgmem import MemConfig
    from kgmem.graph import ExactScorer, GraphTensors, KnowledgeGraph
    from kgmem.modules import KGReadHead
    # A -r-> B -s-> {A, C} -t-> D (from A) / E (from C): the walk from A must not come back through A
    kg = KnowledgeGraph(list("ABCDE"), ["r", "s", "t"], [(0, 0, 1), (1, 1, 0), (1, 1, 2), (0, 2, 3), (2, 2, 4)])
    seen = {}
    for flag in (False, True):
        torch.manual_seed(0)
        head = KGReadHead(8, 8, MemConfig(exclude_head=flag, hops=3, k_ent=1, k_rel=1, k_cand=4, max_tails=4))
        hops_rel = iter([0, 1, 2])

        class FixedRel(torch.nn.Module):                  # each hop picks relations r, s, t in turn
            def __init__(self, r):
                super().__init__(); self.r = r
            def forward(self, h, keys, aux_idx=None):
                N = h.shape[0]
                return torch.full((N, 1), self.r), torch.ones(N, 1), None
        head.rel = torch.nn.ModuleList([FixedRel(next(hops_rel)) for _ in range(3)])
        head.relm = None
        head.ent.forward = lambda x, keys, *a, **k: (torch.zeros(1, 1, 1, dtype=torch.long), torch.ones(1, 1, 1),
                                                     None, None)          # the linked head is A
        g = GraphTensors(ent_keys=torch.zeros(5, 8), rel_keys=torch.zeros(3, 8), name_tok=torch.zeros(5, 1).long(),
                         name_len=torch.ones(5).long(), scorer=ExactScorer(kg, 4), kg=kg)
        ctx = type("C", (), dict(aux_idx=None, link_cache=None, write_mask=None, aux_mention=None, relm_cache=None))()
        _, cand_ids, cand_w, _ = head(torch.zeros(1, 1, 8), g, ctx)
        ids, w = cand_ids[0, 0].tolist(), cand_w[0, 0].tolist()
        seen[flag] = {kg.entities[i] for i, x in zip(ids, w) if x > 0}
    assert "A" in seen[False] and "D" in seen[False]                 # back through A: its t-tail D, and A itself
    assert "A" not in seen[True] and "D" not in seen[True] and "E" in seen[True]


def test_answers_only_batches_carry_no_relation_labels(data):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from kgmem import MemoryLM
    ents, rels, triples = metaqa.load_kb(data / "kb" / "kb.txt")
    idx = {n: i for i, n in enumerate(ents)}
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2)
    mlm = MemoryLM(LlamaForCausalLM(cfg).eval(), tok, metaqa.metaqa_cfg(d_q=32, d_val=32, n_slots=8))
    pools = {h: (metaqa.load_qa(data / f"{h}-hop" / "vanilla" / "qa_train.txt", idx)[0], {}) for h in (1, 2, 3)}
    b = metaqa.metaqa_batch(mlm, ents, pools, random.Random(0), B=2, rel_sup=False)
    assert (b.aux_rel == -100).all() and (b.aux_hops >= 0).all()          # hop counts and topics stay


def test_walk_never_passes_back_through_the_topic(data):
    from collections import defaultdict
    ents, rels, triples = metaqa.load_kb(data / "kb" / "kb.txt")
    idx = {n: i for i, n in enumerate(ents)}
    adj = defaultdict(set)
    for h, r, t in triples:
        adj[(h, r)].add(t)
    # Juarez -> William Dieterle -> {Kismet, Juarez}: Juarez is dropped on the way, so only Kismet's year remains
    years = metaqa.walk(adj, idx["Juarez"], (0, metaqa.NREL + 0, 3))
    assert years == {idx["1944"]}
    assert metaqa.walk(adj, idx["Kismet"], (0,)) == {idx["William Dieterle"]}


def test_rel_exists_keeps_a_hop_on_relations_its_heads_have():
    from kgmem import MemConfig
    from kgmem.graph import ExactScorer, GraphTensors, KnowledgeGraph
    from kgmem.modules import KGReadHead
    # A -r-> B -s-> C, and B has no r edge: hop 2 prefers r (0.6) over s (0.4), but only s leads anywhere
    kg = KnowledgeGraph(list("ABC"), ["r", "s"], [(0, 0, 1), (1, 1, 2)])
    got = {}
    for flag, soft, w2 in ((False, 0.0, 0.4), (True, 0.0, 0.4), (True, 0.1, 0.4), (True, 0.1, 0.01)):
        torch.manual_seed(0)
        head = KGReadHead(8, 8, MemConfig(rel_exists=flag, rel_exists_soft=soft, hops=2, k_ent=1, k_rel=2, k_cand=4,
                                          max_tails=4))
        dists = iter([torch.tensor([[0, 1]]), torch.tensor([[0, 1]])])
        weights = iter([torch.tensor([[1.0, 0.0]]), torch.tensor([[1 - w2, w2]])])

        class FixedRel(torch.nn.Module):
            def __init__(self, ids, w):
                super().__init__(); self.ids, self.w = ids, w
            def forward(self, h, keys, aux_idx=None):
                return self.ids.expand(h.shape[0], -1), self.w.expand(h.shape[0], -1), None
        head.rel = torch.nn.ModuleList([FixedRel(next(dists), next(weights)) for _ in range(2)])
        head.relm = None
        head.ent.forward = lambda x, keys, *a, **k: (torch.zeros(1, 1, 1, dtype=torch.long), torch.ones(1, 1, 1),
                                                     None, None)
        with torch.no_grad():
            head.halt.weight.zero_(); head.halt.bias.copy_(torch.tensor([-9.0, 9.0]))      # answer at hop 2
        g = GraphTensors(ent_keys=torch.zeros(3, 8), rel_keys=torch.zeros(2, 8), name_tok=torch.zeros(3, 1).long(),
                         name_len=torch.ones(3).long(), scorer=ExactScorer(kg, 4), kg=kg)
        ctx = type("C", (), dict(aux_idx=None, link_cache=None, write_mask=None, aux_mention=None, relm_cache=None))()
        with torch.no_grad():
            _, cand_ids, cand_w, _ = head(torch.zeros(1, 1, 8), g, ctx)
        w = {}
        for i, x in zip(cand_ids[0, 0].tolist(), cand_w[0, 0].tolist()):
            w[kg.entities[i]] = w.get(kg.entities[i], 0.0) + x
        got[flag, soft, w2] = w.get("C", 0.0)
    assert abs(got[False, 0.0, 0.4] - 0.4) < 0.01 and got[True, 0.0, 0.4] > 0.99
    # rel_exists_soft: each hop's candidates also scale by m / (m + 0.1), m the weight on relations its heads have
    # (hop 1: m = 1; hop 2: m = w2). Real belief keeps most of its weight; a sliver is no longer renormalised to 1
    assert abs(got[True, 0.1, 0.4] - (1 / 1.1) * (0.4 / 0.5)) < 1e-4
    assert abs(got[True, 0.1, 0.01] - (1 / 1.1) * (0.01 / 0.11)) < 1e-4


def test_rel_named_favours_a_relation_the_question_spells():
    from kgmem import MemConfig
    from kgmem.graph import ExactScorer, GraphTensors, KnowledgeGraph
    from kgmem.modules import KGReadHead
    # A -r-> B and A -s-> C; the hop prefers r (0.6) to s (0.4), but the question spells s's name (token 7)
    kg = KnowledgeGraph(list("ABC"), ["r", "s"], [(0, 0, 1), (0, 1, 2)])
    got = {}
    for gamma in (0.0, 30.0):
        torch.manual_seed(0)
        head = KGReadHead(8, 8, MemConfig(rel_named=gamma, hops=1, k_ent=1, k_rel=2, k_cand=4, max_tails=4))

        class FixedRel(torch.nn.Module):
            def forward(self, h, keys, aux_idx=None):
                return torch.tensor([[0, 1]]).expand(h.shape[0], -1), torch.tensor([[0.6, 0.4]]).expand(h.shape[0], -1), None
        head.rel = torch.nn.ModuleList([FixedRel()])
        head.relm = None
        head.ent.forward = lambda x, keys, *a, **k: (torch.zeros(1, 3, 1, dtype=torch.long), torch.ones(1, 3, 1),
                                                     None, None)
        g = GraphTensors(ent_keys=torch.zeros(3, 8), rel_keys=torch.zeros(2, 8), name_tok=torch.zeros(3, 1).long(),
                         name_len=torch.ones(3).long(), scorer=ExactScorer(kg, 4), kg=kg, rel_alias={(7,): 1})
        ctx = type("C", (), dict(aux_idx=None, link_cache=None, write_mask=None, aux_mention=None, relm_cache=None,
                                 tok_ids=torch.tensor([[3, 7, 4]])))()
        with torch.no_grad():
            _, cand_ids, cand_w, _ = head(torch.zeros(1, 3, 8), g, ctx)
        w = dict(zip((kg.entities[i] for i in cand_ids[0, -1].tolist()), cand_w[0, -1].tolist()))
        got[gamma] = (w.get("B", 0.0), w.get("C", 0.0))
    assert got[0.0][0] > got[0.0][1]                      # r wins without the rule
    assert got[30.0][1] > 0.9 * sum(got[30.0])            # the spelled relation wins with it
