"""WebQSP path search, synthetic function graphs, and LoRA on local fixtures."""
import random

import numpy as np

from experiments.retrieval import funclib
from experiments.qa import webqsp


def test_rel_name_reads_freebase_names_and_their_inverses():
    assert webqsp.rel_name("people.person.nationality", False) == "people person nationality"
    assert webqsp.rel_name("location.country.official_language", True) == \
        "reverse of location country official language"


def test_answer_path_prefers_the_most_precise_shortest_path():
    # topic 0; relation 0 reaches the answer 2 and a non-answer 3; relation 1 reaches only 2; 2 hops via a CVT node 4
    NR = 3
    tri = np.array([(0, 0, 2), (0, 0, 3), (0, 1, 2), (0, 2, 4), (4, 2, 5)], dtype=np.int32)
    path, ends = webqsp.answer_path(tri, [0], [2], NR)
    assert path == [1] and ends == {2}
    path, ends = webqsp.answer_path(tri, [0], [5], NR)             # through the unnamed node
    assert path == [2, 2] and ends == {5}
    path, _ = webqsp.answer_path(np.array([(2, 0, 0)], dtype=np.int32), [0], [2], NR)
    assert path == [0 + NR]                                         # an edge into the topic is walked backwards
    assert webqsp.answer_path(tri, [0], [9], NR) == (None, None)


def test_funclib_questions_follow_their_chains_and_respect_types():
    lib = funclib.library(300, seed=0)
    assert len({f["name"] for f in lib}) == 300
    assert all(f["key"].startswith(f["name"] + "(") for f in lib)
    rng = random.Random(0)
    for side in ("train", "held_out"):
        world = funclib.FuncWorld(rng, lib, side)
        rels = world.kg.relations
        for _ in range(50):
            q, ans, head, chain = world.sample_question(rng, p_two_hop=0.5)
            fs = [lib[rels.index(c)] for c in chain]
            assert any(f["held_out"] for f in fs) == (side == "held_out")
            assert head in world.vals[fs[0]["in_type"]] and ans in world.vals[fs[-1]["out_type"]]
            assert all(a["out_type"] == b["in_type"] for a, b in zip(fs, fs[1:]))
            assert ans != head or len(chain) == 2
            assert head in q and not any(f["name"] in q for f in fs)   # asked in words, not by name


def test_lora_starts_as_the_identity_and_trains():
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from kgmem.lora import add_lora, load_lora, lora_state
    from kgmem.memlm import find_decoder_layers
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    m = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    ids = torch.tensor([tok("the cat sat on the mat").input_ids])
    with torch.no_grad():
        before = m(ids).logits
    mods = add_lora(find_decoder_layers(m), upto=1, r=4, alpha=8.0)
    assert len(mods) == 14                                  # 7 linears in each of layers 0 and 1
    with torch.no_grad():
        assert torch.allclose(m(ids).logits, before)        # B starts at zero
    for mod in mods.values():
        mod.B.data.normal_(std=0.1)
    sd = lora_state(mods)
    with torch.no_grad():
        changed = m(ids).logits
    assert not torch.allclose(changed, before)
    for mod in mods.values():
        mod.B.data.zero_()
    load_lora(mods, sd)
    with torch.no_grad():
        assert torch.allclose(m(ids).logits, changed)       # saved and restored exactly
