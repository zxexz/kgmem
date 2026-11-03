import random

import torch
from kgmem.graph import KnowledgeGraph, ExactScorer
from kgmem.modules import MemConfig, DeltaScratchpad, pointer_next_tokens


def test_exact_scorer():
    kg = KnowledgeGraph(["A", "B", "C", "D"], ["r", "s"])
    kg.add("A", "r", "B"); kg.add("A", "r", "C"); kg.add("B", "s", "D")
    sc = ExactScorer(kg, max_tails=4)
    t, p = sc(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 0]))
    assert set(t[0][p[0] > 0].tolist()) == {1, 2} and torch.allclose(p[0].sum(), torch.tensor(1.))
    assert t[1][p[1] > 0].tolist() == [3]
    assert p[2].sum() == 0                                   # no edge -> no mass
    kg.set("B", "s", "A")                                   # hot edit, scorer rebuilds lazily
    t, p = sc(torch.tensor([1]), torch.tensor([1]))
    assert t[0][p[0] > 0].tolist() == [0]


def test_pointer_trie_walk():
    # names: [5,6,7] and [5,9]; history ends with ... 5 6  -> candidate0 proposes 7, candidate1 proposes 5 (restart)
    cand = torch.tensor([[5, 6, 7], [5, 9, -1]])[None, None]          # (1,1,2,3)
    clen = torch.tensor([3, 2])[None, None]
    ids = torch.tensor([[1, 2, 5, 6]])
    nxt, done, partial = pointer_next_tokens(ids, cand, clen)
    assert nxt[0, 0].tolist() == [7, 5] and partial[0, 0].tolist() == [True, False]
    ids = torch.tensor([[1, 5, 6, 7]])                               # name 0 complete
    nxt, done, _ = pointer_next_tokens(ids, cand, clen)
    assert done[0, 0].tolist() == [True, False]


def test_delta_rule_is_exact_ram():
    """DeltaScratchpad.forward itself, with hand-set weights: one-hot keys and β saturated at 1 make each write
    overwrite exactly one row, and reads (which happen before the write) return the last value stored there.
    Token layout: [slot one-hot (8) | value (4) | write flag | 3 unused]."""
    n, dv = 8, 4
    pad = DeltaScratchpad(16, MemConfig(n_slots=n, d_val=dv, d_att=n, hard_addressing=True))
    with torch.no_grad():
        for p in pad.parameters():
            p.zero_()                                         # also v_r: no read-modify-write
        for ptr in (pad.ptr_w, pad.ptr_r):                    # point at earlier tokens naming the same slot
            ptr.q.weight[:, :n] = 10 * torch.eye(n); ptr.k.weight[:, :n] = 10 * torch.eye(n)
        pad.addr.weight[:, :n] = torch.eye(n)
        pad.v.weight[:, n:n + dv] = torch.eye(dv)
        pad.beta.weight[0, n + dv] = 60.0; pad.beta.bias.fill_(-30.0)     # β = sigmoid(±30): 1 on writes, ~0 on reads
    v1, v2, v3 = torch.randn(3, dv)
    def tok(slot, v=None):
        x = torch.zeros(16); x[slot] = 1
        if v is not None:
            x[n:n + dv] = v; x[n + dv] = 1
        return x
    h = torch.stack([tok(3, v1), tok(5, v3), tok(3, v2), tok(3), tok(5)])[None]   # write, write, overwrite, read, read
    reads, S, _ = pad(h, pad.init_state(1, "cpu"))
    assert pad.last_aux["beta"][0, :3].eq(1).all()
    assert torch.allclose(reads[0, 2], v1, atol=1e-6)        # read-before-write sees the old value
    assert torch.allclose(reads[0, 3], v2, atol=1e-6) and torch.allclose(reads[0, 4], v3, atol=1e-6)
    assert torch.allclose(S[0, 3], v2, atol=1e-6) and torch.allclose(S[0, 5], v3, atol=1e-6)
    assert S[0, [0, 1, 2, 4, 6, 7]].abs().sum() == 0         # no other row was touched


def test_heldout_names_are_disjoint_from_training():
    from kgmem.tasks import World, is_heldout
    rng = random.Random(0)
    train = {e for _ in range(20) for e in World(rng).kg.entities}
    held = {e for _ in range(5) for e in World(rng, names="eval").kg.entities}
    assert not (train & held) and all(map(is_heldout, held)) and not any(map(is_heldout, train))


def test_key_projection_cache_tracks_table_and_weights():
    from kgmem.modules import TopKLinker
    lk, q, keys = TopKLinker(8, 8, 4, 2), torch.randn(3, 8), torch.randn(5, 8)
    cached = lambda k: (lk(q, k), lk._kcache[2])[1]
    with torch.no_grad():
        K1 = cached(keys)
        assert cached(keys) is K1                                  # reused across decode steps
        lk.kp.weight.add_(1.0)                                     # an optimizer step
        K2 = cached(keys)
        assert K2 is not K1
        assert cached(torch.randn(5, 8)) is not K2                 # graph swap to a same-shape table
        K3 = cached(keys)
    assert torch.allclose(K3, torch.nn.functional.normalize(lk.kp(lk.knorm(keys)), dim=-1), atol=1e-6)


def test_pykeen_scorer_chunked_topk_equals_softmax():
    from kgmem.graph import PyKEENScorer
    class Fake(torch.nn.Module):                       # stands in for a pykeen ERModel: score_t + num_entities
        def __init__(self):
            super().__init__()
            self.ent, self.rel, self.num_entities = torch.nn.Parameter(torch.randn(50, 6)), torch.randn(3, 6), 50
        def score_t(self, hr_batch):
            return (self.ent[hr_batch[:, 0]] + self.rel[hr_batch[:, 1]]) @ self.ent.T
    m, h, r = Fake(), torch.randint(0, 50, (37,)), torch.randint(0, 3, (37,))
    t, p = PyKEENScorer(m, max_tails=4, temperature=2.0, max_scores=50 * 5)(h, r)     # 5 rows per chunk
    with torch.no_grad():
        ref = (m.score_t(torch.stack([h, r], 1)) / 2.0).softmax(-1).topk(4, -1)
    assert torch.equal(t, ref.indices) and torch.allclose(p, ref.values, atol=1e-6)


def test_carry_over_continues_a_started_name():
    from kgmem.modules import carried_candidates
    name_tok = torch.tensor([[5, 6, 7], [8, 9, -1]]); name_len = torch.tensor([3, 2])
    ids = torch.tensor([[1, 2, 5, 6]])                     # position 1 retrieved entity 0, then 5 6 emitted
    hist_ids = torch.tensor([[[1], [0], [1], [1]]])        # later positions retrieved the *wrong* entity
    hist_w = torch.ones(1, 4, 1)
    props, w, anym, completed = carried_candidates(ids, hist_ids, hist_w, name_tok, name_len, Tn=1)
    assert anym[0, 0] and (props[0, 0][w[0, 0] > 0] == 7).all()     # still proposes entity 0's 3rd token
    assert not completed[0, 0]
    ids = torch.tensor([[1, 2, 5, 6, 7]]); hist_ids = torch.tensor([[[1], [0], [1], [1], [1]]])
    _, _, _, completed = carried_candidates(ids, hist_ids, torch.ones(1, 5, 1), name_tok, name_len, Tn=1)
    assert completed[0, 0]                                             # name finished -> stop proposing


def test_carry_share_counts_a_dominant_low_weight_candidate():
    from kgmem.modules import carried_candidates
    name_tok = torch.tensor([[5, 6], [8, 9]]); name_len = torch.tensor([2, 2])
    ids = torch.tensor([[1, 2, 5, 6]])
    hist_ids = torch.tensor([[[1, 0], [0, 1], [1, 0], [1, 0]]])
    hist_w = torch.tensor([[[0.0, 0.0], [0.03, 0.002], [0.0, 0.0], [0.0, 0.0]]])   # entity 0 on top, under 0.05
    _, _, _, completed = carried_candidates(ids, hist_ids, hist_w, name_tok, name_len, Tn=1)
    assert not completed[0, 0]                                         # the old absolute rule misses it
    _, _, _, completed = carried_candidates(ids, hist_ids, hist_w, name_tok, name_len, Tn=1, share=0.2)
    assert completed[0, 0]
    hist_w = torch.tensor([[[0.0, 0.0], [0.002, 0.03], [0.0, 0.0], [0.0, 0.0]]])   # entity 0 now a minor candidate
    _, _, _, completed = carried_candidates(ids, hist_ids, hist_w, name_tok, name_len, Tn=1, share=0.2)
    assert not completed[0, 0]


def test_end_gate_proposes_the_line_end_after_a_finished_name():
    from types import SimpleNamespace
    from kgmem import MemConfig
    from kgmem.modules import MemoryBlock, copy_inputs
    blk = MemoryBlock(16, MemConfig(end_gate=True, carry_share=0.2, use_scratchpad=False))
    blk.end_id = 3
    g = SimpleNamespace(name_tok=torch.tensor([[5, 6], [8, 9]]), name_len=torch.tensor([2, 2]))
    ctx = SimpleNamespace(graph=g, cand_ids=torch.tensor([[[1]]]), cand_w=torch.tensor([[[0.5]]]),
                          cand_hist_ids=torch.tensor([[[1], [0], [1], [1]]]), cand_hist_w=torch.ones(1, 4, 1),
                          copy_x=torch.zeros(1, 1, 16))
    with torch.no_grad():
        blk.end_gate.bias.fill_(10.0)
    nxt, w, mass, lam = copy_inputs(blk, ctx, torch.tensor([[1, 2, 5, 6]]))      # entity 0 just finished
    assert w[0, 0, -1] == 1 and nxt[0, 0, -1] == 3 and w[0, 0, :-1].sum() == 0 and lam[0, 0] > 0.99
    nxt, w, mass, lam = copy_inputs(blk, ctx, torch.tensor([[1, 2, 5, 9]]))      # nothing finished
    assert w[0, 0, -1] == 0 and lam[0, 0] < 0.5


def test_cross_pairs_pair_each_query_with_other_variables_last_writes():
    from kgmem.memlm import _cross_pairs
    pair = torch.tensor([[0, 30, 5], [0, 40, 9], [1, 20, 3]])         # [b, query_pos, own last value_pos]
    neg = torch.tensor([[0, 5, 9], [0, 5, 12], [0, 9, 12], [1, 3, 7]])  # different variables' last writes
    got = sorted(map(tuple, _cross_pairs(pair, neg).tolist()))
    assert got == [(0, 30, 9), (0, 30, 12), (0, 40, 5), (0, 40, 12), (1, 20, 7)]


def test_empty_read_does_not_push_tokens():
    """Read normalization must attenuate near-empty states instead of amplifying them."""
    from kgmem.modules import MemoryBlock
    blk = MemoryBlock(16, MemConfig(d_q=8, d_val=8, n_slots=4, read_norm=True))
    assert blk.read_input(torch.zeros(1, 3, 8)).abs().max() == 0
    assert blk.read_input(torch.full((1, 3, 8), 1e-3)).norm(dim=-1).max() < 1e-2
    big = torch.randn(1, 3, 8) * 10
    assert torch.allclose(blk.read_input(big), blk.read_norm(big))


def test_strandedcpu_scorer_computes_relations():
    """Check strandedcpu arithmetic relations and caching; skip if the dependency is unavailable."""
    import pytest
    try:
        from kgmem.strandedcpu import StrandedCPUScorer, strandedcpu_graph, coerce, _strandedcpu_api
        _strandedcpu_api()
    except ImportError:
        pytest.skip("strandedcpu is not available")
    kg, sc = strandedcpu_graph(range(30), ["square", "successor", "digit sum"])
    sq, succ, ds = (kg.rel_index[r] for r in ("square", "successor", "digit sum"))
    assert kg.tails("5", "square") == ["25"] and kg.tails("6", "square") == [] and kg.tails("29", "successor") == []
    t, p = sc(torch.tensor([5, 12, 19]), torch.tensor([sq, succ, ds]))
    assert t[:, 0].tolist() == [25, 13, 10] and p[:, 0].tolist() == [1.0, 1.0, 1.0]
    runs = sc.runs
    sc(torch.tensor([5]), torch.tensor([sq]))
    assert sc.runs == runs                                    # cached: deterministic programs run once
    assert coerce(" 37") == 37 and coerce("-4") == -4 and coerce("Velmora") is None


def test_l2_keys_make_soft_overwrites_exact():
    """With softmax (L1) keys spread over several slots, overwriting a variable leaves part of the old value behind;
    with l2_keys the read along the same address returns exactly the last value written."""
    n, dv = 8, 4
    v1, v2 = torch.randn(dv), torch.randn(dv)
    k = torch.tensor([0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0])                 # a spread address, same for both writes
    for l2 in (False, True):
        pad = DeltaScratchpad(16, MemConfig(n_slots=n, d_val=dv, l2_keys=l2))
        u = pad.unit(k) if l2 else k
        S = torch.zeros(n, dv)
        for v in (v1, v2):
            S = S + u[:, None] * (v - u @ S)[None, :]
        err = float((u @ S - v2).abs().max())
        assert (err < 1e-6) == l2, (l2, err)


def test_carry_share_does_not_finish_a_prefix_of_a_heavier_name():
    from kgmem.modules import carried_candidates
    name_tok = torch.tensor([[5, 6, -1], [5, 6, 7]]); name_len = torch.tensor([2, 3])     # "10" and "100"
    ids = torch.tensor([[1, 2, 5, 6]])
    hist_ids = torch.tensor([[[0, 1], [0, 1], [0, 1], [0, 1]]])
    heavy_long = torch.tensor([[[0.0, 0.0], [0.34, 0.66], [0.0, 0.0], [0.0, 0.0]]])
    props, w, anym, completed = carried_candidates(ids, hist_ids, heavy_long, name_tok, name_len, Tn=1, share=0.2)
    assert not completed[0, 0] and (props[0, 0][w[0, 0] > 0] == 7).all()       # keeps spelling "100"
    heavy_short = torch.tensor([[[0.0, 0.0], [0.66, 0.34], [0.0, 0.0], [0.0, 0.0]]])
    _, _, _, completed = carried_candidates(ids, hist_ids, heavy_short, name_tok, name_len, Tn=1, share=0.2)
    assert completed[0, 0]


def test_carry_commit_keeps_the_gate_open_while_a_started_name_is_spelled():
    from types import SimpleNamespace
    from kgmem import MemConfig
    from kgmem.modules import MemoryBlock, copy_inputs
    ids = torch.tensor([[1, 2, 5, 6]])                     # entity 0 = [5, 6, 7] was retrieved at position 1
    hist_ids = torch.tensor([[[1], [0], [1], [1]]])
    x = torch.zeros(1, 4, 16); x[0, 1, 0], x[0, 3, 0] = 1.0, -1.0        # gate open at position 1, shut at 3
    out = {}
    for commit in (False, True):
        blk = MemoryBlock(16, MemConfig(carry_share=0.2, carry_commit=commit, use_scratchpad=False))
        with torch.no_grad():
            blk.copy_gate.weight.zero_(); blk.copy_gate.weight[0, 0] = 10.0; blk.copy_gate.bias.zero_()
        g = SimpleNamespace(name_tok=torch.tensor([[5, 6, 7], [8, 9, -1]]), name_len=torch.tensor([3, 2]))
        ctx = SimpleNamespace(graph=g, cand_ids=hist_ids, cand_w=torch.ones(1, 4, 1), cand_hist_ids=hist_ids,
                              cand_hist_w=torch.ones(1, 4, 1), copy_x=x)
        nxt, w, mass, lam = copy_inputs(blk, ctx, ids)
        out[commit] = lam[0, 3].item()
        assert 7 in nxt[0, 3][w[0, 3] > 0].tolist()                       # the carried name proposes its 3rd token
    assert out[False] < 0.01 and out[True] > 0.99


def test_exact_mentions_take_the_longest_name_ending_at_each_position():
    from kgmem.modules import exact_mentions
    alias = {(5, 6): 0, (6,): 1, (5, 6, 7): 2, (8,): -1}             # -1: two entities share the name "8"
    got = exact_mentions(torch.tensor([[1, 5, 6, 7, 6, 8]]), 6, alias, 3)
    assert got.tolist() == [[-1, -1, 0, 2, 1, -1]]
    assert exact_mentions(torch.tensor([[1, 5, 6, 7, 6, 8]]), 2, alias, 3).tolist() == [[1, -1]]   # last 2 only


def test_exact_prior_reuses_or_replaces_a_candidate_slot():
    from kgmem.modules import _exact_prior
    idx, w = torch.tensor([[3, 4, 5]] * 3), torch.tensor([[0.5, 0.3, 0.2]] * 3)
    i2, w2 = _exact_prior(idx, w, torch.tensor([4, 7, -1]), 0.9)
    assert i2.tolist() == [[3, 4, 5], [3, 4, 7], [3, 4, 5]]
    assert torch.allclose(w2[0], torch.tensor([0.05, 0.93, 0.02]))
    assert torch.allclose(w2[1], torch.tensor([0.0625, 0.0375, 0.9]))
    assert torch.equal(w2[2], w[2])


def test_exact_link_on_relation_names_moves_only_the_spelled_positions():
    from kgmem import MemConfig
    from kgmem.modules import RelationMentionLinker
    torch.manual_seed(0)
    lk = RelationMentionLinker(16, 16, MemConfig(exact_link=0.9, rel_mention=True)).eval()
    x, keys = torch.randn(1, 4, 16), torch.randn(3, 16)
    with torch.no_grad():
        *_, ref = lk(x, keys)
        *_, got = lk(x, keys, exact=torch.tensor([[-1, 2, -1, 0]]))
    p, q = ref["p"][0], got["p"][0]
    assert torch.equal(p[0], q[0]) and torch.equal(p[2], q[2])
    assert q[1, 2] >= 0.9 and q[3, 0] >= 0.9 and torch.allclose(q.sum(-1), torch.ones(4))


def test_assignment_prose_never_reuses_the_held_out_sentences():
    import random as _r
    from experiments.synthetic import eval_adapters
    from kgmem.tasks import assignment_prose
    gen = assignment_prose(_r.Random(0), 500)
    assert len(set(gen)) > 200 and not set(gen) & set(eval_adapters.ASSIGN)
    held = {w.strip(".,;").lower() for s in eval_adapters.ASSIGN for w in s.split()}
    nouns = {"oven", "alarm", "dial", "setting", "coat", "walls", "soup", "door", "sky", "tea"}
    assert not any(n in s.split() for s in gen for n in nouns & held)              # no eval sentence's subject


def test_carry_top_carries_one_of_several_equal_answers():
    from kgmem.modules import carried_candidates
    name_tok = torch.tensor([[5, 6], [8, 9], [3, 4], [1, 7], [2, 2]])     # five two-token names
    name_len = torch.full((5,), 2)
    ids = torch.tensor([[0, 0, 5]])                                         # name 0's first token was emitted
    hist_ids = torch.tensor([[[0, 1, 2, 3, 4]] * 3])
    hist_w = torch.tensor([[[0.0] * 5, [0.19] * 5, [0.0] * 5]])             # retrieved at position 1: five at 0.19
    _, w, anym, _ = carried_candidates(ids, hist_ids, hist_w, name_tok, name_len, Tn=1, share=0.2)
    assert not anym[0, 0]                                                   # none holds 0.2 of the total
    props, w, anym, _ = carried_candidates(ids, hist_ids, hist_w, name_tok, name_len, Tn=1, share=0.2, top=0.5)
    assert anym[0, 0] and (props[0, 0][w[0, 0] > 0] == 6).all()            # name 0 keeps being spelled


def test_covered_until_marks_positions_inside_a_longer_exact_name():
    from kgmem.modules import covered_until
    alias = {(7,): 0, (5, 6, 7, 8): 1, (9,): 2}           # "Western" = 7 alone, and inside the title 5 6 7 8
    cov = covered_until(torch.tensor([[1, 5, 6, 7, 8, 9]]), alias, 4)[0].tolist()
    big = 1 << 30
    assert cov == [big, 4, 4, 4, big, big]               # 5, 6 and 7 lie inside the title, which ends at 4


def test_merge_topk_sums_repeated_entities_before_cutting():
    from kgmem.modules import merge_topk
    ids = torch.tensor([[3, 7, 3, 3, 9], [1, 1, 2, 0, 0]])
    w = torch.tensor([[0.1, 0.25, 0.1, 0.1, 0.05], [0.2, 0.2, 0.5, 0.0, 0.0]])
    out_ids, out_w = merge_topk(ids, w, 2)
    assert out_ids[0].tolist() == [3, 7] and torch.allclose(out_w[0], torch.tensor([0.3, 0.25]))   # 3 wins, summed
    assert out_ids[1].tolist() == [2, 1] and torch.allclose(out_w[1], torch.tensor([0.5, 0.4]))


def test_carry_rel_scales_fresh_candidates_to_the_carried_mass():
    from types import SimpleNamespace
    from kgmem import MemConfig
    from kgmem.modules import MemoryBlock, copy_inputs
    ids = torch.tensor([[1, 2, 5, 6]])                      # entity 0 = [5, 6, 7] retrieved at position 1, weight 0.01
    hist_ids = torch.tensor([[[1], [0], [1], [1]]])
    hist_w = torch.tensor([[[0.0], [0.01], [0.0], [0.0]]])
    g = SimpleNamespace(name_tok=torch.tensor([[5, 6, 7], [8, 9, -1]]), name_len=torch.tensor([3, 2]))
    share = {}
    for rel in (False, True):
        blk = MemoryBlock(16, MemConfig(carry_share=0.2, carry_fresh=0.3, carry_rel=rel, use_scratchpad=False))
        ctx = SimpleNamespace(graph=g, cand_ids=torch.tensor([[[1]]]), cand_w=torch.tensor([[[0.9]]]),   # confident fresh
                              cand_hist_ids=hist_ids, cand_hist_w=hist_w, copy_x=torch.zeros(1, 1, 16))
        nxt, w, mass, lam = copy_inputs(blk, ctx, ids)
        share[rel] = float(w[0, 0][nxt[0, 0] == 7].sum() / mass[0, 0])          # pointer share of the carried 3rd token
    assert share[False] < 0.1 and abs(share[True] - 1 / 1.3) < 1e-4            # fresh = 0.3 x carried mass
