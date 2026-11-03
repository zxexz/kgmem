"""Integration tests with a tiny random Llama and a downloaded tokenizer."""
import random

import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from kgmem import MemConfig, MemoryLM
from kgmem.modules import copy_mixture_probs
from kgmem.tasks import World, kg_batch, scratchpad_batch


@pytest.fixture(scope="module")
def mlm():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2)
    model = LlamaForCausalLM(cfg).eval()
    m = MemoryLM(model, tok, MemConfig(d_q=32, d_val=32, n_slots=8))
    for p in m.block.parameters():                       # make every path non-trivial
        torch.nn.init.normal_(p, std=0.2)
    m.block.eval()
    return m


def test_train_step_runs_and_has_grads(mlm):
    rng = random.Random(0)
    w = World(rng); mlm.load_graph(w.kg)
    for b in (kg_batch(mlm, w, rng, 2), scratchpad_batch(mlm, rng, 2)):
        mlm.block.train()
        loss, logs = mlm.loss(b)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mlm.trainable_parameters())
        for p in mlm.trainable_parameters():
            p.grad = None
    mlm.block.eval()


@torch.no_grad()
def test_incremental_decoding_matches_full_forward(mlm):
    """KV-cached generation (mention-attention cache + scratchpad state) must equal a full re-forward."""
    rng = random.Random(1)
    w = World(rng); mlm.load_graph(w.kg)
    q, a, _, _ = w.sample_question(rng)
    ids = torch.tensor([mlm.tok(f"Q: {q}\nA:" + " " + a + "\n", add_special_tokens=False).input_ids])
    T0 = ids.shape[1] - 5
    # incremental
    mlm.ctx.clear(); mlm.ctx.aux_idx = None; mlm.ctx.write_mask = None; mlm.ctx.state = None
    mlm.ctx.aux_mention = None
    out = mlm.lm_step(input_ids=ids[:, :T0], use_cache=True)
    inc = [copy_mixture_probs(mlm.block, mlm.ctx, ids[:, :T0], out.logits)]
    for t in range(T0, ids.shape[1]):
        out = mlm.lm_step(input_ids=ids[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
        inc.append(copy_mixture_probs(mlm.block, mlm.ctx, ids[:, :t + 1], out.logits))
    # full re-forward for each prefix
    for j, t in enumerate(range(T0, ids.shape[1] + 1)):
        logits = mlm.forward(ids[:, :t])
        full = copy_mixture_probs(mlm.block, mlm.ctx, ids[:, :t], logits)
        assert torch.allclose(full, inc[j], atol=1e-4), f"mismatch at prefix {t}"


def test_incremental_decoding_matches_full_forward_with_inference_rules():
    """Cached and full forwards must agree with copy-gate history and exact aliases enabled."""
    _incremental_matches_full(carry_share=0.2, carry_commit=True, end_gate=True, exact_link=0.9)


def test_incremental_decoding_matches_full_forward_with_relation_mentions():
    """Cached and full forwards must agree with relation mentions and edge-existence filtering."""
    _incremental_matches_full(carry_share=0.2, carry_commit=True, end_gate=True, exact_link=0.9, hops=3,
                              rel_mention=True, relm_raw_cos=30.0, relm_first=True, rel_halt=True, rel_exists=True)


def test_incremental_decoding_matches_full_forward_with_relation_decoder():
    """The relation decoder's incremental cache must agree with a full forward."""
    _incremental_matches_full(carry_share=0.2, carry_commit=True, end_gate=True, exact_link=0.9, hops=3,
                              rel_decoder=32, rel_exists=True, rel_named=30.0)


def _incremental_matches_full(**over):
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2)
    m = MemoryLM(LlamaForCausalLM(cfg).eval(), tok, MemConfig(d_q=32, d_val=32, n_slots=8, **over))
    for p in m.block.parameters():
        torch.nn.init.normal_(p, std=0.2)
    m.block.eval()
    rng = random.Random(1)
    w = World(rng); m.load_graph(w.kg)
    q, a, _, _ = w.sample_question(rng)
    ids = torch.tensor([m.tok(f"Q: {q}\nA:" + " " + a + "\n", add_special_tokens=False).input_ids])
    T0 = ids.shape[1] - 5
    with torch.no_grad():
        m.ctx.clear(); m.ctx.aux_idx = None; m.ctx.write_mask = None; m.ctx.state = None; m.ctx.aux_mention = None
        m.ctx.tok_ids = ids[:, :T0]
        out = m.lm_step(input_ids=ids[:, :T0], use_cache=True)
        inc = [copy_mixture_probs(m.block, m.ctx, ids[:, :T0], out.logits)]
        for t in range(T0, ids.shape[1]):
            m.ctx.tok_ids = ids[:, :t + 1]
            out = m.lm_step(input_ids=ids[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
            inc.append(copy_mixture_probs(m.block, m.ctx, ids[:, :t + 1], out.logits))
        for j, t in enumerate(range(T0, ids.shape[1] + 1)):
            logits = m.forward(ids[:, :t])
            full = copy_mixture_probs(m.block, m.ctx, ids[:, :t], logits)
            assert torch.allclose(full, inc[j], atol=1e-4), f"mismatch at prefix {t}"


@torch.no_grad()
def test_persistent_scratchpad_roundtrip(mlm, tmp_path):
    mlm.reset_memory()
    mlm.remember("Set kel to fox.")
    s1 = mlm.block.persistent_state.clone()
    assert s1.abs().sum() > 0
    mlm.save(tmp_path / "a.pt")
    mlm.reset_memory()
    mlm.load(tmp_path / "a.pt")
    assert torch.equal(mlm.block.persistent_state, s1)


def test_sparse_logits_match_full(mlm):
    """Logits only at answer positions must give exactly the same loss as full-sequence logits."""
    rng = random.Random(3)
    w = World(rng); mlm.load_graph(w.kg)
    for b in (kg_batch(mlm, w, rng, 2), scratchpad_batch(mlm, rng, 2)):
        l_sparse, _ = mlm.loss(b, sparse_logits=True)
        l_full, _ = mlm.loss(b, sparse_logits=False)
        assert torch.allclose(l_sparse, l_full, atol=1e-5), (l_sparse, l_full)


def test_stop_weight_weights_only_the_token_after_each_answer(mlm):
    rng = random.Random(5)
    w = World(rng); mlm.load_graph(w.kg)
    b = kg_batch(mlm, w, rng, 2)
    lp, wt = mlm.answer_logprobs(b, stop_weight=3.0)
    bi, ti = (b.labels[:, 1:] != -100).nonzero(as_tuple=True)
    ends = {(r, e - 1) for (r, _, e, _) in b.spans if e < b.ids.shape[1]}
    assert ends and all(float(x) == (3.0 if (int(r), int(t)) in ends else 1.0) for r, t, x in zip(bi, ti, wt))
    assert all(mlm.tok.decode([int(b.ids[r, t + 1])]) == "\n" for r, t in ends)     # and that token is the line end


def test_stage1_on_cached_hidden_matches_full_forward(mlm):
    """Auxiliary loss from cached attach-layer states must equal full-forward auxiliary loss."""
    rng = random.Random(4)
    w = World(rng); mlm.load_graph(w.kg)
    for b in (kg_batch(mlm, w, rng, 2), scratchpad_batch(mlm, rng, 2)):
        _, full = mlm.loss(b)
        h = mlm.hidden_at_layer(b.ids, b.attn)
        ph = mlm.hidden_at_layer(b.prefix.ids, b.prefix.attn) if b.prefix is not None else None
        _, cached = mlm.stage1_loss(b, h, ph)
        for k in cached:
            assert abs(cached[k] - full[k]) < 1e-4, (k, cached[k], full[k])


def test_aux_loss_is_finite_when_no_mention_was_matched(mlm):
    """On real text a batch can have no string-matched mention; that used to make the loss NaN."""
    rng = random.Random(5)
    w = World(rng); mlm.load_graph(w.kg)
    b = kg_batch(mlm, w, rng, 2)
    b.aux_mention = torch.full_like(b.aux_mention, -1)
    mlm.block.train()
    loss, _ = mlm.loss(b)
    loss.backward()
    grads = [p.grad for p in mlm.trainable_parameters() if p.grad is not None]
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
    for p in mlm.trainable_parameters():
        p.grad = None
    mlm.block.eval()


def test_single_session_scratchpad_supervises_writes(mlm):
    """Supervise value-token writes and variable pointers when assignments share the query batch."""
    b = scratchpad_batch(mlm, random.Random(6), B=2, split_session=False)
    assert b.prefix is None and (b.beta_tgt[b.ptr_w[:, 0], b.ptr_w[:, 1]] == 1).all()
    with torch.no_grad():
        mlm.forward(b.ids, b.attn)
        with_ptr = mlm._pad_aux_loss(b)
        b.ptr_w = None
        assert torch.isfinite(with_ptr) and with_ptr > mlm._pad_aux_loss(b)


def test_write_session_stops_at_the_attach_layer(mlm):
    """A prefix write skips later layers while preserving scratchpad loss and gradients.

    Compare with the full-prefix forward to detect changes in the truncated path.
    """
    from kgmem.memlm import find_decoder_layers
    b = scratchpad_batch(mlm, random.Random(7), B=2)
    calls = []
    hook = find_decoder_layers(mlm.lm)[-1].register_forward_hook(lambda *a: calls.append(1))
    try:
        with torch.no_grad():
            mlm.write_session(b.prefix.ids, b.prefix.attn)
    finally:
        hook.remove()
    assert not calls

    def loss_and_grads():
        loss, _ = mlm.loss(b)
        loss.backward()
        g = [p.grad.clone() if p.grad is not None else None for p in mlm.trainable_parameters()]
        for p in mlm.trainable_parameters():
            p.grad = None
        return loss.detach(), g

    mlm.block.train()
    new = loss_and_grads()
    mlm.write_session = lambda ids, attn=None, state=None: (mlm.forward(ids, attn, state=state, keep=1),
                                                           mlm.ctx.state)[1]
    try:
        old = loss_and_grads()
    finally:
        del mlm.write_session
        mlm.block.eval()
    assert torch.allclose(new[0], old[0], atol=1e-5)
    assert all((a is None and c is None) or torch.allclose(a, c, atol=1e-5) for a, c in zip(new[1], old[1]))


def test_evaluate_leaves_the_training_graph_loaded(mlm):
    """train.py with --swap_every > 1 keeps training on its world after an eval, so evaluate() must restore it."""
    from experiments.synthetic import train
    w = World(random.Random(8)); mlm.load_graph(w.kg)
    g = mlm.ctx.graph
    train.evaluate(mlm, 12345, n_worlds=1, B=1)
    mlm.block.eval()
    assert mlm.ctx.graph is g and mlm.ctx.enabled


def test_overwrites_of_one_variable_are_paired(mlm):
    """addr_same pairs each earlier write of a variable with its last write, and nothing else."""
    b = scratchpad_batch(mlm, random.Random(9), B=4, n_vars=3, n_assign=6)
    s, p = b.prefix.addr_same, b.prefix
    assert s.numel() > 0 and (s[:, 1] < s[:, 2]).all()
    var_at = {(bb, v): vp for bb, v, vp in p.ptr_w.tolist()}             # value position -> variable-name position
    name = lambda bb, v: p.ids[bb, var_at[(bb, v)]].item()
    assert all(name(bb, e) == name(bb, l) for bb, e, l in s.tolist())


def test_graph_gate_is_supervised(mlm):
    rng = random.Random(10)
    w = World(rng); mlm.load_graph(w.kg)
    for b in (kg_batch(mlm, w, rng, 2), scratchpad_batch(mlm, rng, 2)):
        _, logs = mlm.loss(b)
        assert "aux_gate" in logs and logs["aux_gate"] == logs["aux_gate"]          # present and not NaN


def test_relation_worlds_and_plain_text_train(mlm):
    """RelWorld (sampled relation names, named or numeric entities) and plain-text batches both give a finite
    loss with gradients; the plain loss is a KL to the memory-off model, so it is >= 0."""
    from kgmem.tasks import RelWorld, plain_batch, is_heldout
    rng = random.Random(11)
    for numeric in (False, True):
        rw = RelWorld(rng, n_ent=12, n_rel=3, numeric=numeric)
        assert not any(is_heldout(r) for r in rw.kg.relations)
        mlm.load_graph(rw.kg)
        b = kg_batch(mlm, rw, rng, 2)
        mlm.block.train()
        loss, logs = mlm.loss(b)
        loss.backward()
        assert torch.isfinite(loss) and logs["link_acc"] == logs["link_acc"]
    texts = [mlm.tok("The river runs past the old mill and under a bridge.", add_special_tokens=False).input_ids]
    loss, logs = mlm.loss(plain_batch(mlm, rng, texts, B=2))
    loss.backward()
    assert torch.isfinite(loss) and logs["kl_plain"] >= -1e-5
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mlm.trainable_parameters())
    for p in mlm.trainable_parameters():
        p.grad = None
    mlm.block.eval()


def test_relation_mention_linker_trains_and_decodes_incrementally():
    """Check relation-mention gradients and agreement of cached and full forwards.

    Use a separate model because the shared fixture disables relation mentions.
    """
    from kgmem.tasks import RelWorld
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                                         num_attention_heads=4, num_key_value_heads=2)).eval()
    m = MemoryLM(model, tok, MemConfig(d_q=32, d_val=32, n_slots=8, rel_mention=True, rel_key_pool="last",
                                       relm_first=True, rel_halt=True, l2_keys=True, relm_raw_cos=30.0,
                                       hop_order=0.1, cont_keys=True, readout_gate=True,
                                       gate_pad_value=True))
    for p in m.block.parameters():
        torch.nn.init.normal_(p, std=0.2)
    rng = random.Random(12)
    rw = RelWorld(rng, n_ent=12, n_rel=3)
    m.load_graph(rw.kg)
    b = kg_batch(m, rw, rng, 2)
    assert (b.aux_rel_mention[:, 0] >= 0).all()                    # RelWorld questions name their relations
    assert (b.aux_rel_span_start <= b.aux_rel_mention).all()
    m.block.train()
    loss, logs = m.loss(b)
    loss.backward()
    assert torch.isfinite(loss) and "relm0_acc" in logs and m.block.kg.relm.mix.weight.grad.abs().sum() > 0
    assert m.block.kg.halt_rel.weight.grad.abs().sum() > 0
    m.block.eval()
    q, a, _, _ = rw.sample_question(rng)
    ids = torch.tensor([m.tok(f"Q: {q}\nA: {a}\n", add_special_tokens=False).input_ids])
    T0 = ids.shape[1] - 4
    with torch.no_grad():
        m.ctx.clear(); m.ctx.aux_idx = None; m.ctx.write_mask = None; m.ctx.state = None; m.ctx.aux_mention = None
        out = m.lm_step(input_ids=ids[:, :T0], use_cache=True)
        inc = [copy_mixture_probs(m.block, m.ctx, ids[:, :T0], out.logits)]
        for t in range(T0, ids.shape[1]):
            out = m.lm_step(input_ids=ids[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
            inc.append(copy_mixture_probs(m.block, m.ctx, ids[:, :t + 1], out.logits))
        for j, t in enumerate(range(T0, ids.shape[1] + 1)):
            full = copy_mixture_probs(m.block, m.ctx, ids[:, :t], m.forward(ids[:, :t]))
            assert torch.allclose(full, inc[j], atol=1e-4), f"mismatch at prefix {t}"


def test_question_kl_is_zero_when_the_memory_changes_nothing(mlm):
    """question_kl compares the decoded distribution with the memory on and off at non-answer positions. With the
    block's outputs zeroed and the copy gate shut, the two are identical, so the KL is 0."""
    rng = random.Random(13)
    w = World(rng); mlm.load_graph(w.kg)
    b = kg_batch(mlm, w, rng, 2)
    saved = {k: v.clone() for k, v in mlm.block.state_dict().items()}
    with torch.no_grad():
        for lin in (mlm.block.out_kg, mlm.block.out_pad, mlm.block.pad_readout):
            lin.weight.zero_()
        mlm.block.copy_gate.bias.fill_(-50.0)
        assert float(mlm.question_kl(b)) < 1e-4
    mlm.block.load_state_dict(saved)
    loss, logs = mlm.loss(b, keep_base=1.0)
    assert torch.isfinite(loss) and logs["kl_question"] > 0


def test_cont_keys_are_unit_vectors_and_train():
    """cont_keys=True: write and read addresses are unit vectors (not distributions over slots), and the addressing
    losses are finite and reach the address map, both when writes are in a prefix session and in the batch itself.
    readout_gate=True: the gate on the scratchpad's logit readout gets its own supervision."""
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                                         num_attention_heads=4, num_key_value_heads=2)).eval()
    m = MemoryLM(model, tok, MemConfig(d_q=32, d_val=32, n_slots=8, l2_keys=True, cont_keys=True, readout_gate=True))
    for p in m.block.parameters():
        torch.nn.init.normal_(p, std=0.2)
    m.block.train()
    for split in (True, False):
        b = scratchpad_batch(m, random.Random(3), B=3, split_session=split)
        m.block.zero_grad()
        loss, logs = m.loss(b)
        loss.backward()
        aux = m.block.pad.last_aux
        assert torch.allclose(aux["kr"].norm(dim=-1), torch.ones(()), atol=1e-5) and (aux["kr"] < 0).any()
        assert torch.isfinite(loss) and m.block.pad.addr.weight.grad.abs().sum() > 0
        assert "aux_ro_gate" in logs and m.block.ro_gate.weight.grad.abs().sum() > 0     # the readout gate is supervised
    m.block.eval()


def test_read_on_given_states_matches_the_forward():
    """Retrieval from externally supplied states must match retrieval in a calibrated LM forward."""
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2)
    m = MemoryLM(LlamaForCausalLM(cfg).eval(), tok, MemConfig(d_q=32, d_val=32, n_slots=8, hops=2, rel_decoder=32,
                                                               rel_exists=True, exact_link=0.9))
    for p in m.block.parameters():
        torch.nn.init.normal_(p, std=0.2)
    m.block.eval()
    rng = random.Random(3)
    w = World(rng)
    m.calibrate([f"Q: {w.sample_question(rng)[0]}\nA:" for _ in range(8)])
    assert not torch.allclose(m.block.in_scale, torch.ones_like(m.block.in_scale))
    m.load_graph(w.kg)
    q = w.sample_question(rng)[0]
    ids = torch.tensor([m.bos + tok(f"Q: {q}\nA:", add_special_tokens=False).input_ids])
    T = ids.shape[1]
    with torch.no_grad():
        m.forward(ids, torch.ones_like(ids), aux_idx=torch.tensor([T - 1]))
    full_ids, full_w = m.ctx.cand_ids[0, -1].clone(), m.ctx.cand_w[0, -1].clone()
    full_rel = [r[0].clone() for r in m.ctx.kg_aux["rel"]]
    h = m.hidden_at_layer(ids, torch.ones_like(ids))
    out = m.read(h, ids)
    assert torch.equal(m.ctx.cand_ids[0, -1], full_ids) and torch.allclose(m.ctx.cand_w[0, -1], full_w, atol=1e-6)
    for hop, r in zip(out["hops"], full_rel):
        assert hop["rel"][0] == int(r.argmax())
    assert len(out["halt"]) == 2 and abs(sum(out["halt"]) - 1) < 1e-5 and out["candidates"]
