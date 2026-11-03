"""Gemma 4 compatibility checks using a reduced random model and real tokenizer.

Preserve the configured sliding/full attention pattern, MoE, tied embeddings,
logit softcap, and multimodal wrapper while reducing model dimensions. These
tests download config and tokenizer files, not model weights, and skip when
Hub access is unavailable.
"""
import copy
import random

import pytest
import torch

from kgmem import MemConfig, MemoryLM
from kgmem.modules import copy_mixture_probs
from kgmem.tasks import World, kg_batch, scratchpad_batch

REPO = "google/gemma-4-26B-A4B-it"


@pytest.fixture(scope="module")
def mlm():
    tr = pytest.importorskip("transformers")
    try:
        cfg = tr.AutoConfig.from_pretrained(REPO)
        tok = tr.AutoTokenizer.from_pretrained(REPO)
    except Exception as e:                                    # offline / gated
        pytest.skip(f"cannot fetch {REPO}: {e}")
    cfg = copy.deepcopy(cfg)
    tc = cfg.text_config
    pattern = list(tc.layer_types[:6])                         # 5 sliding + 1 full, same as the real model
    tc.layer_types = pattern
    for k, v in dict(num_hidden_layers=len(pattern), hidden_size=64, intermediate_size=64,
                     moe_intermediate_size=32, num_experts=4, top_k_experts=2, num_attention_heads=4,
                     num_key_value_heads=2, num_global_key_value_heads=1, head_dim=16,
                     global_head_dim=32, sliding_window=16).items():
        setattr(tc, k, v)
    vc = getattr(cfg, "vision_config", None)
    if vc is not None:
        for k, v in dict(hidden_size=32, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=16).items():
            if hasattr(vc, k):
                setattr(vc, k, v)
    torch.manual_seed(0)
    model = tr.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()   # config says bf16; fp32 for exact checks
    m = MemoryLM(model, tok, MemConfig(d_q=32, d_val=32, n_slots=8))
    for p in m.block.parameters():
        torch.nn.init.normal_(p, std=0.2)
    m.block.eval()
    return m


def test_attaches_after_a_global_attention_layer(mlm):
    tc = mlm.lm.config.get_text_config()
    assert tc.layer_types[mlm.layer_idx] == "full_attention"
    assert mlm.bos == [mlm.tok.bos_token_id]                   # Gemma needs <bos>


def test_train_step(mlm):
    rng = random.Random(0)
    w = World(rng); mlm.load_graph(w.kg)
    for b in (kg_batch(mlm, w, rng, 2), scratchpad_batch(mlm, rng, 2)):
        mlm.block.train()
        loss, _ = mlm.loss(b)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mlm.trainable_parameters())
        for p in mlm.trainable_parameters():
            p.grad = None
    mlm.block.eval()


@torch.no_grad()
def test_incremental_decoding_matches_full_forward(mlm):
    rng = random.Random(1)
    w = World(rng); mlm.load_graph(w.kg)
    q, a, _, _ = w.sample_question(rng)
    ids = torch.tensor([mlm.bos + mlm.tok(f"Q: {q}\nA: {a}\n", add_special_tokens=False).input_ids])
    T0 = ids.shape[1] - 5
    mlm.ctx.clear(); mlm.ctx.aux_idx = None; mlm.ctx.write_mask = None; mlm.ctx.state = None
    mlm.ctx.aux_mention = None
    out = mlm.lm_step(input_ids=ids[:, :T0], use_cache=True)
    inc = [copy_mixture_probs(mlm.block, mlm.ctx, ids[:, :T0], out.logits)]
    for t in range(T0, ids.shape[1]):
        out = mlm.lm_step(input_ids=ids[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
        inc.append(copy_mixture_probs(mlm.block, mlm.ctx, ids[:, :t + 1], out.logits))
    for j, t in enumerate(range(T0, ids.shape[1] + 1)):
        full = copy_mixture_probs(mlm.block, mlm.ctx, ids[:, :t], mlm.forward(ids[:, :t]))
        assert torch.allclose(full, inc[j], atol=1e-4), f"mismatch at prefix {t}"


@torch.no_grad()
def test_generate_and_remember(mlm):
    rng = random.Random(2)
    w = World(rng); mlm.load_graph(w.kg)
    q, _, _, _ = w.sample_question(rng)
    assert isinstance(mlm.generate(f"Q: {q}\nA:", max_new_tokens=4), str)
    mlm.reset_memory(); mlm.remember("Set kel to fox.")
    assert mlm.block.persistent_state.abs().sum() > 0


def test_sparse_logits_match_full(mlm):
    rng = random.Random(3)
    w = World(rng); mlm.load_graph(w.kg)
    b = kg_batch(mlm, w, rng, 2)
    l_sparse, _ = mlm.loss(b, sparse_logits=True)
    l_full, _ = mlm.loss(b, sparse_logits=False)
    assert torch.allclose(l_sparse, l_full, atol=1e-5), (l_sparse, l_full)
