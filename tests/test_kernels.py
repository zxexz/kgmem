"""Compare CUDA outputs and gradients with torch reference implementations; skip without CUDA."""
import pytest
import torch

from kgmem import MemConfig, kernels
from kgmem.modules import DeltaScratchpad

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def close(a, b, rel=2e-5):
    """Compare with a tolerance relative to the tensor's largest magnitude.

    Different float32 reduction orders can change summed gradients by about 1e-6
    of that scale; elementwise relative tolerance is too strict near zero.
    """
    return (a - b).abs().max().item() <= rel * a.abs().max().item() + 1e-7


def _run(pad, h, S0, use_kernel, monkeypatch):
    monkeypatch.setattr(kernels, "delta_rule_ok", (lambda *a: kernels._ext() is not None) if use_kernel
                        else (lambda *a: False))
    for p in pad.parameters():
        p.grad = None
    h = h.detach().clone().requires_grad_(True)
    S0 = S0.detach().clone().requires_grad_(True)
    reads, S, _ = pad(h, S0)
    g = torch.Generator(device="cuda").manual_seed(1)
    wr, ws = torch.randn(reads.shape, generator=g, device="cuda"), torch.randn(S.shape, generator=g, device="cuda")
    ((reads * wr).sum() + (S * ws).sum()).backward()
    return reads, S, h.grad, S0.grad, {n: p.grad.clone() for n, p in pad.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("n,d,T,B", [(64, 256, 37, 3), (16, 48, 9, 2), (32, 100, 1, 4), (8, 32, 70, 1)])
def test_delta_rule_kernel_matches_the_torch_loop(n, d, T, B, monkeypatch):
    assert kernels._ext() is not None, kernels._ERROR
    torch.manual_seed(0)
    cfg = MemConfig(n_slots=n, d_val=d, d_att=32, cont_keys=True, l2_keys=True)
    pad = DeltaScratchpad(96, cfg).cuda()
    with torch.no_grad():
        pad.v_r.weight.normal_(std=0.05)                 # non-zero, as in trained adapters
        pad.beta.bias.fill_(0.0)                         # writes happen
    h = torch.randn(B, T, 96, device="cuda")
    S0 = torch.randn(B, n, d, device="cuda") * 0.1
    ref = _run(pad, h, S0, False, monkeypatch)
    got = _run(pad, h, S0, True, monkeypatch)
    for name, a, b in zip(("reads", "S", "dh", "dS0"), ref[:4], got[:4]):
        assert close(a, b), (name, (a - b).abs().max().item())
    assert ref[4].keys() == got[4].keys()
    for k in ref[4]:
        assert close(ref[4][k], got[4][k]), (k, (ref[4][k] - got[4][k]).abs().max().item())


def test_delta_rule_kernel_without_grad_and_from_zero_state(monkeypatch):
    torch.manual_seed(1)
    cfg = MemConfig(n_slots=64, d_val=256, d_att=32, cont_keys=True, l2_keys=True)
    pad = DeltaScratchpad(96, cfg).cuda()
    h, S0 = torch.randn(2, 20, 96, device="cuda"), pad.init_state(2, "cuda")
    with torch.no_grad():
        monkeypatch.setattr(kernels, "delta_rule_ok", lambda *a: False)
        r_ref, S_ref, _ = pad(h, S0)
        monkeypatch.setattr(kernels, "delta_rule_ok", lambda *a: kernels._ext() is not None)
        r, S, _ = pad(h, S0)
    assert close(r_ref, r) and close(S_ref, S)


@pytest.mark.parametrize("share,top", [(0.0, 0.0), (0.2, 0.0), (0.0, 0.5)])
def test_copy_head_matching_kernels_match_torch(share, top, monkeypatch):
    from kgmem import modules
    g = torch.Generator().manual_seed(3)
    E, L, B, Th, Tn, C = 40, 5, 3, 30, 12, 7
    name_len = torch.randint(1, L + 1, (E,), generator=g)
    name_tok = torch.randint(0, 4, (E, L), generator=g)                  # 4-token vocab: many shared prefixes
    name_tok[torch.arange(L)[None, :] >= name_len[:, None]] = -1
    ids = torch.randint(0, 4, (B, Th), generator=g)
    hist_ids = torch.randint(0, E, (B, Th, C), generator=g)
    hist_w = torch.rand(B, Th, C, generator=g) ** 3 * 0.2               # many under 0.05, as in trained adapters
    cand_ids = hist_ids[:, -Tn:]
    to = lambda *ts: [t.cuda() for t in ts]
    ids, name_tok, name_len, hist_ids, hist_w, cand_ids = to(ids, name_tok, name_len, hist_ids, hist_w, cand_ids)
    monkeypatch.setattr(kernels, "matching_ok", lambda ids: False)
    ref_p = modules.pointer_next_tokens(ids, name_tok[cand_ids], name_len[cand_ids])
    ref_c = modules.carried_candidates(ids, hist_ids, hist_w, name_tok, name_len, Tn, share, top)
    monkeypatch.setattr(kernels, "matching_ok", lambda ids: kernels._ext() is not None)
    got_p = modules.pointer_next_tokens(ids, name_tok[cand_ids], name_len[cand_ids])
    got_c = modules.carried_candidates(ids, hist_ids, hist_w, name_tok, name_len, Tn, share, top)
    for a, b in zip(ref_p, got_p):
        assert torch.equal(a, b)
    (rp, rw, ra, rc), (gp, gw, ga, gc) = ref_c, got_c
    assert torch.equal(rw, gw) and torch.equal(ra, ga) and torch.equal(rc, gc)
    assert rc.any() and ra.any()                                         # the case exercises both
    assert torch.equal(rp[rw > 0], gp[gw > 0])                           # proposals wherever they carry weight
