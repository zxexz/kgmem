"""CUDA implementations of the delta rule and copy-name matching.

The no-decay scratchpad recurrence is:

    r_t = S^T ur_t
    v_t = vh_t + W r_t
    e_t = v_t - S^T uw_t
    S += beta_t uw_t e_t^T

One CUDA block handles a sequence, holding columns of S in registers. Training
stores pre-step states and errors for the reverse-time backward pass; dW is
computed afterward by matmul. pointer_match and carried_match return matching
indices and masks; torch gathers the weights to retain their gradients.

load_inline compiles and caches the extension. Callers fall back to torch when
CUDA, the build, or a supported shape is unavailable (n in N_SUPPORTED,
d <= 1024, fp32). Set KGMEM_KERNELS=0 to disable the extension.
"""
from __future__ import annotations

import os

import torch

N_SUPPORTED = (8, 16, 32, 64, 128)
MAX_STATE_FLOATS = 1 << 28                   # Cap saved backward states at 1 GiB; larger batches use the torch recurrence.

_CPP = """
#include <torch/extension.h>
std::vector<torch::Tensor> delta_fwd(torch::Tensor ur, torch::Tensor uw, torch::Tensor vh, torch::Tensor beta,
                                     torch::Tensor wt, torch::Tensor s0, bool save);
std::vector<torch::Tensor> delta_bwd(torch::Tensor ur, torch::Tensor uw, torch::Tensor beta, torch::Tensor w,
                                     torch::Tensor states, torch::Tensor err, torch::Tensor g_reads,
                                     torch::Tensor g_st);
std::vector<torch::Tensor> pointer_match(torch::Tensor ids, torch::Tensor cand_tok, torch::Tensor cand_len);
std::vector<torch::Tensor> carried_match(torch::Tensor ids, torch::Tensor hist_ids, torch::Tensor hist_w,
                                         torch::Tensor name_tok, torch::Tensor name_len, int64_t Tn, double share,
                                         double top);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float warp_sum(float x) {
  for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
  return x;
}

// One block per sequence, blockDim = D rounded up to a warp; thread j < D owns column j of S.
template <int N>
__global__ void delta_fwd_k(const float* __restrict__ ur, const float* __restrict__ uw, const float* __restrict__ vh,
                            const float* __restrict__ beta, const float* __restrict__ wt, const float* __restrict__ s0,
                            float* __restrict__ reads, float* __restrict__ st, float* __restrict__ states,
                            float* __restrict__ err, int T, int D, int save) {
  extern __shared__ float sm[];
  float* sur = sm; float* suw = sm + N; float* sr = sm + 2 * N;
  const int b = blockIdx.x, j = threadIdx.x;
  const bool act = j < D;
  float S[N];
#pragma unroll
  for (int i = 0; i < N; ++i) S[i] = act ? s0[((size_t)b * N + i) * D + j] : 0.f;
  for (int t = 0; t < T; ++t) {
    const size_t bt = (size_t)b * T + t;
    for (int i = j; i < N; i += blockDim.x) { sur[i] = ur[bt * N + i]; suw[i] = uw[bt * N + i]; }
    __syncthreads();
    float r = 0.f, p = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i) { r = fmaf(sur[i], S[i], r); p = fmaf(suw[i], S[i], p); }
    if (act) {
      if (save) {
#pragma unroll
        for (int i = 0; i < N; ++i) states[(bt * N + i) * D + j] = S[i];
      }
      sr[j] = r;
      reads[bt * D + j] = r;
    }
    __syncthreads();
    if (act) {
      float v = vh[bt * D + j];
      for (int k = 0; k < D; ++k) v = fmaf(wt[(size_t)k * D + j], sr[k], v);     // (W r)_j, W^T row k coalesced
      const float e = v - p;
      if (save) err[bt * D + j] = e;
      const float bb = beta[bt];
#pragma unroll
      for (int i = 0; i < N; ++i) S[i] = fmaf(bb * suw[i], e, S[i]);
    }
    __syncthreads();                                                               // smem is reused next step
  }
  if (act) {
#pragma unroll
    for (int i = 0; i < N; ++i) st[((size_t)b * N + i) * D + j] = S[i];
  }
}

// Reverse walk. dS is the gradient w.r.t. the state after step t; S (loaded) is the state before it.
template <int N>
__global__ void delta_bwd_k(const float* __restrict__ ur, const float* __restrict__ uw, const float* __restrict__ beta,
                            const float* __restrict__ w, const float* __restrict__ states, const float* __restrict__ err,
                            const float* __restrict__ g_reads, const float* __restrict__ g_st,
                            float* __restrict__ g_ur, float* __restrict__ g_uw, float* __restrict__ g_vh,
                            float* __restrict__ g_beta, float* __restrict__ g_s0, int T, int D) {
  extern __shared__ float sm[];
  const int nw = blockDim.x / 32;
  float* sur = sm; float* suw = sm + N; float* sdv = sm + 2 * N; float* red = sdv + blockDim.x;
  float* sred = red + 2 * N * nw;
  const int b = blockIdx.x, j = threadIdx.x, lane = j & 31, warp = j >> 5;
  const bool act = j < D;
  float dS[N];
#pragma unroll
  for (int i = 0; i < N; ++i) dS[i] = act ? g_st[((size_t)b * N + i) * D + j] : 0.f;
  for (int t = T - 1; t >= 0; --t) {
    const size_t bt = (size_t)b * T + t;
    for (int i = j; i < N; i += blockDim.x) { sur[i] = ur[bt * N + i]; suw[i] = uw[bt * N + i]; }
    __syncthreads();
    const float bb = beta[bt];
    float S[N];
#pragma unroll
    for (int i = 0; i < N; ++i) S[i] = act ? states[(bt * N + i) * D + j] : 0.f;
    const float e = act ? err[bt * D + j] : 0.f;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i) q = fmaf(suw[i], dS[i], q);
    const float de = bb * q;                                                       // = dv = dvh = -dp
    if (act) g_vh[bt * D + j] = de;
    sdv[j] = act ? de : 0.f;
    __syncthreads();
    float dr = 0.f;
    if (act) {
      dr = g_reads[bt * D + j];
      for (int k = 0; k < D; ++k) dr = fmaf(w[(size_t)k * D + j], sdv[k], dr);     // (W^T dv)_j, W row k coalesced
    }
    const float dp = -de;
    // column reductions: d uw_i = sum_j (b dS_ij e_j + S_ij dp_j), d ur_i = sum_j S_ij dr_j, d beta = sum_j q_j e_j
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const float a = warp_sum(fmaf(bb * dS[i], e, S[i] * dp));
      const float c = warp_sum(S[i] * dr);
      if (lane == 0) { red[warp * 2 * N + i] = a; red[warp * 2 * N + N + i] = c; }
    }
    const float qb = warp_sum(q * e);
    if (lane == 0) sred[warp] = qb;
#pragma unroll
    for (int i = 0; i < N; ++i) dS[i] = fmaf(sur[i], dr, fmaf(suw[i], dp, dS[i]));
    __syncthreads();
    for (int i = j; i < N; i += blockDim.x) {
      float a = 0.f, c = 0.f;
      for (int k = 0; k < nw; ++k) { a += red[k * 2 * N + i]; c += red[k * 2 * N + N + i]; }
      g_uw[bt * N + i] = a; g_ur[bt * N + i] = c;
    }
    if (j == 0) { float s = 0.f; for (int k = 0; k < nw; ++k) s += sred[k]; g_beta[bt] = s; }
    __syncthreads();
  }
  if (act) {
#pragma unroll
    for (int i = 0; i < N; ++i) g_s0[((size_t)b * N + i) * D + j] = dS[i];
  }
}

#define DISPATCH_N(N_, ...) switch (N_) { \
  case 8: { constexpr int NN = 8; __VA_ARGS__; break; } \
  case 16: { constexpr int NN = 16; __VA_ARGS__; break; } \
  case 32: { constexpr int NN = 32; __VA_ARGS__; break; } \
  case 64: { constexpr int NN = 64; __VA_ARGS__; break; } \
  case 128: { constexpr int NN = 128; __VA_ARGS__; break; } \
  default: TORCH_CHECK(false, "delta_rule: unsupported n ", N_); }

std::vector<torch::Tensor> delta_fwd(torch::Tensor ur, torch::Tensor uw, torch::Tensor vh, torch::Tensor beta,
                                     torch::Tensor wt, torch::Tensor s0, bool save) {
  const int B = ur.size(0), T = ur.size(1), N = ur.size(2), D = vh.size(2);
  auto o = vh.options();
  auto reads = torch::empty({B, T, D}, o), st = torch::empty({B, N, D}, o);
  auto states = save ? torch::empty({B, T, N, D}, o) : torch::empty({0}, o);
  auto err = save ? torch::empty({B, T, D}, o) : torch::empty({0}, o);
  const int threads = (D + 31) / 32 * 32;
  const size_t smem = (2 * N + threads) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_N(N, delta_fwd_k<NN><<<B, threads, smem, stream>>>(
      ur.data_ptr<float>(), uw.data_ptr<float>(), vh.data_ptr<float>(), beta.data_ptr<float>(), wt.data_ptr<float>(),
      s0.data_ptr<float>(), reads.data_ptr<float>(), st.data_ptr<float>(), save ? states.data_ptr<float>() : nullptr,
      save ? err.data_ptr<float>() : nullptr, T, D, save ? 1 : 0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {reads, st, states, err};
}

std::vector<torch::Tensor> delta_bwd(torch::Tensor ur, torch::Tensor uw, torch::Tensor beta, torch::Tensor w,
                                     torch::Tensor states, torch::Tensor err, torch::Tensor g_reads,
                                     torch::Tensor g_st) {
  const int B = ur.size(0), T = ur.size(1), N = ur.size(2), D = err.size(2);
  auto o = err.options();
  auto g_ur = torch::empty({B, T, N}, o), g_uw = torch::empty({B, T, N}, o), g_vh = torch::empty({B, T, D}, o);
  auto g_beta = torch::empty({B, T}, o), g_s0 = torch::empty({B, N, D}, o);
  const int threads = (D + 31) / 32 * 32, nw = threads / 32;
  const size_t smem = (2 * N + threads + 2 * N * nw + nw) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_N(N, delta_bwd_k<NN><<<B, threads, smem, stream>>>(
      ur.data_ptr<float>(), uw.data_ptr<float>(), beta.data_ptr<float>(), w.data_ptr<float>(),
      states.data_ptr<float>(), err.data_ptr<float>(), g_reads.data_ptr<float>(), g_st.data_ptr<float>(),
      g_ur.data_ptr<float>(), g_uw.data_ptr<float>(), g_vh.data_ptr<float>(), g_beta.data_ptr<float>(),
      g_s0.data_ptr<float>(), T, D));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {g_ur, g_uw, g_vh, g_beta, g_s0};
}

// Copy head: for candidate c at position pos, the longest j < len with ids[pos-j+1..pos] == name[:j] proposes
// name[j]; done if the whole name was just emitted. One thread per (b, i, c).
__global__ void pointer_match_k(const int64_t* __restrict__ ids, const int64_t* __restrict__ ct,
                                const int64_t* __restrict__ cl, int64_t* __restrict__ nxt, bool* __restrict__ done,
                                bool* __restrict__ partial, int B, int Th, int Tn, int C, int L) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)B * Tn * C) return;
  const int b = idx / ((long)Tn * C), i = (idx / C) % Tn;
  const int pos = Th - Tn + i;
  const int64_t* name = ct + idx * L;
  const int64_t len = cl[idx];
  const int64_t* row = ids + (long)b * Th;
  int best = 0; bool dn = false;
  for (int j = 1; j <= L; ++j) {
    const int start = pos - j + 1;
    if (start < 0) break;                                         // the history is shorter than j
    bool ok = true;
    for (int k = 0; k < j && ok; ++k) ok = name[k] == row[start + k];
    if (ok && j < len) best = j;
    if (ok && j == len) dn = true;
  }
  nxt[idx] = name[best < L ? best : L - 1];
  done[idx] = dn; partial[idx] = best > 0;
}

// Carry-over: for lag j, position pos-j's candidates whose first j tokens equal the j tokens emitted since.
// One thread per (b, i): proposals and masks for every (j, c), and whether a retrieved name was just finished.
__global__ void carried_match_k(const int64_t* __restrict__ ids, const int64_t* __restrict__ hid,
                                const float* __restrict__ hw, const int64_t* __restrict__ nt,
                                const int64_t* __restrict__ nl, int64_t* __restrict__ props, bool* __restrict__ match,
                                bool* __restrict__ any_m, bool* __restrict__ completed, int B, int Th, int Tn, int C,
                                int L, float share, float top) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)B * Tn) return;
  const int b = idx / Tn, i = idx % Tn, pos = Th - Tn + i;
  const int64_t* row = ids + (long)b * Th;
  bool comp = false, any = false;
  for (int j = 1; j <= L; ++j) {
    const int src = pos - j;
    const long out = (idx * (L - 1) + (j - 1)) * C;               // (b, i, j-1, c) for j < L
    if (src < 0) {
      if (j < L) for (int c = 0; c < C; ++c) { props[out + c] = 0; match[out + c] = false; }
      continue;
    }
    const int64_t* cid = hid + ((long)b * Th + src) * C;
    const float* cw = hw + ((long)b * Th + src) * C;
    float total = 0.f, best = 0.f;
    if (share > 0.f) for (int c = 0; c < C; ++c) total += cw[c];
    if (top > 0.f) for (int c = 0; c < C; ++c) best = fmaxf(best, cw[c]);
    float comp_w = 0.f, cont_w = 0.f; bool any_comp = false;
    for (int c = 0; c < C; ++c) {
      const int64_t* tok = nt + cid[c] * L;
      const int64_t ln = nl[cid[c]];
      const bool strong = top > 0.f ? (cw[c] >= top * best && cw[c] > 0.f)
                        : share > 0.f ? cw[c] > share * total : cw[c] > 0.05f;
      bool full = strong;
      for (int k = 0; k < j && full; ++k) full = tok[k] == row[src + 1 + k];
      if (full && j == ln) { comp_w += cw[c]; any_comp = true; }
      const bool m = full && j < ln;
      if (m) cont_w += cw[c];
      if (j < L) {
        props[out + c] = tok[j] < 0 ? 0 : tok[j];
        match[out + c] = m;
        any = any || m;
      }
    }
    comp = comp || (share > 0.f || top > 0.f ? comp_w > cont_w : any_comp);
  }
  any_m[idx] = any; completed[idx] = comp;
}

std::vector<torch::Tensor> pointer_match(torch::Tensor ids, torch::Tensor cand_tok, torch::Tensor cand_len) {
  const int B = cand_tok.size(0), Tn = cand_tok.size(1), C = cand_tok.size(2), L = cand_tok.size(3);
  const int Th = ids.size(1);
  auto nxt = torch::empty({B, Tn, C}, ids.options());
  auto bo = ids.options().dtype(torch::kBool);
  auto done = torch::empty({B, Tn, C}, bo), partial = torch::empty({B, Tn, C}, bo);
  const long n = (long)B * Tn * C;
  if (n > 0) {
    pointer_match_k<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int64_t>(), cand_tok.data_ptr<int64_t>(), cand_len.data_ptr<int64_t>(), nxt.data_ptr<int64_t>(),
        done.data_ptr<bool>(), partial.data_ptr<bool>(), B, Th, Tn, C, L);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {nxt, done, partial};
}

std::vector<torch::Tensor> carried_match(torch::Tensor ids, torch::Tensor hist_ids, torch::Tensor hist_w,
                                         torch::Tensor name_tok, torch::Tensor name_len, int64_t Tn, double share,
                                         double top) {
  const int B = ids.size(0), Th = ids.size(1), C = hist_ids.size(2), L = name_tok.size(1);
  auto props = torch::empty({B, Tn, L - 1, C}, ids.options());
  auto bo = ids.options().dtype(torch::kBool);
  auto match = torch::empty({B, Tn, L - 1, C}, bo);
  auto any_m = torch::empty({B, Tn}, bo), completed = torch::empty({B, Tn}, bo);
  const long n = (long)B * Tn;
  if (n > 0) {
    carried_match_k<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int64_t>(), hist_ids.data_ptr<int64_t>(), hist_w.data_ptr<float>(), name_tok.data_ptr<int64_t>(),
        name_len.data_ptr<int64_t>(), props.data_ptr<int64_t>(), match.data_ptr<bool>(), any_m.data_ptr<bool>(),
        completed.data_ptr<bool>(), B, Th, (int)Tn, C, L, (float)share, (float)top);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {props, match, any_m, completed};
}
"""

_EXT = None
_ERROR: str | None = None


def _ext():
    """Return the compiled extension, or None; _ERROR records build failures."""
    global _EXT, _ERROR
    if _EXT is None and _ERROR is None:
        if os.environ.get("KGMEM_KERNELS", "1") == "0":
            _ERROR = "KGMEM_KERNELS=0"
        elif not torch.cuda.is_available():
            _ERROR = "no CUDA device"
        else:
            try:
                from torch.utils.cpp_extension import load_inline
                _EXT = load_inline("kgmem_kernels_v3", cpp_sources=_CPP, cuda_sources=_CUDA,
                                   functions=["delta_fwd", "delta_bwd", "pointer_match", "carried_match"],
                                   extra_cuda_cflags=["-O3"], verbose=False)
            except Exception as exc:                                    # no nvcc / ninja, or a compile error
                _ERROR = f"{type(exc).__name__}: {str(exc)[:300]}"
    return _EXT


def delta_rule_ok(x: torch.Tensor, n: int, d: int, B: int, T: int, needs_grad: bool) -> bool:
    if not (x.is_cuda and x.dtype == torch.float32 and n in N_SUPPORTED and d <= 1024 and T > 0):
        return False
    if needs_grad and B * T * n * d > MAX_STATE_FLOATS:
        return False
    return _ext() is not None


class _DeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ur, uw, vh, beta, w, s0):
        save = any(ctx.needs_input_grad)
        reads, st, states, err = _ext().delta_fwd(ur, uw, vh, beta, w.t().contiguous(), s0, save)
        if save:
            ctx.save_for_backward(ur, uw, beta, w, states, err, reads)
        return reads, st

    @staticmethod
    def backward(ctx, g_reads, g_st):
        ur, uw, beta, w, states, err, reads = ctx.saved_tensors
        g_reads = torch.zeros_like(reads) if g_reads is None else g_reads.contiguous()
        g_st = torch.zeros(ur.shape[0], ur.shape[2], reads.shape[2], device=reads.device) if g_st is None \
            else g_st.contiguous()
        g_ur, g_uw, g_vh, g_beta, g_s0 = _ext().delta_bwd(ur, uw, beta, w.contiguous(), states, err, g_reads, g_st)
        D = reads.shape[-1]
        g_w = g_vh.reshape(-1, D).t() @ reads.reshape(-1, D)          # dW_jk = sum_bt dv_j r_k
        return g_ur, g_uw, g_vh, g_beta, g_w, g_s0


def delta_rule(ur, uw, vh, beta, w, s0):
    """Run the no-decay delta recurrence on float32 CUDA tensors.

    ur and uw: (B, T, n); vh: (B, T, d); beta: (B, T); w: (d, d);
    s0: (B, n, d). Return reads (B, T, d) and final state (B, n, d).
    The value update uses v = vh + r @ w.T.
    """
    c = lambda t: t.float().contiguous()
    return _DeltaRule.apply(c(ur), c(uw), c(vh), c(beta), w.float(), c(s0))


def matching_ok(ids: torch.Tensor) -> bool:
    return ids.is_cuda and _ext() is not None


def pointer_match(ids, cand_tok, cand_len):
    """Return next tokens, completion flags, and partial-match flags on CUDA."""
    return tuple(_ext().pointer_match(ids.long().contiguous(), cand_tok.long().contiguous(),
                                      cand_len.long().contiguous()))


def carried_match(ids, hist_ids, hist_w, name_tok, name_len, Tn: int, share: float, top: float = 0.0):
    """Match carried candidates on CUDA and gather their weights in torch.

    Return proposals, weights, partial matches, and completions in the reference
    implementation's layout. The gather preserves gradients to hist_w.
    """
    B, Th = ids.shape
    L, C = name_tok.shape[1], hist_ids.shape[-1]
    props, match, any_m, completed = _ext().carried_match(ids.long().contiguous(), hist_ids.long().contiguous(),
                                                          hist_w.detach().float().contiguous(),
                                                          name_tok.long().contiguous(), name_len.long().contiguous(),
                                                          int(Tn), float(share), float(top))
    pos = torch.arange(Th - Tn, Th, device=ids.device)
    src = pos[:, None] - torch.arange(1, L, device=ids.device)[None, :]               # (Tn, L-1)
    cw = hist_w[:, src.clamp(min=0)] * (src >= 0)[None, :, :, None]                      # (B, Tn, L-1, C)
    return props.reshape(B, Tn, -1), (cw * match).reshape(B, Tn, -1), any_m, completed
