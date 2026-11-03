# GPU kernels and measured costs

[Documentation](../index.md) / [Explanation](index.md)

The first H100 profile spent 184 ms of wall time on 22 ms of GPU work across 7,042 launches.
The torch scratchpad recurrence launched about 8 kernels per token in the forward pass and 16 in backward;
copy-head matching added about ten launches per name-length step. `kgmem/kernels.py` fuses those loops.

`delta_rule` computes `r = Sᵀu_r; v = v_h + W r; S += β u_w (v − Sᵀu_w)ᵀ` for a sequence in one launch.
Each block handles one sequence, with thread j keeping column j of S, 64 values, in registers.
The dot products are thread-local and reads of the read-to-value map W are coalesced. Backward stores the
pre-update states and write errors, walks tokens in reverse and reduces columns with warp shuffles.
One matmul computes dW afterward.

`pointer_match` and `carried_match` compare name prefixes, with one thread per (sequence, position,
candidate) or (sequence, position). They return indices and masks. Torch still gathers the corresponding
weights, preserving their gradients.

| H100, SmolLM2-135M, batch 8 | torch | fused |
|---|---|---|
| scratchpad recurrence forward, 62-token batch | 43.1 ms | 4.7 ms |
| copy-head matching, one graph training step | 7.6 ms | 1.7 ms |
| training step, graph batch | 323 ms | 175 ms |
| training step, scratchpad batch | 181.5 ms | 104 ms |
| `generate`, 16 tokens | 694 ms | 628 ms |

The frozen LM's three forward passes account for most of the remaining training cost; its per-token
work dominates generation. The v9 stages ran at 0.19–0.22 s/step on the H100. The
[CPU recipes](../how-to/train-adapters.md) ran at 0.3–0.8 s/step on a 14-core Mac.

`torch.utils.cpp_extension.load_inline` builds the kernels once with nvcc and caches them.
The torch implementation remains the reference and handles CPU runs, `decay`, and `KGMEM_KERNELS=0`.
`tests/test_kernels.py` compares outputs and every gradient, including shapes that do not fill a warp.
Program-backed relations use strandedcpu's own CUDA kernel through
`strandedcpu_graph(..., backend="auto")`; see [computed relations](strandedcpu.md).
