# Vulkan backend: current state, learnings, and gaps

This document summarizes what we have learned while working on the Vulkan backend of
ik_llama.cpp, what has been fixed, how to get good performance, and what is still missing.

## TL;DR

- The Vulkan backend is **correct** for the ops it supports, but its **op/type coverage is
  far behind the CUDA backend and behind ik_llama's own graph builder**.
- The single biggest performance trap: ik_llama fuses the FFN `up`+`gate` matmuls into
  `GGML_OP_FUSED_UP_GATE` (and MoE models into `MOE_FUSED_UP_GATE`) **by default**. The Vulkan
  backend now implements both ops (as two matmuls into the destination and a temporary buffer,
  followed by the fused activation combine), so dense and MoE FFNs run on the GPU without the
  `-no-fug` / `-no-fmoe` workarounds.
- The imatrix quant types from the IQK/K-family (`IQ2_K`...`IQ6_K`, `IQ4_KS`, `IQ2_KS`,
  `IQ4_KSS`, `IQ5_KS`, `IQ3_KS`, `IQ2_KL`) and the KT-family (`IQ1_KT`...`IQ4_KT`) are **now
  supported** by the Vulkan backend (see "What we fixed"): single-token decode runs a native
  `mul_mat_vec` kernel per type and prompt processing runs a native dequant-to-F16 step followed
  by the F16 matmul. The `*_R4` repack variants and `Q6_0`, `MXFP4`, `IQ1_BN`, `IQ2_BN` are
  still not supported and fall back to the CPU backend.

## What we fixed

### 0. IQK / KT quant families (QK_K = 256 imatrix quants)

All 15 base IQK/KT types are now supported by `MUL_MAT`, `MUL_MAT_ID` (vec and mat-mat paths)
and `GET_ROWS`:

- **decode (mul_mat_vec)**: a dedicated GLSL kernel per type (`mul_mat_vec_iq{1,2,3,4,5,6}_k*`,
  `mul_mat_vec_iq{2,3,4,5}_ks*`, `mul_mat_vec_iq4_kss*`, `mul_mat_vec_iq2_kl*`,
  `mul_mat_vec_iq{1,2,3,4}_kt*`) that reads the quantized bytes with explicit row/block byte
  addressing (the KS/KL/KT formats carry a per-row scale header, `row_meta_size` 2 or 4) and
  dots against the F32/F16 activations.
- **prompt processing (mul_mat)**: a dequant-to-F16 kernel per type (`dequant_iqX_*`) feeding the
  existing F16 matmul. The flat dequant shaders are row-meta aware (blocks-per-row and meta size
  are passed in the push constants) and the dispatch passes the tensor's full byte size (the
  per-row scale headers must be included in the source subbuffer).
- **GET_ROWS (quantized token embeddings)**: a byte-addressed `get_rows_iqk.comp` shader
  dequantizes one element per thread (1024 per workgroup, matching the pipeline's
  elements-per-workgroup denom). The per-row scale header is read from the explicit row byte
  offset (only the first block's header sits at `block_byte - row_meta_size`).
- The KT-family value decode (`QuantizerIQKT::set_values`, the multiplicative-hash lookup into
  `iq4k_values`) is implemented in `iqk_hash_values` in `iqk_tables.comp`.
- `supports_op` accepts the 15 types for `MUL_MAT`, `MUL_MAT_ID`, `GET_ROWS` and
  `FUSED_UP_GATE`/`MOE_FUSED_UP_GATE`; `ggml_vk_dim01_contiguous` accounts for `row_meta_size`.

Note on the reference: the CPU backend's own `to_float` and its optimized `iqk_mul_mat` kernels
use slightly different scale conventions for several of these experimental types (e.g. the
KT-family calibration factors 1.01/1.05 appear in the vec_dot kernels but not in `to_float`),
and the CPU quantizer for the meta types has uninitialized-memory reads that depend on heap
state. The Vulkan kernels follow the CUDA vec_dot convention (the established GPU reference).
`tests/test-iqk-quants.cpp` validates against the scalar dequant; the KT family is allowed a
looser tolerance because of the calibration factor.

### 1. Fused up-gate / MoE fused up-gate (`-fug` / `-fmoe`)

`GGML_OP_FUSED_UP_GATE` and `GGML_OP_MOE_FUSED_UP_GATE` are now implemented:

- Dense: two `MUL_MAT`s (gate into the destination, up into a temp buffer) + the fused
  activation combine (`silu/gelu/relu/swiglu_oai`).
- MoE with separate up/gate weights: two `MUL_MAT_ID`s + per-expert bias adds (via a small
  `add_bias_id` kernel that gathers the bias by expert id) + the combine.
- MoE with fused gate+up weights (`as_gate == nullptr`, first half of the rows are gate,
  second half up): one `MUL_MAT_ID` of the full `[k, 2*n_ff, n_expert]` matrix into a temp
  buffer, optional bias adds on each half, then a split combine kernel that reads both halves.
- The activation "limit" (`op_params[1]`, step35/deepseek4 style models) is honored: the
  existing `fused_mul_*` shaders clamp when the limit is non-zero, and a new
  `fused_mul_swiglu_oai` shader implements the OAI variant.
- `supports_op` accepts the ops when the weight type is one the matmul path supports (F32/F16/
  BF16 and the supported quant types, including the 15 IQK/KT types), the activations are F32
  and the unary op is one of the four supported ones; anything else falls back to the CPU
  backend as before. (The IQK/KT types were originally missing from this switch, which made
  every FFN of an IQK/KT model fall back to the CPU — see "Fixed: large IQK/KT models were
  slow on Vulkan" below.)

A work-size bug in `ggml_graph_plan` was fixed along the way: the `MOE_FUSED_UP_GATE` plan
skipped the quantized-activation work buffer when the gate weights are fused (`src[1] == NULL`),
which made the CPU reference compute overflow its work buffer (heap corruption). The dense
`FUSED_UP_GATE` plan also sized the work buffer from the weights instead of the activations.
Both now size it from `src[2]` (the activations).

### 2. Synchronization: one wait per graph instead of per batch

The backend waited on a fence at the end of every submitted batch of nodes, using a CPU
**spin** (`getFenceStatus` polling with `_mm_pause`). This serialized CPU command recording
with GPU execution and pegged a CPU core while waiting.

- Batches are now submitted without a fence so command recording overlaps GPU execution
  within a graph (the "almost ready" fence is still used for the last ~20%).
- The GPU is waited for **once at the end of each `graph_compute`** (an empty submission on
  the same queue signals the fence when all queued work is done), before the command pools
  are reset.
- The backend `.synchronize` is now enabled in the interface so the scheduler can flush
  pending work, and a `submit_pending` flag ensures the command pools are only reset after
  the GPU has finished.

Why wait per `graph_compute` and not once per decode (as upstream llama.cpp does)? This
backend has no synchronization between its transfer and compute queues (the timeline
semaphore scaffolding is unused), so host reads and cross-backend copies performed by the
scheduler between splits can race with in-flight compute. Waiting per split keeps the
intra-graph overlap while guaranteeing the scheduler sees completed results. This was a
correctness bug in an earlier iteration (fire-and-forget across splits produced garbage
output).

### 3. Fence mix-up found while debugging a hang

`ggml_vk_wait_for_fence()` spins on `ctx->fence`, but the graph finalization had been
submitting an empty batch with `ctx->device->fence` — a *different* fence object. The
result was an infinite spin with an idle GPU. The fix is to use `ctx->fence` consistently.

### 4. Cooperative-matrix assert on coopmat2 GPUs

`GGML_ASSERT((GGML_KQ_MASK_PAD % rows_cols[0]) == 0)` in the flash-attention shader setup
crashed on any GPU with `NV_coopmat2` support (e.g. the RTX 3090), in both static and
dynamic builds. The coopmat2 shaders use clamped tensor layouts and handle the mask padding
explicitly, so the assert is now relaxed for row granularities larger than
`GGML_KQ_MASK_PAD`.

### 5. Integrated GPUs are enumerated

The instance init only added discrete GPUs to `vk_instance.device_indices`. Integrated GPUs
(e.g. an AMD Strix Halo iGPU) are now included as well, matching upstream llama.cpp. This
makes a device like the AMD Radeon 8060S show up as `Vulkan1` and selectable via
`-dev Vulkan1` without `GGML_VK_VISIBLE_DEVICES`.

### 6. `-dev`/`--device` integration

`-dev CUDA0`, `-dev Vulkan1`, `-dev CUDA0,Vulkan1` etc. select devices from the ggml
backend registry. See `docs/build.md` and the `-dev` commit for details.

## Benchmarks (RTX 3090, Vulkan0)

Qwen2.5-Coder-0.5B-Instruct-Q8_0 (dense, `-c 2048`, single token batch):

| configuration | tokens/s |
|---|---|
| before the fused up-gate support (FFN on CPU) | ~17 |
| fused up-gate on Vulkan (default) | ~420 |
| `-no-fug` | ~430 |
| CUDA backend, same model | ~600 |

The GPU kernels themselves are fast (single-digit to tens of µs per op); the wall-clock
cost before was the CPU-fallback splits and their data movement.

The fused up-gate result is bit-identical to the `-no-fug` path on Vulkan (both use the same
matmul pipelines; only the kernel dispatch differs), and `tests/test-fused-up-gate` checks
this for dense and MoE (fused and separate weights, with and without per-expert biases) across
several quant types and batch sizes.

## What we learned (architecture notes)

- The op kernels are dispatched per node with a dry-run pass (descriptor-set budgeting) and
  a record pass per `graph_compute`. The dry run is cheap after the pipelines are compiled;
  the first `graph_compute` is slow because it compiles shaders (`need_compiles`).
- The batch batching logic (`mul_mat_bytes_per_submit`, `nodes_per_submit`) already aimed
  at CPU/GPU overlap; it was defeated by the per-batch fence wait.
- The transfer/compute queue pair exists (`single_queue`, `transfer_queue`) but the async
  interfaces (`set_tensor_async`, `get_tensor_async`, `cpy_tensor_async`, events) are all
  `NULL` in the backend interface, and the submission `wait_semaphores`/`signal_semaphores`
  are never populated. Any work that relies on cross-queue ordering must be guarded by a
  fence wait.
- The scheduler splits the decode graph into many small splits (one `graph_compute` call
  each); anything slow in `graph_compute` is multiplied by the split count.
- Intel cooperative-matrix policy: this fork only enables coopmat on Xe2 (SIMD16) GPUs.
  Upstream additionally allows Xe1 integrated GPUs with the Intel proprietary Windows
  driver. AMD RADV is always allowed (RDNA3+ hardware permitting).

## Remaining gaps

### Op coverage (the big one)

These ops are produced by ik_llama's graph builder but are **not implemented** in the
Vulkan backend, so they fall back to the CPU backend with expensive copies:

- **Gated delta-net** (`qwen35` / `qwen3next` recurrent layers; e.g. Qwen3.5-0.8B): the
  recurrent layer builds `GGML_OP_SSM_CONV` (causal conv over the qkv projection, with a
  per-sequence conv state and per-step state save), `GGML_OP_L2_NORM` (q/k normalization;
  the `l2_norm.comp` shader exists but is not wired in), `GGML_UNARY_OP_SOFTPLUS` (gate
  `softplus(alpha+dt)*A`) and `GGML_OP_DELTA_NET` (the fused recurrent op
  `f(q,k,v,g,beta,state) -> [output | new_state]`). `GGML_OP_REDUCE` is only needed for
  multi-device splits of the layer. Unlike the fused up-gate, these ops are **stateful**
  (they read and write the KV-cache state every step), so CPU fallback is not merely slow:
  the per-step GPU<->CPU state round-trip currently produces garbage logits (NaN /
  "Failed to sample token"). Implementing them requires new shaders + pipeline variants
  and `supports_op` / `build_graph` entries; `DELTA_NET` is the large one (a fused,
  flash-attention-style recurrent kernel; it is ik_llama-specific, so there is no upstream
  Vulkan implementation to port).
- `GGML_OP_MULTI_ADD` exists but check the specific fused-mul-multiadd variants
  (`fused_mmad`); `-no-mmad` disables them

`GGML_OP_FUSED_UP_GATE` and `GGML_OP_MOE_FUSED_UP_GATE` are now implemented (see
"What we fixed"). Note that the fused up-gate is implemented as two matmuls + a combine
kernel rather than a single fused kernel, so on Vulkan it is roughly on par with the
`-no-fug` path rather than faster.

To close the remaining gaps, implement the missing ops as Vulkan kernels and extend
`ggml_backend_vk_supports_op` / `build_graph`.

### Quant type coverage

The Vulkan `MUL_MAT` supports `F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K,
Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL` plus
the full **IQK/K-family** (`IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K, IQ2_KS, IQ3_KS, IQ4_KS, IQ4_KSS,
IQ5_KS, IQ2_KL`) and **KT-family** (`IQ1_KT, IQ2_KT, IQ3_KT, IQ4_KT`). The decode path uses
native per-type `mul_mat_vec` kernels; prompt processing dequantizes to F16 on the GPU and uses
the F16 matmul (correct, and on-GPU, but not as fast as a native quantized matmul kernel).

The `*_R4` repack variants (`IQ2_K_R4, IQ3_K_R4, IQ4_K_R4, IQ5_K_R4, IQ4_KS_R4, IQ5_KS_R4`)
and the remaining CUDA-supported types (`Q6_0, MXFP4, IQ1_BN, IQ2_BN, IQ1_S_R4, IQ1_M_R4`)
are still not supported and run their matmuls on CPU.

### Performance / architecture

- No async tensor copies, no events, no transfer/compute queue synchronization. Enabling
  those (as upstream did) would allow true per-decode overlap and remove the per-`graph_compute`
  fence wait.
- The spin in `ggml_vk_wait_for_fence` is still a busy-wait for the final fence; upstream
  keeps the same pattern, but a blocking `vkWaitForFences` for the tail would reduce CPU
  usage further at a small latency cost.
- `-sm graph`/`-sm attn` split modes are rejected by the Vulkan backend.

### Integration

- With `GGML_BACKEND_DL`, per-device memory reports 0 MiB (the backend memory functions are
  not linked into the llama library), so `--fit` does not work in that configuration.
- RPC servers are not part of the backend registry; they are appended after the registered
  backends in `model->devices`.

### Testing

- Only NVIDIA (RTX 3090, `NV_coopmat2`) and AMD (Strix Halo iGPU, RADV, `KHR_coopmat`)
  have been exercised. Intel, Apple (Metal is separate), and other AMD parts are untested
  in this codebase.
- `tests/test-fused-up-gate.cpp` compares the fused up-gate ops (dense and MoE: fused and
  separate weights, with/without per-expert biases, single and multi token, several weight
  types) against the CPU reference on a target backend, and additionally checks that the
  fused path is bit-identical to the equivalent non-fused graph on the same backend.
  Run as `test-fused-up-gate CPU|Vulkan0|CUDA0`. Note that the IQ4_XS CPU-vs-GPU comparison
  has a large pre-existing gap (visible even in plain `MUL_MAT`), so that type is allowed a
  looser tolerance.
- `tests/test-iqk-quants.cpp` validates the 15 IQK/KT types against the scalar dequant
  reference (the format definition): single-token decode, small batches, larger K,
  multi-token (dequant-to-F16 path), MoE (`MUL_MAT_ID`) and `GET_ROWS` (quantized token
  embeddings, both single-block and multi-block rows). Run as
  `test-iqk-quants CPU|Vulkan0|CUDA0`. The KT family is allowed a looser tolerance because
  of its calibration factor (see above).
- `test-backend-ops` does not currently compile against this fork's headers.

### Fixed: large IQK/KT models were slow on Vulkan

A 32B Qwen2.5-Coder model quantized to `IQ4_KT` (16.7 GiB, fully offloaded to a 24 GiB RTX
3090) previously decoded at only ~0.6 tok/s with ~400-500% CPU usage. The root cause turned
out to be neither of the initially suspected GPU kernel costs:

- **`FUSED_UP_GATE` / `MOE_FUSED_UP_GATE` fell back to the CPU backend for IQK/KT weight
  types.** `supports_op` had the 15 IQK/KT types in the `MUL_MAT` and `GET_ROWS` switches
  but not in the fused up-gate switch, so every FFN `gate`+`up` matmul ran on the CPU.
  Each such split copied the IQ4_KT weights device→host (~23 ms for the two 141 MB FFN
  matrices over PCIe), computed on CPU, and copied the result back — the decode graph has
  ~60-130 such splits per token, which accounted for ~97% of the wall time (the GPU per-op
  timings were microseconds; the output projection was ~0.7 ms). The fix adds the 15
  IQK/KT types to the fused up-gate `supports_op` switch; the fused path is two `MUL_MAT`s
  (native `mul_mat_vec` for single token, dequant-to-F16 for batches) plus the combine,
  all already supported for these types.
- **`GET_ROWS` (quantized token embeddings) ran on the CPU.** A byte-addressed
  `get_rows_iqk.comp` shader now covers the 15 IQK/KT types. Two bugs were found and
  fixed while adding test coverage: the workgroup size (1024 elements per workgroup,
  matching the pipeline denom, instead of 256 which under-launched for `k > 1024`) and
  the per-row scale header read (`block_byte - META` is only the row start for block 0;
  the header is now read from the explicit row byte offset).
- **The dequant-to-F16 mat-mat kernels read the per-row scale header at
  `block_byte - row_meta_size`, which is only the row start for block 0.** This silently
  corrupted weights for any row with more than one 256-element block (`k > 256`), i.e.
  every prompt with more than `mul_mat_vec_max_cols` (8) tokens: prompt processing takes
  the mat-mat path (`dequant` + F16 matmul), and the FFN/output dequant produced garbage
  logits — the model degenerated to repeating "!". Fixed in the 7 affected dequant
  shaders (`iq1_kt`, `iq2_kt`, `iq3_kt`, `iq4_ks`, `iq4_kss`, `iq4_kt`, `iq5_ks`) by
  reading the header from the row start (`row_bytes * (ib / nbpb)`). The vec
  (`mul_mat_vec`) and `GET_ROWS` paths were unaffected (they already used the row start).

After the fix, decode runs at ~17 tok/s and prompt processing at ~41 tok/s on the same
machine — a ~28x / ~19x speedup — with the GPU doing the FFN work (the earlier ~80% GPU
utilization at 0.6 tok/s was the GPU waiting on the CPU splits).

Remaining performance notes:

- The decode (`mul_mat_vec`) kernels still dequantize per element with the KT-family
  multiplicative-hash decode (4 hash rounds per weight); the output projection
  `[5120, 152064]` alone is ~0.7 ms/token. This is ALU-heavy but no longer the dominant
  term — a native quantized matmul kernel for the mat-mat path would help prompt
  processing, which currently dequantizes each weight matrix to F16 on the GPU.
- The scheduler no longer copies the IQ4_KT FFN weights per split; the remaining
  device→host traffic is limited to the logits.

An earlier symptom (the model generating "!" repeatedly) was from a pre-fix build (wrong
`ql`/`qh` offsets and 16-bit reads in the IQ4_KT kernels); the current build generates
coherent text.
