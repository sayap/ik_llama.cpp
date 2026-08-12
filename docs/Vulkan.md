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
- Quant types from the IQK family (`IQ4_KT`, ...) are **not** in the Vulkan backend's
  `supports_op` list, so models quantized with those types run all their matmuls on CPU
  regardless. Use a supported type (`Q8_0`, `Q6_K`, `Q4_K`, `Q4_0`, ...) or the CUDA backend.

## What we fixed

### 0. Fused up-gate / MoE fused up-gate (`-fug` / `-fmoe`)

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
  BF16 and the supported quant types), the activations are F32 and the unary op is one of the
  four supported ones; anything else falls back to the CPU backend as before.

A work-size bug in `ggml_graph_plan` was fixed along the way: the `MOE_FUSED_UP_GATE` plan
skipped the quantized-activation work buffer when the gate weights are fused (`src[1] == NULL`),
which made the CPU reference compute overflow its work buffer (heap corruption). The dense
`FUSED_UP_GATE` plan also sized the work buffer from the weights instead of the activations.
Both now size it from `src[2]` (the activations).

### 1. Synchronization: one wait per graph instead of per batch

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

### 2. Fence mix-up found while debugging a hang

`ggml_vk_wait_for_fence()` spins on `ctx->fence`, but the graph finalization had been
submitting an empty batch with `ctx->device->fence` — a *different* fence object. The
result was an infinite spin with an idle GPU. The fix is to use `ctx->fence` consistently.

### 3. Cooperative-matrix assert on coopmat2 GPUs

`GGML_ASSERT((GGML_KQ_MASK_PAD % rows_cols[0]) == 0)` in the flash-attention shader setup
crashed on any GPU with `NV_coopmat2` support (e.g. the RTX 3090), in both static and
dynamic builds. The coopmat2 shaders use clamped tensor layouts and handle the mask padding
explicitly, so the assert is now relaxed for row granularities larger than
`GGML_KQ_MASK_PAD`.

### 4. Integrated GPUs are enumerated

The instance init only added discrete GPUs to `vk_instance.device_indices`. Integrated GPUs
(e.g. an AMD Strix Halo iGPU) are now included as well, matching upstream llama.cpp. This
makes a device like the AMD Radeon 8060S show up as `Vulkan1` and selectable via
`-dev Vulkan1` without `GGML_VK_VISIBLE_DEVICES`.

### 5. `-dev`/`--device` integration

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

- `GGML_OP_SSM_CONV`, `GGML_OP_DELTA_NET` and friends (recurrent / hybrid models)
- `GGML_OP_L2_NORM` (and possibly other norm variants)
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
Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL`.
Missing are the IQK-family types used by ik_llama imatrix quants: `IQ4_KT` (and any other
`*_KT` types). Models using those types run all matmuls on CPU. Adding them requires new
dequant/mul_mat shaders and pipeline variants.

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
- `test-backend-ops` does not currently compile against this fork's headers.
