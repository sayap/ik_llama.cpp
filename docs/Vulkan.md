# Vulkan backend: current state, learnings, and gaps

This document summarizes what we have learned while working on the Vulkan backend of
ik_llama.cpp, what has been fixed, how to get good performance, and what is still missing.

## TL;DR

- The Vulkan backend is **correct** for the ops it supports, but its **op/type coverage is
  far behind the CUDA backend and behind ik_llama's own graph builder**.
- The single biggest performance trap: ik_llama fuses the FFN `up`+`gate` matmuls into
  `GGML_OP_FUSED_UP_GATE` (and MoE models into `MOE_FUSED_UP_GATE`) **by default**, but the
  Vulkan backend does not implement those ops. Every layer's FFN then runs on the **CPU
  backend** with expensive GPU↔CPU copies per split, which both tanks throughput and burns
  CPU cores.
- Immediate workaround: run with `-no-fug` (and `-no-fmoe` for MoE models) to make the
  graph use plain `MUL_MAT`s, which the Vulkan backend executes on the GPU. On an RTX 3090
  this goes from ~17 tok/s to ~400 tok/s for a small dense Q8_0 model (~CUDA speed).
- Quant types from the IQK family (`IQ4_KT`, ...) are **not** in the Vulkan backend's
  `supports_op` list, so models quantized with those types run all their matmuls on CPU
  regardless of `-no-fug`. Use a supported type (`Q8_0`, `Q6_K`, `Q4_K`, `Q4_0`, ...) or
  the CUDA backend.

## What we fixed

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
| before the sync fix (fused FFN on CPU) | ~17 |
| after sync fix, still fused | ~17 (CPU fallback dominates) |
| after sync fix, `-no-fug` | ~400 |
| CUDA backend, same model | ~440 |

The GPU kernels themselves are fast (single-digit to tens of µs per op); the wall-clock
cost was the CPU-fallback splits and their data movement.

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

- `GGML_OP_FUSED_UP_GATE` (dense FFN up+gate fusion, default on) — workaround: `-no-fug`
- `GGML_OP_MOE_FUSED_UP_GATE` (MoE up+gate fusion, default on) — workaround: `-no-fmoe`
- `GGML_OP_SSM_CONV`, `GGML_OP_DELTA_NET` and friends (recurrent / hybrid models)
- `GGML_OP_L2_NORM` (and possibly other norm variants)
- `GGML_OP_MULTI_ADD` exists but check the specific fused-mul-multiadd variants
  (`fused_mmad`); `-no-mmad` disables them

To close this, implement the missing ops as Vulkan kernels (a fused two-matmul + activation
shader for the up/gate family) and extend `ggml_backend_vk_supports_op` / `build_graph`.

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
- The correctness of the `-no-fug` path has been spot-checked against the CUDA backend;
  a broader op-by-op comparison (e.g. `test-backend-ops`) would be valuable, but that test
  does not currently compile against this fork's headers.
