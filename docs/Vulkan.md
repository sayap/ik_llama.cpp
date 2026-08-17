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
  `mul_mat_vec` kernel per type and prompt processing runs a flat-dequant-to-F16 + tensor-core
  matmul (the dequant kernels are vectorized and the quantized weights are read once; see
  "vs CUDA" for the measured result). `MXFP4` is now supported too (native
  `mul_mat_vec` / `MUL_MAT_ID` / `GET_ROWS` plus the flat-dequant-to-F16 prompt path; see
  "MXFP4" below). The `*_R4` repack variants and `IQ1_BN`, `IQ2_BN` are still not
  supported and fall back to the CPU backend.

## What we fixed

### 0. IQK / KT quant families (QK_K = 256 imatrix quants)

All 15 base IQK/KT types are now supported by `MUL_MAT`, `MUL_MAT_ID` (vec and mat-mat paths)
and `GET_ROWS`:

- **decode (mul_mat_vec)**: a dedicated GLSL kernel per type (`mul_mat_vec_iq{1,2,3,4,5,6}_k*`,
  `mul_mat_vec_iq{2,3,4,5}_ks*`, `mul_mat_vec_iq4_kss*`, `mul_mat_vec_iq2_kl*`,
  `mul_mat_vec_iq{1,2,3,4}_kt*`) that reads the quantized bytes with explicit row/block byte
  addressing (the KS/KL/KT formats carry a per-row scale header, `row_meta_size` 2 or 4) and
  dots against the F32/F16 activations. On devices with `VK_KHR_shader_integer_dot_product`
  the KT-family hash decode's 4-shift + 3-add byte sum is replaced by one `dotPacked4x8EXT`
  (a `_dot4` shader variant is selected at pipeline creation; the scalar fallback remains).
- **decode Q8_1 activations (Trellis KT + IQK)**: on integer-dot devices the
  `mul_mat_vec` path now quantizes the F32 activations to Q8_1 and dots packed-int8 decodes
  against them with `dotPacked4x8EXT`. The Trellis types (`IQ1_KT`..`IQ4_KT`) use the
  multiplicative-hash decode (mirroring CUDA's `vec_dot_iq{1,2,3,4}_kt_q8_1`). The IQK types
  use shared table decodes with a CUDA-style 8-threads-per-block mapping and a uint32
  weight-buffer view: `IQ2_K` (2-bit `iq2k_table` via `dotPacked4x8EXT` offset trick),
  `IQ3_K` (3-bit `iq3nl_values` packed into a 4-bit direct-indexed table), `IQ4_K`/`IQ4_KS`/
  `IQ4_KSS` (4-bit `iq4k_values` byte-pair table) and `IQ5_K` (5-bit `iq5nl_values` table).
  IQ3_K's 110-byte block is only 2-byte aligned, so it uses an unaligned-safe uint32 loader.
  This removes the scalar-FMA activation dot from the decode FFN and attention projections.
- **prompt processing (mul_mat)**: on NV_coopmat2 devices the mat-mat path runs the
  **coopmat2 tensor-core matmul with inline dequant** (`mul_mm_cm2.comp` + per-type decode
  functions in `dequant_funcs_cm2.comp`; see "vs CUDA" below for the details and the measured
  result). The dequant-to-F16 kernels (`dequant_iqX_*`) feeding the F16 matmul remain as the
  fallback for the `MUL_MAT_ID` (MoE) mat-mat path and for non-coopmat2 devices. The flat
  dequant shaders are row-meta aware (blocks-per-row and meta size are passed in the push
  constants) and the dispatch passes the tensor's full byte size (the per-row scale headers
  must be included in the source subbuffer).

  `IQ4_KT` additionally has a native Q8_1 integer-dot mmq (`mul_mmq.comp`): a byte-addressed
  tile loader runs the hash decode with `dot4` and packs the values as signed int8, then the
  standard dp4a vec-dot accumulates against Q8_1 activations. It is gated to devices without
  cooperative matrices — on tensor-core GPUs it is correct but slower than the dequant+F16 path
  (see "vs CUDA" below).
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

### 7. Gated delta-net (`qwen35` / `qwen3next`)

The four recurrent ops are now implemented on Vulkan (see "Op coverage"):

- **`SSM_CONV`**: a single-sequence fast path (`ssm_conv.comp`, parallel over rows and
  32-token blocks) plus a `ssm_conv_final_state.comp` kernel. This covers the qwen35/
  qwen3next graph (the builder always uses `n_kv == 1`); multi-sequence routing and
  per-step conv checkpointing fall back to CPU.
- **`L2_NORM`**: wired in and made stride-aware. The pre-existing shader assumed a
  contiguous `[ne0, rows]` layout (q/k are permuted views for prompt batches) and used
  `inversesqrt(max(sum, eps))` instead of the CPU's `1/max(sqrt(sum), eps)`.
- **`SOFTPLUS`**: new `softplus.comp` shader (f32/f16) matching `ggml_compute_softplus_f32`.
- **`DELTA_NET`**: new `delta_net.comp` shader, mirroring the CUDA kernel's
  decomposition. A workgroup handles `CS` rows (spec constant, min(32, `S_V`)) and the
  `S_V` threads are arranged as `S_V/CS` column groups × `CS` rows; each thread keeps
  only `CS` floats of its state row in registers, partial sums are reduced across column
  groups in shared memory, and the q/k dot-product scalar is reduced in parallel (q/k
  re-normalization is skipped since the graph already L2-norms them). This replaced the
  original one-row-per-thread version (full 128-float state row, serial thread-0
  reductions) and cut `DELTA_NET` from ~2019 µs to ~662 µs per 512-token batch (~3×).
  v/g/beta use the permuted-view strides from the tensors, and the new state is written
  into the result tail with the existing `CPY` node persisting it (the CPU-side src[7]
  fused-copy optimization is not needed).

`tests/test-delta-net.cpp` checks all four ops against the CPU reference (single- and
multi-token, both repeat types, several head sizes) on a target backend, and passes on
`Vulkan0` and `Vulkan1`. A Qwen3.6-27B IQ4_KS model now runs end-to-end on `Vulkan0`.

### 8. Flash-attention precision for qwen35/qwen3next in dynamic builds

The qwen35/qwen3next full-attention layers were missing from the
`should_use_f32_precision` arch list in `llm_build_kqv()` / `build_std_attention()`, so
they ran flash attention with `GGML_PREC_F16` and produced garbage logits (`"!!!!"`,
NaN, "Failed to sample token"). Static Vulkan builds mask this (`GGML_USE_VULKAN`
forces F32 for every arch), so it only surfaced with `GGML_BACKEND_DL=ON`, where the
arch list is the only thing forcing F32. Added `LLM_ARCH_QWEN3NEXT`,
`LLM_ARCH_QWEN35` and `LLM_ARCH_QWEN35MOE` to both lists.

### 9. Q6_0 quant

`Q6_0` (32-element blocks: fp16 scale + 8 bytes of high bits + 16 bytes of nibbles)
is now supported by `MUL_MAT` and `GET_ROWS`, using the same paths as the other
legacy quants: native `mul_mat_vec` decode (`dequantize`/`dequantize4` in
`dequant_funcs.comp`) and the flat dequant-to-F16 prompt path (`dequant_q6_0.comp`;
like the IQK/KT families, this beats the cm2 inline dequant). The flat dequant was
initially a naive byte-addressed scalar shader; it is now uint32-load + f16vec4-store
(like the IQK/KT dequants), which cut the Q6_0 projection matmuls ~2× each.
`MUL_MAT_ID` (MoE experts) is wired too, for both the vec and mat-mat paths. This
lets Qwen3.6-27B IQK models (whose qkv/out projections are `Q6_0`) run those matmuls
on Vulkan instead of falling back to the CPU. The single-token decode (`mul_mat_vec`)
path now has a native Q8_1-activation kernel (`mul_mat_vec_q6_0_q8_1.comp`) as well:
CUDA's `vec_dot_q6_0_q8_1` is ported 1:1 (2 threads per 32-element block, each decoding
16 elements to packed int8 with the -32 offset folded into the block-sum correction
`sumi*d8 - 16*ds.y`), so the Q6_0 FFN/output projections quantize the F32 activations
to Q8_1 and dot with `dotPacked4x8EXT` instead of the scalar FMA decode. This lifts
decode ~5% on Qwen3.6-27B (the output projection reaches ~870 GB/s, near the 3090's
memory peak). CUDA instead has a native int8 mmq for `Q6_0` (decode to int8 + INT8
tensor cores), which Vulkan does not yet do.

### 9b. MXFP4 quant

`MXFP4` (OCP microscaling 4-bit: 32 4-bit e2m1 values per block plus one shared E8M0
scale exponent, 17-byte blocks) is now supported by `MUL_MAT`, `MUL_MAT_ID` and
`GET_ROWS`. It is treated like the other 32-element-block legacy quants:

- **decode (`mul_mat_vec`)**: the generic `mul_mat_vec.comp` with a `DATA_A_MXFP4`
  `dequantize`/`dequantize4`/`get_dm` (`dequant_funcs.comp`). The 16-entry value table
  is kept in shared memory, and `get_dm` reconstructs the power-of-two scale from the
  E8M0 byte, mirroring `ggml_e8m0_to_fp32_half`.
- **prompt (`mul_mat`)**: on `NV_coopmat2` devices, a native cm2 inline-dequant
  matmul (`mul_mm_cm2.comp` + `dequantFuncMXFP4`) reads the 17-byte blocks directly;
  `ggml_vk_get_mul_mat_mat_pipeline` no longer falls back to dequant+F16 for MXFP4.
  Non-coopmat2 devices keep the flat `dequant_mxfp4.comp` → F16 matmul fallback.
- **`MUL_MAT_ID`**: the mat-mat-id path now uses the same native cm2 inline-dequant
  matmul on coopmat2 (this matters for large MoE models: the old dequant-to-F16 path
  dequantized the *entire* expert matrix — 8 GiB of F16 for DeepSeek-V4's fused
  256×4096×4096 gate+up experts — and OOM'd). The vec path uses a native
  `mul_mat_vec_id_mxfp4` shader.
- **`GET_ROWS`**: the generic `get_rows_quant.comp` with the MXFP4 dequant.

`supports_op` accepts MXFP4 for `MUL_MAT`, `MUL_MAT_ID`, `GET_ROWS` and
`FUSED_UP_GATE`/`MOE_FUSED_UP_GATE`. Correctness is covered by
`tests/test-iqk-quants.cpp` (MXFP4 added to the type list); all cases pass on
`Vulkan0` and `Vulkan1`. A Qwen2.5-Coder-32B MXFP4 model runs end-to-end on
`Vulkan0` (~29 tok/s decode, ~324 tok/s prompt at 300 tokens), fully offloaded.

### 10. Gated delta-net PP/TG performance and remaining gaps

Qwen3.6-27B-IQK (17 GB; qkv/gate/ssm_out/attn projections are `Q6_0`, FFN is `IQ4_KS`),
RTX 3090, `llama-server` + tiny warmup prompt, 2870-token prompt / 128-token decode:

| | CUDA0 | Vulkan0 | gap |
|---|---|---|---|
| PP | ~1420 tok/s | ~1110 tok/s | ~1.28x |
| TG | ~42.6 tok/s | ~35.8 tok/s | ~1.19x |

(TG reflects the Q8_1-activation Q6_0 vec kernel plus the IQ4_KS uint32-load
and byte-indexed-table decode; it was ~30.2 tok/s / ~1.41x before those. PP is
unchanged — it goes through the dequant-to-F16 + F16 matmul path.)

Per-op profiling of the Vulkan prompt path (per 512-token batch) shows the remaining
cost is dominated by the quantized matmuls, not the recurrent ops:

- `FUSED_UP_GATE` + FFN-down (`IQ4_KS`): ~276 ms/batch.
- qkv/gate/ssm_out/attn projections (`Q6_0`): ~120 ms/batch (after the uint32+f16vec4
  dequant rewrite; was ~230 ms with the naive byte-addressed dequant).
- `DELTA_NET`: ~32 ms/batch (~5%), after the CUDA-style register-tiled rewrite (was
  ~97 ms with the one-row-per-thread shader).
- `SSM_CONV`: ~3 ms/batch.

Remaining gaps:

- **The generic coopmat2 F16 matmul** runs below FP16 peak and is now the main PP
  lever; it affects the FFN and every projection (see the `Qwen2.5-Coder` "vs CUDA"
  section for the tuning analysis).
- **`Q6_0` has no native int8 mmq on Vulkan** (CUDA decodes to int8 and uses INT8
  tensor cores); `coopmat_int_support` is detected but unused.
- **Decode (`mul_mat_vec`) TG** (~1.19x) is FFN/output-projection bound. The `Q6_0`
  vec path is now a native Q8_1 + dot4 kernel (see "Q6_0 quant"), and the `IQ4_KS`
  Q8_1 table-decode kernel reads its 4-byte-aligned weights through a uint32 view
  (instead of 4 byte loads per word) and uses a byte-indexed shared table. The IQ4_KS
  FFN (~44% of decode) now runs at ~730 GB/s; the Q6_0 output projection reaches
  ~870 GB/s, so the remaining IQ4_KS gap is the shared-memory table lookup (removing
  it experimentally lifts decode to ~39 tok/s).

### 11. Graph parallelism plumbing (`-sm graph`)

The pieces the scheduler needs to split a model graph across multiple devices are now
in place:

- **Backend-agnostic split buffer type** (`ggml_backend_split_buffer_type` in
  `ggml-backend.cpp`): the `init_tensor`/`set_tensor`/`get_tensor` logic that was
  originally added as a Vulkan split buffer now lives in ggml core and allocates each
  `ggml_split_tensor_t` shard on the per-slot buffer type supplied by the caller.
  `split_dim` -1 (replicated), 0 (row split), 1 (column split) and 2 (contiguous only)
  are supported; the explicit per-device `ranges` form stored in `tensor->op_params`
  (used by the gated delta-net qkv/conv/gate weights and by MoE/MLA distributions) is
  supported for `split_dim` 0 and 1. `get_tensor` implements the inverse for the
  contiguous -1/0/1 cases (the `ranges` and `split_dim=2` forms are not invertible on the
  wrapper; the loader reads those per-rank).
- `ggml_backend_vk_split_buffer_type()` is now a thin wrapper over the generic type with
  one entry per Vulkan device. `ggml_backend_supports_buft()` accepts the generic split
  type for every backend (it is a container for per-device sub-buffers).
- **`GGML_OP_REDUCE`** (`supports_op` + `ggml_vk_op_reduce`): a host-staged
  all-reduce. Each partial tensor is pulled with `ggml_backend_tensor_get`, summed in F32
  (F16/F32 supported), and the full sum is written back to every participant. REDUCE runs
  as its own scheduler split after the producers have completed, so the host round-trip
  is correct even though the backend still has no events/async copies. It is slow relative
  to a peer-to-peer ring reduce, but it is the correct first implementation.

### 11b. Lean lazy host-staged reduce (dual-NVIDIA decode)

The two-Vulkan-device host-staged reduce was rewritten to cut the per-exchange driver
overhead and overlap the exchange with scheduler work:

- **Lazy stash**: the REDUCE node no longer performs the exchange. It submits the two
  partial→host-staging reads right away (as `vkCmdCopyBuffer` on each device's **compute
  queue**, so they are queue-ordered after the producing kernels) and returns. The fence
  waits, the CPU add and the fire-and-forget sum write-backs run later, at the first
  point where the result becomes observable (just before the next compute batch is
  submitted, at `synchronize`, or at any host read of a device tensor).
- **Proper cross-stage memory barriers**: the copies read a compute-written tensor and
  feed compute-consumed tensors, so they use explicit `COMPUTE→TRANSFER` and
  `TRANSFER→COMPUTE` pipeline barriers. The generic `ggml_vk_sync_buffers` barrier uses
  the queue's own stage flags on both sides, which does *not* cover a transfer-stage copy
  on the compute queue — without the fix the copies silently read/write stale data.
- **Fence lifecycle**: `dev->fence` is reset **after** every wait. Leaving it signaled
  makes the next `vkQueueSubmit(…, dev->fence)` + `waitForFences` return immediately and
  race with the copy (this was the source of a hard-to-find decode corruption).
- **One combined wait**: the partial reads are fenced and then waited per-device
  (each fence on its own device); the sum is written back with fire-and-forget
  compute-queue copies (visible to the next split by queue order). The previous
  sequence (per-device flush, transfer-queue read, wait, transfer-queue write, wait)
  used ~6 submissions + ~6 fence waits per exchange; the new one uses N read submits +
  N waits + N write submits for N devices.
- **N devices**: the lazy host-staged path now handles any number of participating
  Vulkan devices (previously `>2` devices fell back to the slow `tensor_get` loop).
  Each device contributes one partial; the reads are submitted on all devices' compute
  queues at stash time and the N-way sum is broadcast back to every participant.
- `ggml_backend_sched_compute_splits` no longer `synchronize`s Vulkan backends before a
  REDUCE split (their exchange flushes itself at the right point); non-Vulkan
  participants (mixed CUDA runs) still get the host-staged sync.

On two NVIDIA RTX PRO 4000 (Blackwell) GPUs, Qwen3.6-27B-IQK:

| decode (`-n 128`, `--temp 0`) | tok/s | | prompt (1141 tok) | tok/s |
|---|---|---|---|---|
| single Vulkan0 | ~23.1 | | single Vulkan0 | ~500 |
| `-sm graph` 2 GPUs | **~25.1** | | `-sm graph` 2 GPUs | **~611** |
| `-sm graph` 3 GPUs | ~21.8 | | `-sm graph` 3 GPUs | ~122 |

All `-sm graph` runs are **byte-identical** to the single-GPU reference (96-token
greedy decode). Two devices now beat a single GPU for both decode and prompt
eval; three devices do not: the per-layer exchange skew grows with the number of
participants, the split count (and per-split scheduler latency) rises with device
count, and the recurrent/attention ops are not split three ways — the marginal
shard parallelism no longer pays for the added serialization. The 3-GPU prompt
number is also host-bound: `nbytes >= 1MB` prompts take the N-way host-staged path
(the shared-buffer P2P path is 2-device only), so each reduce moves the full tensor
across PCIe several times.

Remaining gap to CUDA's `-sm graph` (~44 tok/s on the same pair): CUDA's reduce is
`cudaMemcpyPeerAsync` + on-stream kernels (no host round trip, no per-layer drain),
while the Vulkan host-staged reduce must wait for every device to finish each layer's
shard before the next layer can start. A no-op reduce (fence waits skipped) measures
~32 tok/s, so even a free exchange only reaches ~1.4× single-GPU here — the rest is
split imbalance and per-split scheduler latency. Closing the gap needs an on-device
cross-device copy (the CUDA-interop `cuMemcpyPeer` path) ordered against Vulkan
compute via a same-device external-semaphore import (see "Performance / architecture"
below).

### 11c. Future: async on-device reduce on AMD/mesa (`DMA_BUF` + `SYNC_FD`)

The default decode reduce is host-staged and the shared-buffer P2P path is only
synchronous and only auto-engaged above 1 MiB (`GGML_VK_P2P=1` forces it). The
P2P path's data movement is already on-device on mesa (the shared buffer is
`DEVICE_LOCAL` + `DMA_BUF`, imported on both cards), but it is **host-ordered**: each
`ggml_vk_buffer_copy` does flush + submit + fence wait, `vk_reduce_add` allocates a
descriptor pool + temp context + fence per call, and the whole exchange is ~5
serialized submit+wait pairs — so its fixed overhead dominates decode-sized (20 KB)
tensors and it loses to the lean lazy host path. On NVIDIA the same path falls back
to host-visible `OPAQUE_FD`, so it is not even on-device.

Three steps close this on AMD (the one place pure Vulkan has a real P2P route):

1. **Make the P2P reduce lazy/async**, mirroring the host path: submit the
   partial→shared copies fenceless on each device's queue, wait each device's fence
   once (not once per copy), cache the add pipeline's descriptor set (kill the
   per-call `createDescriptorPool`), and make the broadcast copies fire-and-forget.
   That drops the fixed cost to ~2 submits + ~2 waits, at which point the on-device
   `DMA_BUF` data path should beat the host path for decode.
2. **Vendor-aware heuristic**: once step 1 lands, route decode reduces through the
   P2P path on `external_dma_buf_support` devices (e.g. `nbytes >= (dma_buf ? 4096
   : 1MB)` instead of the flat 1 MiB threshold), keeping `OPAQUE_FD` devices on the
   host-staged path.
3. **Wire cross-device `SYNC_FD` semaphores** (mesa allows importing
   `VkImportSemaphoreFdInfoKHR` across devices; NVIDIA's `OPAQUE_FD` is
   same-device-only and `SYNC_FD` is absent). This replaces the host fence waits with
   on-device signal/wait ordering — the Vulkan equivalent of CUDA's stream/event
   reduce — and removes the per-layer bubble entirely. The scaffolding already exists
   (`ggml_vk_create_timeline_semaphore`, the `wait_semaphores`/`signal_semaphores`
   plumbing in `ggml_vk_submit`); it is simply never populated.

Caveats: cross-device `DMA_BUF` memory import sits in a spec-gray area
(`VUID-VkMemoryAllocateInfo-None-00644` is violated in practice on mesa) and mesa's
P2P needs `CONFIG_DMABUF_MOVE_NOTIFY`. Cross-*node* (RPC) reduces are a separate and
much larger problem: every exchange then pays a full network round-trip, which is
orders of magnitude above the µs-scale fence waits discussed here and dominates any
multi-node `-sm graph`/`--rpc` split until the RPC layer overlaps transfers with
compute.
- `llama_default_buffer_type_split()` keeps the CUDA/SYCL specialized split buffers for
  homogeneous static builds, and otherwise builds a per-slot buffer-type map from the
  registry and returns the generic split buffer. This makes `-sm graph` work in
  `GGML_BACKEND_DL` builds and for mixed-backend device selections, not just static
  single-backend builds. The Vulkan backend-init path no longer hard-rejects
  `-sm graph`/`-sm attn`; those modes initialize all Vulkan devices like `-sm layer`.

`tests/test-vk-split.cpp` validates the generic split buffer (contiguous, delta-net
`ranges`, an IQ4_KS row-meta split, and a mixed Vulkan0+CPU per-slot map) and a
cross-device reduce through a real Vulkan0+Vulkan1 ggml scheduler.

End-to-end `-sm graph` was validated on Qwen3.6-27B-IQK across Vulkan0 (RTX 3090) +
Vulkan1 (AMD Strix Halo): output matches the single-device run. Two bugs were found and
fixed along the way: the split buffer's contiguous row split omitted `row_meta_size` from
its block-data source offset (corrupting IQ4_KS FFN weights), and `create_split()`'s
mem_used-adjusted heuristic rounded a small device down to zero shards for lopsided
memory splits. `create_split()` now uses a largest-remainder proportional split with a
minimum-one-chunk guarantee per participating device, so the default memory split and
lopsided `-ts` ratios both work. Still pending: the CUDA foreign-buffer REDUCE fallback
for mixed CUDA+Vulkan runs and the MLA `-sm attn` `split_dim=2` path.

Re-validated end-to-end on two NVIDIA RTX PRO 4000 Blackwell GPUs (`Vulkan0,Vulkan1`,
`-sm graph`, Qwen3.6-27B-IQK): default split, `-smf32`, `-ts 8,1` and `GGML_VK_P2P=1` all
run and produce coherent output (the default f16-reduce and the P2P path are bit-identical
on a 32-token decode). Three additional bugs were found and fixed along the way:

- **`llama_max_devices()` returned 1 in `GGML_BACKEND_DL` builds** (no `GGML_USE_*` macro
  is defined), so `-ts 8,1` was rejected as "invalid parameter". It now returns 16, matching
  upstream, which also fixes `llama-bench`'s tensor-split handling in DL builds.
- **`FUSED_RMS_NORM` had no f16 source pipeline**, but with `-sm graph` the default f16
  reduce feeds f16 activations into the next layer's fused RMS norm, so the default
  (`-smf16`) split crashed on prompt processing with a null pipeline. A `fused_rms_norm_f16`
  shader variant is now generated (`A_TYPE=f16`, f32 weight/dst), and `supports_op`/pipeline
  selection accept the f16-source form. `RMS_NORM`/`FUSED_RMS_NORM` `supports_op` now match
  the actually-supported type combinations instead of returning `true` unconditionally.
- **The opt-in `GGML_VK_P2P=1` shared-buffer reduce was broken on NVIDIA in three ways**: the
  manual ADD dispatch used the wrong workgroup layout (only the first ~half of each tensor was
  reduced, with overlapping workgroups), the device-local `OPAQUE_FD` import silently produced
  a non-shared allocation (so the shared buffer is now host-visible on NVIDIA), and the
  cross-device storage-buffer read was not coherent (the remote partial is now first copied
  into a per-device device-local staging buffer with `vkCmdCopyBuffer`, and the ADD only touches
  two local buffers). The import fd is now closed, the reduce-add pipeline is requested during
  the dryrun so it is actually compiled, and the static reduce-pair map is cleared in
  `ggml_backend_vk_free` so the imported/exported memory is not freed after device teardown.

### 12. DSA / GLM-DSA / DeepSeek-V4 sparse-attention ops

All 8 sparse-attention ops are now implemented — the 7-op DSA family shared by GLM-5.2
(`glm-dsa`, via `src/graphs/build_deepseek2.cpp`) and DeepSeek2/Mistral4/BailingMoE3/
OpenPangu, plus the DeepSeek-V4 `DS4_COMP` extra (`src/graphs/build_deepseek4.cpp`):

- `INDEXER_TOPK` — two-kernel implementation (a score kernel `indexer_topk_score.comp`
  with F32/F16 `k` variants, plus `indexer_topk_select.comp` for the per-row top-k). Query
  rows are tiled so the score scratch buffer is bounded; the select kernel reads the score
  buffer and does a threshold-rescan tournament (no big register arrays, which avoided a
  driver crash on Blackwell).
- `MASK_TOPK` / `MASK_TO_IDX` — F32/F16 variants; `mask_to_idx` preserves increasing
  source-position order like the CPU reference.
- `SINKHORN` / `HC_PRE` / `HC_POST` — one thread per token / one thread per output element.
- `DS4_COMP` — both type 0 and type 1.
- `LATENT_ATTN` — dense (mode 0) and indexed (mode 1), F32/F16/Q8_0 cache, one thread
  per query row with online softmax.

`tests/test-dsa.cpp` validates every op against the CPU reference (all S sizes, both DS4
compression types, F32/F16/Q8_0 latent caches, F32/F16 indexer/mask types); 50/50 pass on
`Vulkan0` and `CPU`. `INDEXER_TOPK` is compared as a *set* per row because the CPU
reference's `iqk_bucket_topk` (used when `n_rows < n_threads`) returns most buckets in
original order rather than sorted.

### 13. SET_ROWS is enabled (KV-cache scatter)

`GGML_OP_SET_ROWS` was disabled since the Vulkan backend rewrite (`ggml_vk_set_rows` was
`#if 0` and the shader only handled I64 indices as `uvec2`). It is now wired up
(`supports_op` / `build_graph` / `compute_forward` / `ggml_vk_op_get_pipeline`), and the
`copy_to_quant.comp` SET_ROWS mode gained an I32-index variant (`set_rows_*_i32`),
selected at runtime from `src[1]->type`. This unblocks the DeepSeek-V4 KV-cache writes
(`raw_k_write_idxs` are I32), which previously forced the whole cache-write split onto the
CPU backend and crashed.

### 14. Dryrun cache now fingerprints the graph content

The `cached_cgraph == cgraph` dryrun cache (added for llama graph_reuse) could false-hit
when a freed cgraph was reallocated at the same address with a *different* graph, skipping
pipeline compilation and crashing on a null pipeline (reproducible as `test-delta-net`
SIGSEGVing at softplus in DL builds). The cache key now also includes a fingerprint of
every node and source (`op`, `type`, `ne`, `nb`, `op_params`, plus tensor identity pointers
for the RMS_NORM+MUL fusion topology).

### 15. Hybrid CPU/GPU view-allocation (ROPE_BACK) — fixed

Running DeepSeek-V4-Flash-0731-full with
`-dev Vulkan0,Vulkan1,Vulkan2 -sm layer -ngl 99 -cmoe` (attention on Vulkan, MoE FFN on
CPU) exposed a ggml-core scheduler bug: `ggml_rope_ext_inplace` (ROPE_BACK) produces a
*view* tensor; the scheduler could run it on Vulkan while the gallocr allocated its dst
view on the CPU buffer (because the next consumer — the MoE FFN — is on CPU under
`-cmoe`). `ggml_backend_view_init` then inherited the wrong buffer and dispatch hit a null
device buffer. Fixed in the scheduler: ops that produce a view and write it in-place now
stay on the same backend as the view source.

### 16. DSV4 flash-attention coverage

- Added FA head size **512** (K=V=512) for DSV4 raw attention; previously
  `fa_get_head_sizes` rejected it and the whole attention fell back to CPU.
- The dense Vulkan FA shader implements only `src[0..3]`. The DSV4 **indexed** variant
  (`src[5]`, a per-row top-k I32 gather) and the per-head **attention sinks** (`src[4]`)
  are now implemented by a dedicated scalar kernel `flash_attn_indexed.comp` (one thread
  per (query row, head), online softmax over the gathered keys, sink folded into the
  denominator/max exactly like the CPU reference). K/V F16 and Q8_0 are supported,
  matching the CUDA DSA reader. `supports_op` routes indexed FA to it when the graph is
  single-KV-head / single-batch (the shape the DSV4 builder and `dsa_attn.cu` produce),
  and `tests/test-dsa.cpp` now validates it against the CPU reference (single- and
  multi-token, F16/Q8_0 KV, with/without sinks, with trailing `-1` index padding, and the
  DSV4 512/512 head shape).
- Dense FA *with* sinks (no indexer) is now implemented too: the scalar and cm2 FA
  shaders take a `sinks` binding and fold the per-head sink into the final
  renormalization exactly like the CPU generic-FA path, and `supports_op` accepts
  `src[4] != NULL && src[5] == NULL` (split_k is disabled when sinks are present).
  `tests/test-dsa.cpp` validates it (`test_flash_attn_dense_sinks`) against the CPU
  reference.
- **Remaining → fixed**: end-to-end DSV4 decode was not byte-identical to CUDA. All
  Vulkan DSA kernels (HC_PRE/HC_POST/DS4_COMP/MASK_TOPK/MASK_TO_IDX/SET_ROWS) and the
  indexed FA match the CPU reference at real shapes in `tests/test-dsa.cpp`; the
  dense-with-sinks FA also matches. The actual root cause of the remaining gibberish
  was the **"flipped" rope** (`op_params[15] == 1`, used by DSV4 to place the rotated
  dims at the *end* of the 512-wide head instead of the start). The Vulkan rope
  shaders ignored that flag and always rotated the leading `n_dims` dims, corrupting
  the Q/K/V positions for every attention layer. Fixed by threading a `rope_offset`
  push constant into the `rope_neox`/`rope_norm` shaders (the leading non-rope dims
  are copied verbatim and the trailing `n_dims` dims are rotated), matching
  `ggml_compute_forward_rope_f32`. `DeepSeek-V4-Flash-0731-full` now decodes
  coherently on `-dev Vulkan0,Vulkan1,Vulkan2 -sm layer -cmoe` (byte-identical to
  CUDA on the tested greedy prompt). A defensive null-buffer abort in
  `ggml_vk_op_f32` turns any future cross-backend miss into a clear error instead of
  a segfault.

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

### Adding a new op (recipe)

Follow the `DELTA_NET` pattern in `ggml/src/ggml-vulkan.cpp`:

1. `ggml_backend_vk_supports_op` — accept the op only for the type combinations the
   shaders actually handle (don't return `true` unconditionally; an over-broad
   `supports_op` is a null-pipeline crash, not a CPU fallback).
2. `ggml_vk_build_graph` op switch — add `case GGML_OP_*:`, calling a dedicated
   `ggml_vk_<op>(...)` helper.
3. `ggml_vk_compute_forward` dispatch — route the node to the same helper (see the
   `ssm_conv` / `delta_net` cases).
4. Dedicated op function — model on `ggml_vk_delta_net`: request the pipeline in the
   dryrun, bind buffers / set push constants / dispatch in the record pass, and return
   `false` for unsupported shapes/types so the scheduler falls back to CPU.
5. GLSL shader — add `ggml/src/vulkan-shaders/<op>.comp` and register it in
   `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp` via
   `string_to_spv("<op>_f32", "<op>.comp", {})`; rebuild the `vulkan-shaders-gen` target
   so the SPIR-V is regenerated.
6. Dedicated ops with their own function don't go through the generic `ggml_vk_op_f32`
   dryrun short-circuit list.

### MoE offload vs `-sm graph` (notes, not yet benchmarked)

For large MoE models on the 3090 + Strix Halo, the reference hybrid config is:

    -dev CUDA0 -ngl 99 -cmoe -ub 2048

`-cmoe` keeps the MoE FFN (`ffn_*_exps`) in host RAM; `offload_op` (dc663fe6) streams those
FFN tensors into the 3090 when `batch_size * n_active >= min_batch * n_tot`, amortized by the
large ubatch. TG (`batch=1`) keeps the FFN on the CPU (memory-bound, DDR). So the hybrid is
"attention resident on 3090 + FFN streamed for PP + FFN on CPU for TG".

`-sm graph` does not stream; it statically shards tensors. Two variants matter:

- `-sm graph -cmoe`: the ncmoe override still force-places FFN on the CPU, so this is just
  the hybrid plus a tensor-parallel attention split we do not need — expected worse.
- `-sm graph` (no `-cmoe`): the FFN experts are statically split (~24 GB in 3090 VRAM, the
  rest in the iGPU's local RAM). PP becomes "3090 shard + iGPU shard + reduce" with no PCIe
  streaming; TG becomes "3090 VRAM + iGPU RAM" (iGPU RAM is the same DDR the CPU reads).

The open question for PP is a race between (a) the hybrid's PCIe streaming of the full FFN
vs (b) the iGPU computing its shard from local RAM. That is only likely to favor `-sm graph`
when the hybrid is PCIe-bound (FFN_bytes / PCIe_BW > compute_time), i.e. very large experts.
For TG the iGPU shares the CPU's DDR, so no win is expected, plus a per-layer reduce.

Benchmark plan (blocked on Vulkan op coverage for the sparse-attention MoE arches, i.e. the
indexer / DSA / CSA / HCA / GLM-DSA ops below; MLA-only arches such as GLM-4.5-Air are not
blocked by this — the MLA op set is already supported):

1. hybrid: `-dev CUDA0 -ngl 99 -cmoe -ub 2048`
2. `-sm graph`: `-dev CUDA0,Vulkan1 -sm graph -ub 2048`

record PP tok/s and TG tok/s for both.

## Remaining gaps

### Priority (highest first)

1. **`MXFP4`** — the micro-scaling 4-bit format (now supported, see "MXFP4 quant").
3. **Indexer / DSA / CSA / HCA / GLM-DSA**: `INDEXER_TOPK`, `MASK_TOPK`, `MASK_TO_IDX`,
   `SINKHORN`, `HC_PRE`, `HC_POST`, `LATENT_ATTN`, `DS4_COMP` — **now implemented** (see
   "DSA / GLM-DSA / DeepSeek-V4 sparse-attention ops" above and `tests/test-dsa.cpp`).
   End-to-end DeepSeek-V4 is now blocked by a separate ggml-core scheduler/gallocr bug for
   in-place view ops at a GPU→CPU boundary (see "Known gap: hybrid CPU/GPU view-allocation").
4. **Hybrid CPU/GPU view-allocation (`ROPE_BACK`)** — **fixed**: the scheduler now keeps
   in-place view ops on the view source's backend, so `-sm layer -cmoe` no longer crashes
   on a null device buffer (see "Hybrid CPU/GPU view-allocation" above).
5. **DSV4 flash-attention indexed + sinks** — **now implemented**: the indexed FA
   variant (`src[5]`, per-row top-k gather) and the per-head attention sinks (`src[4]`)
   run in a dedicated scalar kernel (`flash_attn_indexed.comp`) for F16/Q8_0 KV caches,
   matching the CUDA DSA reader and validated in `tests/test-dsa.cpp` (see "DSV4
   flash-attention coverage"). Dense FA *with* sinks (no indexer) is now implemented
   too (scalar + cm2 paths). **Remaining DSV4 gap → fixed**: the end-to-end gibberish
   was not an FA scheduling issue — it was the DSV4 "flipped" rope (see "DSV4
   flash-attention coverage"); the rope shaders now honor `op_params[15]` and the
   model decodes coherently on Vulkan.
4. **`--fit` with `GGML_BACKEND_DL`** — fixed: per-device memory is now queried through
   the backend registry (`ggml_backend_reg_get_device_memory`), so DL builds report real
   free memory (and `--fit` no longer sees 0 MiB).
5. **`-sm graph` / `-sm attn`** split modes. A backend-agnostic **split buffer type**
   and host-staged **`GGML_OP_REDUCE`** are now implemented, wired into
   `llama_default_buffer_type_split` for static / `GGML_BACKEND_DL` / mixed-backend
   selections, and validated end-to-end on Qwen3.6-27B-IQK across two Vulkan devices
   (see "Graph parallelism plumbing" below). Remaining: the `-sm attn` MLA
   `split_dim=2` load path, and a CUDA foreign-buffer REDUCE fallback for mixed
   CUDA+Vulkan runs.
6. Everything else: Mamba `SSM_SCAN`, the `*_R4` repacks and `IQ1_BN`/`IQ2_BN`, async
   tensor copies/events, the fence busy-wait, and the remaining training/vision ops
   (`GLU`, `RWKV_WKV6/7`, `CONV_2D_DW`, `SIN`/`COS`, ...).

### Op coverage (the big one)

These ops are produced by ik_llama's graph builder but are **not implemented** in the
Vulkan backend, so they fall back to the CPU backend with expensive copies:

- **Gated delta-net** (`qwen35` / `qwen3next` recurrent layers) is now implemented on
  Vulkan: `GGML_OP_SSM_CONV` (single-sequence fast path: parallel over rows and tokens,
  plus a separate final-state kernel), `GGML_OP_L2_NORM` (stride-aware; the shader was
  unwired and had an eps bug), `GGML_UNARY_OP_SOFTPLUS` and `GGML_OP_DELTA_NET` (a fused
  recurrent kernel, one workgroup per (head, seq), S_V ∈ {16,32,64,128} via spec
  constants, with the state row kept in registers). The delta-net op writes the new state
  into the result tail and lets the existing state-copy node persist it (the CPU-style
  src[7] fused-copy optimization is not used). Not yet supported (falls back to CPU):
  per-step conv/state checkpointing (`src[4]`/`src[6]` non-null) and multi-sequence
  `SSM_CONV` routing (`n_kv > 1`). `GGML_OP_REDUCE` (needed for multi-device splits of
  the layer) is now implemented as a host-staged all-reduce.
- **Indexer / DSA / CSA / HCA / GLM-DSA** (DeepSeek2/4, OpenPangu, GLM-DSA sparse
  attention): `GGML_OP_INDEXER_TOPK`, `GGML_OP_MASK_TOPK`, `GGML_OP_MASK_TO_IDX`,
  `GGML_OP_SINKHORN`, `GGML_OP_HC_PRE`, `GGML_OP_HC_POST`, `GGML_OP_LATENT_ATTN`,
  `GGML_OP_DS4_COMP` — **now implemented** (see "DSA / GLM-DSA / DeepSeek-V4
  sparse-attention ops" above). MLA ops (`mul_mat`/`mul_mat_id`/`concat`/`permute`/
  `flash_attn_ext`) are already supported, so MLA-only models run on Vulkan; only the
  `-sm attn` MLA `split_dim=2` load path remains.
- `GGML_OP_MULTI_ADD` exists but check the specific fused-mul-multiadd variants
  (`fused_mmad`); `-no-mmad` disables them

`GGML_OP_FUSED_UP_GATE` and `GGML_OP_MOE_FUSED_UP_GATE` are now implemented (see
"What we fixed"). Note that the fused up-gate is implemented as two matmuls + a combine
kernel rather than a single fused kernel, so on Vulkan it is roughly on par with the
`-no-fug` path rather than faster.

To close the remaining gaps, implement the missing ops as Vulkan kernels and extend
`ggml_backend_vk_supports_op` / `build_graph`.

### Quant type coverage

The Vulkan `MUL_MAT` supports `F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q6_0, Q8_0, MXFP4,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS,
IQ4_NL` plus the full **IQK/K-family** (`IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K, IQ2_KS, IQ3_KS,
IQ4_KS, IQ4_KSS, IQ5_KS, IQ2_KL`) and **KT-family** (`IQ1_KT, IQ2_KT, IQ3_KT, IQ4_KT`). The
decode path uses
native per-type `mul_mat_vec` kernels; prompt processing uses the flat dequant-to-F16 +
tensor-core matmul (the cm2 per-element inline dequant, scalar or V=4, turned out slower and
is no longer used for these types).

Still not supported (their matmuls run on CPU), in priority order:

- `IQ1_BN`, `IQ2_BN`, and the `*_R4` repacks (`IQ2_K_R4`, `IQ3_K_R4`, `IQ4_K_R4`,
  `IQ5_K_R4`, `IQ4_KS_R4`, `IQ5_KS_R4`, `IQ1_S_R4`, `IQ1_M_R4`).

### Performance / architecture

- `graph_compute` now **defers its drain**: batches are submitted without a fence and the
  per-split fence wait at the end of `graph_compute` was removed, so consecutive
  same-device splits overlap (the CPU records the next split while the GPU executes the
  previous one). Host/transfer operations flush pending compute lazily via
  `ggml_vk_device_flush_pending_compute`, and the scheduler's `synchronize` still drains at
  cross-device/host boundaries. This is not full async (no events / cross-queue
  semaphores yet; the transfer and compute queues are still ordered only by these host-side
  flushes), but it removed the ~63% per-split fence wait that profiling showed was the
  dominant `-sm graph` decode cost. On Qwen3.6-27B-IQK (2 Vulkan devices), `-sm graph`
  decode improved from ~13 to ~19 tok/s (`-ts 8,1`) and ~6.4 to ~9.8 tok/s (default split).
- UMA device buffers are now allocated as **cached host-visible system memory**
  (`HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`) instead of `DEVICE_LOCAL`, so host reads
  and writes on UMA devices (e.g. the all-reduce's iGPU side) become a direct `memcpy`
  instead of staging through the transfer queue (the iGPU get+set pair dropped from ~42 us
  to ~0.15 us). This lifts `-sm graph` decode further to ~24.5 tok/s (`-ts 8,1`),
  ~17.0 tok/s (`-ts 1,1`) and ~12.4 tok/s (default split).
- An opt-in **shared-buffer DMA reduce** (`GGML_VK_P2P=1`) is wired: it shares a buffer
  via `external_memory_fd` (host-visible on NVIDIA/OPAQUE_FD, where device-local import is
  same-device-only; `DEVICE_LOCAL` + DMA_BUF is kept for mesa) and reduces with
  `vkCmdCopyBuffer` + a GPU add. The remote partial is first copied into a per-device
  device-local staging buffer (also with `vkCmdCopyBuffer`), so the GPU add only ever touches
  two local buffers — cross-device storage-buffer coherence is not reliable on NVIDIA. It is
  off by default because the host-staged reduce is already near-optimal for UMA devices and
  the extra copies add fixed overhead; on two NVIDIA Blackwell GPUs it is numerically
  identical to the host-staged path on a 32-token decode. The diagnostic
  `tests/test-vk-p2p.cpp` exercises the underlying mechanisms and showed that NVIDIA does
  not support cross-device semaphore import (SYNC_FD is absent on Blackwell; OPAQUE_FD is
  same-device-only), so the DMA copies are host-fence ordered.
- **Real device-to-device P2P on NVIDIA Vulkan is not exposed natively**, but it *is*
  reachable through CUDA interop. A probe (export Vulkan `DEVICE_LOCAL` buffers via
  `OPAQUE_FD`, `cuImportExternalMemory` + `cuExternalMemoryGetMappedBuffer`, then
  `cuMemcpyPeer`) round-trips data between the two GPUs and measures **~44 GB/s in both
  directions** (PCIe P2P). Device groups are *not* a route on this stack: NVIDIA reports
  one physical device per `VK_KHR_device_group` group with no `VK_MEMORY_HEAP_MULTI_INSTANCE_BIT`
  heaps, and `VK_EXT_external_memory_dma_buf` is absent. So a true P2P reduce would need to
  (a) allocate the compute/tensor buffers with `VkExportMemoryAllocateInfo` (OPAQUE_FD),
  (b) import each buffer's fd into CUDA once and cache the device pointer, and (c) replace
  the cross-device leg of `GGML_OP_REDUCE` with `cuMemcpyPeer` + a local GPU add, ordered
  by Vulkan fence waits (CUDA copies are synchronous). That is the most promising remaining
  lever for closing the CUDA `-sm graph` gap.

  The CUDA interop layer is now implemented and **probe-gated** (see `ggml_vk_cuda_*` in
  `ggml-vulkan.cpp`): device-local buffers are allocated `OPAQUE_FD`-exportable, `libcuda.so.1`
  is dlopen'd at runtime, and a one-time probe verifies the import maps the same physical
  memory. Two hard-won details:

  - `cuMemcpyHtoD`/`cuMemcpyDtoH` must be dlsym'd as `cuMemcpyHtoD_v2`/`cuMemcpyDtoH_v2`:
    `cuda.h` remaps the unversioned names to the `_v2` variants, and the unversioned symbols
    in `libcuda.so.1` are legacy stubs that return `CUDA_ERROR_INVALID_CONTEXT`. Using the
    stubs made the import look like it mapped a fresh zeroed allocation.
  - The CUDA copy engine and the Vulkan compute/transfer engines do **not** share a coherent
    cache domain on NVIDIA. The probe validates CUDA-write→Vulkan-transfer-read (OK), but the
    reduce's compute add reading a CUDA-written staging buffer (and the peer copy reading a
    compute-written sum) is racy; routing through `vkCmdCopyBuffer` helps but is still not
    deterministic without a real cross-engine sync. The `cuMemcpyPeer` reduce is therefore
    opt-in (`GGML_VK_CUDA_P2P=1`) and off by default. A robust fix needs either a same-device
    CUDA↔Vulkan external semaphore (import the Vulkan timeline semaphore into CUDA with
    `cuImportExternalSemaphore` + `cuWaitExternalSemaphoresAsync`/`cuSignalExternalSemaphoresAsync`)
    or doing the whole reduce on the CUDA side.
- No events and no async tensor copies yet; the scheduler's `is_async` parallel path is
  still disabled for Vulkan. `-sas` works (it falls back to `synchronize`), but adds only
  ~3% for an equal split and nothing for a lopsided split, because the deferred drain
  already overlaps the two devices' compute and the all-reduce is the remaining
  serialization point.
- The spin in `ggml_vk_wait_for_fence` is still a busy-wait for the final fence; upstream
  keeps the same pattern, but a blocking `vkWaitForFences` for the tail would reduce CPU
  usage further at a small latency cost.
- `-sm graph`/`-sm attn` split modes previously fell back to the layer split path
  because there was no split buffer type available to the generic model loader. A
  backend-agnostic split buffer type and a host-staged `GGML_OP_REDUCE` are now
  implemented in ggml core, so the `-sm graph` tensor-parallel path can schedule across
  multiple Vulkan devices — and across mixed backends in principle (validated with a
  synthetic cross-device reduce through the ggml scheduler and a mixed Vulkan0+CPU split;
  not yet validated end-to-end on a real model). `-sm attn` (MLA) still needs the
  `split_dim=2` load/get path.

### Integration

- Per-device memory is queried through the backend registry
  (`ggml_backend_reg_get_device_memory`), with each backend publishing its
  `get_device_memory` function at registration. This works for both static and
  `GGML_BACKEND_DL` builds, so `--fit` and the device list no longer report 0 MiB in DL.
- RPC servers are not part of the backend registry; they are appended after the registered
  backends in `model->devices`.
- The backend now compiles even when `glslc` does not support `GL_EXT_integer_dot_product`
  or `GL_NV_cooperative_matrix_decode_vector` (so `GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT` /
  `GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT` are undefined). In that configuration
  `ggml-vulkan.cpp` uses the non-`dot4` IQK/KT `mul_mat_vec` / `mul_mat_vec_id` /
  flat-dequant shaders (the `_dot4` and Q8_1-activation paths are compiled out, and
  `ggml_vk_strip_decode_vector()` becomes a no-op). This keeps the backend buildable with
  older toolchains such as Ubuntu's `shaderc 2023.8` / `glslang 14.0.0`, at the cost of the
  dot4 decode speedup on integer-dot-capable GPUs.
- **Ubuntu build-toolchain gotcha**: a proper build needs a `glslc` that understands
  `GL_NV_cooperative_matrix2` and `GL_EXT_integer_dot_product`, plus Vulkan headers that
  define `VK_NV_cooperative_matrix2`. Ubuntu 24.04's system `glslc` (`shaderc 2023.8` /
  `glslang 14.0.0`) is too old: the CMake feature tests fail *silently* and the build comes
  out without `GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT` and
  `GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT`, so the device line prints
  `matrix cores: KHR_coopmat` / `int dot: 0` and the backend falls back to coopmat1
  (16x16x16) F16 matmul + the scalar (non-dot4) KT decode. On a 32B IQ4_KT model that is
  ~3.2x slower PP and ~1.65x slower TG than a proper build, even though the driver fully
  supports coopmat2 and integer dot. Fix: install the LunarG Vulkan SDK (or newer
  `shaderc` + `vulkan-headers`, e.g. the LunarG PPA `packages.lunarg.com/vulkan`; note the
  PPA splits headers into a separate `vulkan-headers` package), then reconfigure from a
  clean build dir. Verify: `glslc --version` (need shaderc 2024+ / glslang 15+; the LunarG
  shaderc package prints a misleading "v2023.8" banner with a 2025 build date, which is
  fine), `grep -c VK_NV_cooperative_matrix2 /usr/include/vulkan/vulkan_core.h` (>= 1), and
  confirm `GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT` / `GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT`
  appear in the build's `flags.make`.

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
- `tests/test-delta-net.cpp` validates `SSM_CONV`, `L2_NORM`, `SOFTPLUS` and `DELTA_NET`
  against the CPU reference on a target backend: single- and multi-token delta-net, both
  repeat types, several head sizes (the multi-token CPU reference is only trustworthy for
  head_dim 64/128 where `iqk_fused_delta_net` handles the v strides). Run as
  `test-delta-net CPU|Vulkan0|CUDA0`.
- `tests/test-dsa.cpp` validates `SINKHORN`, `HC_PRE`, `HC_POST`, `DS4_COMP`, `MASK_TOPK`,
  `MASK_TO_IDX`, `INDEXER_TOPK` and `LATENT_ATTN` against the CPU reference on a target
  backend (all S sizes, F32/F16/Q8_0 latent caches, both DS4 compression types, dense +
  indexed latent attention), plus the DSV4 **indexed flash attention + sinks**
  (`GGML_OP_FLASH_ATTN_EXT` with `src[5]`/`src[4]`: single- and multi-token, F16/Q8_0
  KV, with/without sinks, trailing `-1` index padding, and the 512/512 head shape). Run
  as `test-dsa CPU|Vulkan0|CUDA0`. INDEXER_TOPK is compared as a per-row *set* because
  the CPU reference's bucket top-k is not sorted.
- `tests/test-iqk-quants.cpp` validates the 15 IQK/KT types against the scalar dequant
  reference (the format definition): single-token decode, small batches, larger K,
  multi-token (dequant-to-F16 path), MoE (`MUL_MAT_ID`) and `GET_ROWS` (quantized token
  embeddings, both single-block and multi-block rows). Run as
  `test-iqk-quants CPU|Vulkan0|CUDA0`. The KT family is allowed a looser tolerance because
  of its calibration factor (see above). Note: on `Vulkan0` the multi-token `MUL_MAT`
  cases take the dequant-to-F16 path on both `Vulkan0` and `Vulkan1` (the cm2 inline-dequant
  path is no longer used for the IQK/KT types), so both `MUL_MAT` and `MUL_MAT_ID` exercise
  the flat dequant kernels. The single-token KS/KL/KT cases previously failed on `Vulkan1`
  because the `mul_mat_vec` A subbuffer omitted the per-row scale header (the last rows read
  out of range); the vec dispatch now uses `ggml_nbytes(src0)` for the raw-weight case, and
  all 15 types pass on both `Vulkan0` and `Vulkan1`.
- End-to-end benchmark (RTX 3090, Qwen2.5-Coder-32B-Instruct-IQ4_KT):
  `llama-server -m <model> -dev Vulkan0 -c 4096 -ub 2048 -b 2048 -n 256 -nocb --host 127.0.0.1`
  Send a tiny warmup prompt first — the first prompt compiles the Vulkan pipelines — then
  send one measurement request to `/completion` with
  `{"prompt": ..., "n_predict": 256, "temperature": 0.0, "stream": false}` and read
  `timings.prompt_per_second` / `timings.predicted_per_second` from the response (do not
  send the big prompt twice; the prompt cache makes the second request report only 1
  prompt token). The ~3000-token prompt is generated from a fixed seed 12345 (20-word
  vocabulary, 2600 words). `-nocb` avoids a continuous-batching `vk::DeviceLostError`
  during prompt processing on this model. On driver 610.57.04, with the Q8_1 decode fix
  IQ4_KT measures ~1110 tok/s prompt / ~24.9 tok/s generation (without it ~1105 tok/s
  prompt / ~23.6 tok/s generation); IQ4_KS measures ~1042 tok/s prompt / ~27.2 tok/s
  generation with the Q8_1 + shared table decodes for IQ4_KS and its IQ5_K attention-v
  tensors (vs ~4.1 tok/s before).
- Blackwell (RTX PRO 4000, entry-level Blackwell — ~59% of the 3090's CUDA/tensor cores,
  ~72% of its memory bandwidth, so absolute numbers are lower): with a proper build (see
  the toolchain note in "Integration"), IQ4_KT runs at TG parity with CUDA (~16.8 vs
  ~16.1 tok/s) and PP ~1.14x of CUDA at `-ub 4096` (738 vs 839 tok/s; ~683 tok/s at
  `-ub 2048`, ~528 tok/s at the default `-ub 512`). A coopmat2 F16 tile sweep (BK=32/128,
  BN=128 via an env override) showed the Ampere-tuned config (BK=64, BN=256) is already
  optimal — every variant regressed (BK=32 −12%, BK=128 −28%, BN=128 −21%). The residual
  ~1.12-1.14x PP gap is the dequant+F16 structure, not tile sizing.
  Decode-only measurements use `llama-cli ... -n 64 --temp 0` and the `eval time` line.
  CUDA reference: the same server command with `-dev CUDA0`.
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

- **IQ5_K decode was slow** (fixed): the Q8_1 `mul_mat_vec_iq5_k` shader is now a CUDA-style
  8-threads-per-block port (each thread dots two 16-element sub-blocks against two Q8_1
  blocks, matching `vec_dot_iq5_k_q8_1`), reads the weights through a uint32 buffer view, and
  looks the 5-bit `iq5nl_values` table up through a 64-entry direct-indexed shared-memory
  array. A full IQ5_K 32B model now decodes at ~25.5 tok/s (was ~12.4 tok/s). The earlier
  8-thread attempt had been reverted for a single-token vec correctness bug; the re-derived
  byte addressing (qs/qh/scales/extra) is validated by `test-iqk-quants`. Note that
  IQ4_KS/IQ4_KT/IQ2_K models store some attention tensors as IQ5_K (e.g. IQ4_KS `attn_v`),
  so they also benefit. The F32/F16 `mul_mat_vec_iq5_k` fallback (non-integer-dot devices and
  the `MUL_MAT_ID` vec path) is still the 16-thread byte-addressed shader.
- **TG decode numbers after the uint32-view / 8-thread / Q8_1 rework** (RTX 3090, 32B
  Qwen2.5-Coder, `-c 4096 -n 128 --temp 0`; the 27 GB IQ6_K uses
  `-dev Vulkan0,Vulkan1 -ts 8,1`):

  | quant | size | TG tok/s | | quant | size | TG tok/s |
  |---|---|---|---|---|---|---|
  | IQ1_KT | 7.6 GB | 43.4 | | IQ3_KS | 13.3 GB | 26.0 |
  | IQ2_KT | 9.6 GB | 42.4 | | IQ4_KT | 16.7 GB | 32.9 |
  | IQ2_KS | 9.4 GB | 34.5 | | IQ4_KSS | 16.9 GB | 33.7 |
  | IQ2_K | 10.3 GB | 33.4 | | IQ4_KS | 17.9 GB | 29.2 |
  | IQ2_KL | 11.3 GB | 35.0 | | IQ4_K | 18.7 GB | 28.2 |
  | IQ3_KT | 13.7 GB | 26.8 | | IQ5_KS | 21.6 GB | 30.0 |
  | IQ3_K | 14.3 GB | 25.3 | | IQ5_K | 22.6 GB | 25.5 |
  | | | | | IQ6_K | 27.1 GB | 19.4 (2 GPU) |

  The 1-2 bit quants decode at ~33-43 tok/s and the 3-6 bit at ~25-33 tok/s (IQ6_K is the
  2-GPU outlier). The base `_K` types store some attention tensors in higher-precision
  quants (e.g. IQ2_K keeps attn_output/attn_v as IQ3_K/IQ4_K), so their decode reflects a
  mix. IQ3_K's 110-byte block is only 2-byte aligned, so it uses an unaligned-safe uint32
  loader and stays slightly behind the 4-byte-aligned quants. (Alternatives were tried and
  regressed: a branchless select load and a 3-aligned-word pair loader both measured slower
  than the plain 1-or-2-load branch — the divergence is cheap and the extra always-on
  loads/shifts are not.) Dump a model's per-tensor types with
  `gguf-py/scripts/gguf_dump.py <model.gguf>` (the IQ*_K models mix a few higher-precision
  attention tensors and an output tensor, e.g. IQ3_K keeps attn_v as IQ4_K and output as Q5_K,
  IQ4_K keeps attn_v as IQ5_K and output as Q6_K).
- **PP numbers** (RTX 3090, same 32B models, 2600-token fixed-seed prompt,
  `-ub 2048 -b 2048 -nocb -n 1`, single `Vulkan0` except IQ6_K):

  | quant | PP tok/s | | quant | PP tok/s |
  |---|---|---|---|---|
  | IQ1_KT | 1074 | | IQ3_KS | 1062 |
  | IQ2_KT | 1072 | | IQ4_KT | 1070 |
  | IQ2_KS | 1108 | | IQ4_KSS | 1005 |
  | IQ2_K | 1108 | | IQ4_KS | 1009 |
  | IQ2_KL | 1102 | | IQ4_K | 1015 |
  | IQ3_KT | 1127 | | IQ5_KS | 941 |
  | IQ3_K | 1134 | | IQ5_K | 937 |
  | | | | IQ6_K | 451 (2 GPU) |

  PP goes through the dequant-to-F16 + F16 tensor-core matmul path (not the `mul_mat_vec`
  Q8_1 shaders), which was already optimized. It is dominated by the fixed per-layer F16
  matmul, so it barely depends on quant bit-width (~940-1134 tok/s for the single-device
  quants, i.e. within the docs' ~1.1x-of-CUDA figure). `-nocb` matters: without it IQ2_KS
  showed a spurious ~872 tok/s and IQ3_KT hit a `vk::DeviceLostError`. IQ6_K is the only
  laggard, limited by the multi-GPU split (17% of tensors on the ~3.7x-slower AMD iGPU plus
  per-split cross-device sync) rather than the IQ6_K decode.
  **PP optimization candidates** (see also "vs CUDA" below): (1) the F16 tensor-core matmul
  runs at ~75% of FP16 peak — tile-size / split-k tuning of the generic coopmat2 matmul lifts
  every quant uniformly; (2) for >24 GB models, tune the tensor-split heuristic to keep the
  output projection and more FFN on the fast device, and implement the async tensor copies /
  events (still NULL in the backend interface) so cross-device transfer overlaps compute;
  (3) the flat dequant kernels still feed `MUL_MAT_ID` (MoE) and non-coopmat2 devices — the
  V=4 coopmat2 inline dequant is available but ~1.5x slower than dequant+F16 today.
- The decode (`mul_mat_vec`) kernels still dequantize per element with the KT-family
  multiplicative-hash decode (4 hash rounds per weight); the output projection
  `[5120, 152064]` alone is ~0.7 ms/token.
- The scheduler no longer copies the IQ4_KT FFN weights per split; the remaining
  device→host traffic is limited to the logits.

### vs CUDA: the prompt-processing gap and the paths to close it

On the same RTX 3090 (driver 610.57.04) the IQ4_KT model now runs at ~1110 tok/s prompt
(`-ub 2048`, fixed-seed 3001-token prompt) and ~24.9 tok/s decode on Vulkan, vs
~1240 / ~30 on CUDA (prompt gap ~1.12x, decode gap ~1.20x). The decode (TG) gap is mostly *fixed*, not context-dependent: measured
40.7 ms/token at 17-token context vs 42.5 ms/token at 4399-token context (~0.4 us per
context token). The small growth is the KV-cache read in the decode flash-attention
path, which uses the scalar shader for single-token queries (`N == 1` falls back from
coopmat2 to FA_SCALAR). Note: a 4-token server timing can read ~31 tok/s because the
first token's decode is counted against the prompt eval; 64+ tokens (llama-cli "eval
time" and server "tg") agree at ~24.9 tok/s.

Steady-state per-op profiling (GPU clocks locked by running many iterations) shows both
remaining gaps are dominated by the FFN:

- **Decode** (1.20x gap): the `mul_mat_vec` FFN was ALU-bound on the KT hash decode
  (~320 GB/s weight reads, well below peak). Replacing the per-round byte sum with one
  `dot4` (`dotPacked4x8EXT`) cut the fused gate+up from ~441 us to ~258 us and the FFN
  down from ~237 us to ~138 us per layer, taking decode from ~17.7 to ~23 tok/s. The
  activation dot is now also `dot4`: the decode path quantizes the F32 activations to Q8_1
  and the KT-family `mul_mat_vec` kernels dot the packed-int8 hash values against the Q8_1
  blocks (mirroring CUDA's `vec_dot_iq{1,2,3,4}_kt_q8_1`), removing the remaining scalar-FMA
  activation dot; end-to-end decode is now ~24.9 tok/s (vs ~23.6 without the Q8_1 activation
  dot).
- **Prompt** (1.12x gap): prompt processing takes the mat-mat path for IQK/KT types, which
  dequantizes each weight matrix to F16 on the GPU and runs the F16 matmul (~4.4 ms per
  FFN matmul: ~1.5 ms dequant + ~2.9 ms tensor-core matmul). CUDA has a native quantized
  matmul (`mmq`) that reads the quantized weights once. Three approaches were explored for
  adding one to Vulkan:
  * **The SIMT dot4 mmq (`mul_mmq.comp` with Q8_1 activations) is now implemented for
    `IQ4_KT`** (the only IQK/KT type wired up so far; the other row-meta types could follow
    the same pattern): a byte-addressed tile loader runs the hash decode with `dot4` and
    packs the values as signed int8, then the standard dp4a vec-dot accumulates against
    Q8_1 activations — the same shape as CUDA's `load_tiles_iq4_kt`. Two bugs were found
    and fixed while wiring it up: the raw-weights binding must use `ggml_nbytes()` (the
    `type_size*ne/blck` size omits the per-row META header, so buffer robustness zeroed the
    last rows), and the mmq path must not dispatch the `y_non_contig` src1 copy (it is only
    budgeted when `qy_needs_dequant`, and on coopmat2 `y_non_contig` is forced true for f32
    activations). On the RTX 3090 it is *correct but slower* than the dequant+F16 path
    (261 vs 489 tok/s PP): SIMT dp4a cannot beat the tensor-core F16 matmul, matching the
    earlier reverted attempt. It is therefore enabled only on devices without cooperative
    matrices (AMD/Intel), where the F16 matmul has no tensor cores and reading the
    quantized weights once can win.
  * **The coopmat2 tensor-core inline dequant (`mul_mm_cm2.comp`) is now implemented for
    all 15 IQK/KT types**: the A matrix is loaded one element at a time through a decode
    function (`dequantFuncIQ*` in `dequant_funcs_cm2.comp`) that computes the element's
    absolute byte address from the push constants (`blockCoords[1]` is the 256-element
    block index, `coordInBlock[1]` the element within it) and reads the quantized bytes
    through the byte view of binding 0. The KT hash uses one multiply per element via
    precomputed powers of `0xCBAC1FED` (and a hardware `dot4` for the byte sum). The
    mat-mat path (`MUL_MAT`) now uses these pipelines directly instead of dequant-to-F16.
    Two NV_coopmat2 driver behaviors required workarounds: the decode only receives
    per-element coordinates on the **large** tile config (small/medium configs collapse
    `coordInBlock` for part of the tile, so the pipelines are forced to large and only the
    `l`/`a_l` variants are created), and the `MUL_MAT_ID` path invokes the decode once per
    4-element group (`coordInBlock[1] = col/4`, no per-element offset), so the IQK types
    are kept off the coopmat2 matmul_id path and retain the dequant-to-F16 fallback there.
    Correctness is covered by `tests/test-iqk-quants.cpp` (multi-token mat-mat cases now
    exercise the cm2 path).

    Performance reality check (RTX 3090, real FFN shape `[27648, 5120] x [27648, n]` —
    Qwen2.5-32B's `feed_forward_length` is 27648): the matmul costs ~15-17 ms for **any**
    n (32 to 512); the cost is a fixed ~15 ms A-side per-element decode (~141 M driver
    invocations at ~100 cycles each), and the tensor-core multiply is only ~2 ms. The
    dequant+F16 path is the same (~16-23 ms measured): its flat dequant kernel is also
    ~10x off memory-bound (141 M elements, ~350 MB read / 280 MB write, would be ~1 ms).
    Model-level prompt processing is unchanged (~513 tok/s at batch 2048 on the 32B
    IQ4_KT; ~11k tok/s on the 0.5B quants where IQ4_KS PP is marginally faster than
    IQ4_KT). The docs' earlier hope of a 2-3x prompt speedup assumed the A-side dequant
    was cheap; on Ampere it is the bottleneck in every form tried.

    A V=4 vector-decode path (`GL_NV_cooperative_matrix_decode_vector`, one driver
    invocation per 4 elements with shared row-header/block-selector reads) is implemented
    for all 15 types with SPIR-V stripping for devices without the capability. It was
    first exercised on the NVIDIA beta driver 595.44.14 (which exposes the extension and
    is now enabled at device creation; previously the feature was queried into the wrong
    struct and never enabled). On the RTX 3090 it is correct for IQ4_KT but still ~1.5x
    slower than the flat dequant+F16 path at n=2048 (9.6 vs 6.7 ms per FFN matmul),
    because the inline decode is re-done once per N-tile while the flat dequant runs once
    per batch; several other types' V=4 decoders are also buggy (e.g. IQ2_KL). The
    dequant+F16 path is therefore used for all IQK/KT MUL_MAT on coopmat2.

  * **The INT8 coopmat1 (KHR) tensor-core mmq (`mul_mmq.comp` with `coopmat<int8_t>`)
    was also tried for `IQ4_KT`**: the byte-addressed tile loader packs the hash decode
    into signed int8 and accumulates with `coopMatMulAdd` on the fixed `16x16x32` SINT8
    coopmat1 shape (coopmat2 exposes no INT8; only KHR coopmat1 reaches the int8 tensor
    cores). The `COOPMAT` path in `mul_mmq.comp` turned out to be unfinished dead code
    (undeclared scale buffers, stale variable names), so it was fixed and wired up as an
    experiment. It is correct (`test-iqk-quants` passes) but ~8x *slower* than the
    dequant+F16 path on the RTX 3090 (133 vs 1075 tok/s PP at `-ub 2048`): the
    16x16x32 subgroup-scoped int8 tiles cannot compete with coopmat2's flexible 128x256
    F16 tiles, and IQ4_KT's per-32-element `dl` scale forces a float scale-multiply per
    K-tile, which consumes the whole tensor-core benefit. Reverted; dequant+F16 remains
    the fastest known prompt path for IQK/KT on tensor-core GPUs.

    **The flat dequant kernels are now optimized, and dense IQK/KT MUL_MAT uses them.**
    The 15 `dequant_iqX_*` shaders keep the original 8-threads-per-block mapping but now
    read the quantized bytes through a uint32 view (aliasing the byte view at binding 0),
    write `f16vec4` stores, and (on integer-dot devices) use a hardware `dot4` for the
    KT-family hash byte-sum (`dequant_iqX_dot4` variants). On the RTX 3090 the
    dequant+`MUL_MAT_ID` FFN matmul (`[27648, 5120] x [27648, n]`) drops to ~1.3-2.3 ms
    (n=32) from ~3.8 ms, now faster than the scalar cm2 inline dequant (~3.1 ms). The
    dense `MUL_MAT` path therefore falls back to dequant+F16 for the IQK/KT families
    (the cm2 inline-dequant path, scalar or V=4, is not used for these types). End-to-end prompt processing on the
    32B IQ4_KT model improves from ~500 tok/s to ~850 tok/s at the default ubatch (512);
    with `-ub 2048` (or `-ub 4096`) it reaches ~1090-1110 tok/s, within ~1.1x of CUDA
    (~1240 tok/s), because the larger batch uses the tensor cores more efficiently and
    dequantizes the weights fewer times. The remaining gap is the F16 tensor-core matmul
    itself (running at ~75% of the FP16 peak).

    Note that the earlier ~15-17 ms cm2 / ~16-23 ms flat-dequant numbers above were
    cold-clock artifacts; with 20-iteration warmup the cm2 FFN matmul is ~3.1 ms (n=32)
    to ~4.7 ms (n=512).

    While investigating this, a pre-existing crash was also fixed:
    `ggml_vk_get_mul_mat_mat_pipeline` returned a non-null but empty
    `pipeline_dequant_mul_mat_mat[type]` struct for the IQK/KT types on coopmat1
    (non-coopmat2) devices (only the coopmat2 cm2 variants are ever created for these
    types), so multi-token MUL_MAT dereferenced null pipeline entries instead of falling
    back to dequant+F16 (segfault on the Strix Halo iGPU). The matmul-pipeline lookup
    now returns nullptr when the selected struct has no compiled l/m/s/a_* variants,
    which takes the intended dequant+F16 fallback.
- The base IQK types (IQ2_K..IQ6_K) have per-16-element dequant scales, which fit neither
  the SIMT mmq's per-32-group scale nor a simple cm2 tile without per-16 handling.

An earlier symptom (the model generating "!" repeatedly) was from a pre-fix build (wrong
`ql`/`qh` offsets and 16-bit reads in the IQ4_KT kernels); the current build generates
coherent text.
