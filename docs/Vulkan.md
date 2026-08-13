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
  "vs CUDA" for the measured result). The `*_R4` repack variants and `Q6_0`, `MXFP4`,
  `IQ1_BN`, `IQ2_BN` are still not supported and fall back to the CPU backend.

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
- **decode Q8_1 activations (KT family)**: on integer-dot devices the KT-family decode
  (`mul_mat_vec`) path now quantizes the F32 activations to Q8_1 and the
  `mul_mat_vec_iq{1,2,3,4}_kt_q8_1` shaders dot the packed-int8 hash decode against the Q8_1
  blocks with `dotPacked4x8EXT`, mirroring CUDA's `vec_dot_iq{1,2,3,4}_kt_q8_1`. This removes
  the scalar-FMA activation dot from the decode FFN (selected for contiguous F32 activations).
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

### Priority (highest first)

1. **Gated delta-net** (`qwen35` / `qwen3next`): `SSM_CONV`, `L2_NORM`, `SOFTPLUS`,
   `DELTA_NET`. Stateful, so CPU fallback produces garbage logits — these models cannot
   run on Vulkan at all today.
2. **`Q6_0`** — the only legacy 6-bit quant still missing from `MUL_MAT`.
3. **`MXFP4`** — the micro-scaling 4-bit format.
4. **Indexer / DSA / CSA / HCA / GLM-DSA**: `INDEXER_TOPK`, `MASK_TOPK`, `MASK_TO_IDX`,
   `SINKHORN`, `HC_PRE`, `HC_POST`, `LATENT_ATTN`, `DS4_COMP`. Stateful sparse-attention
   ops (DeepSeek2/4, OpenPangu, GLM-4.5-Air, GLM-DSA).
5. **`--fit` with `GGML_BACKEND_DL`** — per-device memory reports 0 MiB.
6. **`-sm graph` / `-sm attn`** split modes.
7. Everything else: Mamba `SSM_SCAN`, the `*_R4` repacks and `IQ1_BN`/`IQ2_BN`, async
   tensor copies/events, the fence busy-wait, and the remaining training/vision ops
   (`GLU`, `RWKV_WKV6/7`, `CONV_2D_DW`, `SIN`/`COS`, ...).

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
- **Indexer / DSA / CSA / HCA / GLM-DSA** (DeepSeek2/4, OpenPangu, GLM-4.5-Air, GLM-DSA
  sparse attention): `GGML_OP_INDEXER_TOPK`, `GGML_OP_MASK_TOPK`, `GGML_OP_MASK_TO_IDX`,
  `GGML_OP_SINKHORN`, `GGML_OP_HC_PRE`, `GGML_OP_HC_POST`, `GGML_OP_LATENT_ATTN`,
  `GGML_OP_DS4_COMP`. CUDA implements all of these; Vulkan has none, so these
  architectures fall back to the CPU backend (with the same stateful round-trip problem
  as delta-net).
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
native per-type `mul_mat_vec` kernels; prompt processing uses the flat dequant-to-F16 +
tensor-core matmul (the cm2 per-element inline dequant, scalar or V=4, turned out slower and
is no longer used for these types).

Still not supported (their matmuls run on CPU), in priority order:

- `Q6_0` and `MXFP4` (the highest-value missing quants).
- `IQ1_BN`, `IQ2_BN`, and the `*_R4` repacks (`IQ2_K_R4`, `IQ3_K_R4`, `IQ4_K_R4`,
  `IQ5_K_R4`, `IQ4_KS_R4`, `IQ5_KS_R4`, `IQ1_S_R4`, `IQ1_M_R4`).

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
  of its calibration factor (see above). Note: on `Vulkan0` the multi-token `MUL_MAT`
  cases take the dequant-to-F16 path on both `Vulkan0` and `Vulkan1` (the cm2 inline-dequant
  path is no longer used for the IQK/KT types), so both `MUL_MAT` and `MUL_MAT_ID` exercise
  the flat dequant kernels. The single-token KS/KL/KT cases currently fail on `Vulkan1`
  (a pre-existing `mul_mat_vec` row-meta issue on that device, unrelated to the dequant
  kernels).
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
  this measures ~1110 tok/s prompt / ~24.9 tok/s generation; without it ~1105 tok/s
  prompt / ~23.6 tok/s generation. Decode-only measurements use `llama-cli ... -n 64 --temp 0`
  and the `eval time` line. CUDA reference: the same server command with `-dev CUDA0`.
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
  matmul (`mmq`) that reads the quantized weights once. Two approaches were explored for
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
