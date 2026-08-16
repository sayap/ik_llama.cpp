# Handover: sparse-attention (DSA / GLM-DSA) ops for the Vulkan backend

Goal: implement the missing sparse-attention ops so GLM-5.2 (`glm-dsa`) and DeepSeek V4
(`deepseek4`) run on Vulkan instead of falling back to the CPU backend.

## Current state

- `f9556e92` — lean lazy host-staged `-sm graph` reduce (N devices); 2-GPU decode beats
  single-GPU, byte-identical output.
- `76d4978c` — doc notes for the AMD/mesa async P2P reduce follow-up.
- The 8 ops below are implemented in CUDA and in the CPU reference, but **zero in Vulkan**.

## The 8 missing ops (CUDA dispatch, `ggml/src/ggml-cuda.cu` ~4275-4300)

| op | CUDA kernel fn | CUDA file | CPU reference (`ggml.c`) |
|---|---|---|---|
| `INDEXER_TOPK` | `ggml_cuda_op_indexer_topk` | `indexer_topk.cu` (15.7 KB) | ~24200 region |
| `MASK_TOPK`    | `ggml_cuda_op_indexer_mask`  | `indexer_topk.cu` | ~24200 region |
| `MASK_TO_IDX`  | `ggml_cuda_op_mask_to_index` | `indexer_topk.cu` | `ggml_compute_forward_mask_to_idx` (~24549) |
| `SINKHORN`     | `ggml_cuda_op_sinkhorn`       | `sinkhorn.cu` (12.8 KB) | `ggml_compute_forward_sinkhorn` (~24234) |
| `HC_PRE`       | `ggml_cuda_op_hc_pre`         | `sinkhorn.cu` | `ggml_compute_forward_hc_pre` (~24311) |
| `HC_POST`      | `ggml_cuda_op_hc_post`        | `sinkhorn.cu` | `ggml_compute_forward_hc_post` (~24406) |
| `LATENT_ATTN`  | `ggml_cuda_op_latent_attn`    | `latent_attn.cu` (30.7 KB) | `ggml_compute_forward_latent_attn` (~24795) |
| `DS4_COMP`     | `ggml_cuda_op_ds4_comp`       | `ds4_comp.cu` (5.6 KB) | `ggml_compute_forward_ds4_comp_type0/1` (~24595/24680) |

There is also `ggml_cuda_op_latent_attn_indexed` (a fused indexed variant) and
`ggml_cuda_op_indexer_mask` — read `ggml_cuda_compute_forward` and the builder output
to see which variants the graphs actually emit before porting extras.

## Architecture mapping

- **GLM-5.2** = `glm-dsa` → `src/graphs/build_deepseek2.cpp` (shared DSA block with
  DeepSeek2, Mistral4, BailingMoE3, OpenPangu). Needs the **7-op DSA family**, no
  `DS4_COMP`.
- **DeepSeek V4** = `deepseek4` → `src/graphs/build_deepseek4.cpp` → the DSA family
  **plus `DS4_COMP`**.

So the 7-op DSA family unblocks ~5 arches at once; `DS4_COMP` is the DeepSeek4 extra.

## Test models

- **GLM-5.2**: ~200 GB, MoE. No local `.gguf` yet (being copied). Run with
  `-dev Vulkan0 -ngl 99 -cmoe -ub 2048` — `-cmoe` keeps the MoE FFN (the bulk) in CPU
  memory; the attention/DSA ops stay on-GPU, which is exactly what we need to validate.
- **DeepSeek-V4-Flash-0731** (`full` / `fp8-ik`): ~156 GB each, `deepseek4`, MoE. Run
  with `-dev Vulkan0 -ngl 99 -ncmoe 37 -ub 2048` (first 37 layers' MoE on CPU).
- **No `-sm graph` for `deepseek4` yet** — test single-device + CPU offload.
- Tensor types are all Vulkan-supported: `F32`/`Q8_0`/`BF16`/`MXFP4` (+ 3 `I32`
  indexer-state tensors). No new quant-type work needed for these models.
- Neither model fits VRAM (3×24 GB = 72 GB), so `-cmoe`/`-ncmoe` is the test path, and
  the **dev loop must not depend on the big model** (see test harness below).

## Vulkan integration checklist (one op at a time)

Follow the `DELTA_NET` pattern in `ggml/src/ggml-vulkan.cpp`:

1. `ggml_backend_vk_supports_op` (~line 12457) — accept the op and its actually-supported
   type combinations (do **not** return `true` unconditionally; match the real shader
   variants, like the `RMS_NORM` fix did).
2. `ggml_vk_build_graph` op switch (~line 11190) — add the `case GGML_OP_*:`.
3. `ggml_vk_compute_forward` dispatch (~line 11531) — call a dedicated
   `ggml_vk_<op>(ctx, compute_ctx, node, dryrun)` function (the `ssm_conv`/`delta_net`
   calls are right there).
4. Dedicated op function — model on `ggml_vk_delta_net` (~line 9063): request the
   pipeline (dryrun), bind buffers, set push constants, dispatch. Return `false` for
   unsupported shapes/types instead of asserting, so the scheduler falls back to CPU.
5. GLSL shader — `ggml/src/vulkan-shaders/<op>.comp`, registered in
   `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp` via
   `string_to_spv("<op>_f32", "<op>.comp", {})` (see `delta_net`/`ssm_conv` entries).
   Rebuild the `vulkan-shaders-gen` target so the SPIR-V is regenerated.
6. The dryrun short-circuit list (~line 12766) is only for generic `ggml_vk_op_f32` ops;
   dedicated ops with their own function don't go through it.

## Test harness

Create `tests/test-dsa.cpp` modeled on `tests/test-delta-net.cpp`:

- `ggml_backend_load_all()`, then `ggml_backend_reg_init_backend_from_str("Vulkan0")`.
- Build each op as a small ggml graph, run on `backend_cpu` (reference) and the target
  backend, compare with a tolerance (`closef`).
- Covers single- and multi-token, plus the stateful cases where applicable.
- DL-build gotcha: the test links only `libggml.so`; Vulkan symbols live in
  `libggml-vulkan.so`. Either dlopen the module (the delta-net test is built only in
  static builds — see `tests/CMakeLists.txt`), or build in a static Vulkan config for
  tests. This was a friction point earlier.

The CPU reference in `ggml.c` is the ground truth; the CUDA kernels are the performance
reference (and often the easiest to read for the exact indexing math).

## Suggested order

1. `INDEXER_TOPK` — self-contained, everything downstream consumes its output.
2. `MASK_TOPK` / `MASK_TO_IDX` / `SINKHORN` — the selection pipeline.
3. `HC_PRE` / `HC_POST` / `LATENT_ATTN` — the latent-attention block (largest).
4. `DS4_COMP` — small; unblocks DeepSeek V4 end-to-end.

Validate each op in `test-dsa.cpp` before moving on; do the GLM-5.2 smoke test after the
7-op family is in, then DeepSeek V4 after `DS4_COMP`.

## Pitfalls learned this session (read before writing any kernel)

- **Fence lifecycle**: reset `dev->fence` **after** every `waitForFences`, never leave
  it signaled — the next submit+wait then returns immediately and races (this caused a
  nasty decode-corruption bug).
- **Memory barriers across pipeline stages**: `vkCmdCopyBuffer` is a *transfer-stage*
  command even on a compute queue. `ggml_vk_sync_buffers` uses the queue's own stage on
  both sides and does **not** cover it. Use explicit
  `COMPUTE_SHADER→TRANSFER` / `TRANSFER→COMPUTE_SHADER` `pipelineBarrier`s around
  compute-produced/consumed copies.
- **Stateful ops**: the DSA indexer has cross-token state (the 3 `I32` tensors). The
  delta-net op writes new state into the result tail and lets a `CPY` node persist it —
  expect the same shape of plumbing for the indexer; the CPU fallback's "stateful
  round-trip problem" is why these archs currently break on Vulkan, not just slow.
- **Two-pass dispatch**: `build_graph` runs a dryrun (descriptor budgeting) then a
  record pass. Request pipelines in the dryrun; touch buffers only in the record pass.
- **`supports_op` must be honest** about type combos or you get null-pipeline crashes
  (the `FUSED_RMS_NORM` f16 fix and the coopmat1 IQK null-pipeline segfault were both
  this class).
- **Toolchain**: needs a `glslc` with `GL_EXT_integer_dot_product` /
  `GL_NV_cooperative_matrix2` (LunarG SDK, not Ubuntu's system shaderc) — irrelevant for
  these simple scalar/table kernels, but any kernel using `dot4`/coopmat silently
  degrades on a stale toolchain.

## Quick commands

```bash
# build the backend + shaders
cd build-dl && cmake --build . -j$(nproc) --target ggml-vulkan

# correctness test (static Vulkan build) — model on test-delta-net
./bin/test-dsa Vulkan0

# end-to-end smoke tests (once ops land)
./bin/llama-cli -m <GLM-5.2.gguf>  -dev Vulkan0 -ngl 99 -cmoe   -ub 2048 -n 32 -p "Hello"
./bin/llama-cli -m <DSV4.gguf>     -dev Vulkan0 -ngl 99 -ncmoe 37 -ub 2048 -n 32 -p "Hello"

# byte-identical check against single-GPU CPU/reference output
```
