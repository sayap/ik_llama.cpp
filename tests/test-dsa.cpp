// Correctness test for the DSA / GLM-DSA / DeepSeek-V4 sparse-attention Vulkan ops:
//   GGML_OP_SINKHORN, GGML_OP_HC_PRE, GGML_OP_HC_POST, GGML_OP_DS4_COMP,
//   GGML_OP_MASK_TOPK, GGML_OP_MASK_TO_IDX, GGML_OP_INDEXER_TOPK, GGML_OP_LATENT_ATTN.
//
// Usage: test-dsa <backend>   (e.g. "Vulkan0", "CUDA0", "CPU")
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void init_tensor_uniform(ggml_tensor * tensor, float min = -0.5f, float max = 0.5f) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_real_distribution<float> dist(min, max);
    const size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) data[i] = dist(rng);

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16(size);
        ggml_fp32_to_fp16_row(data.data(), f16.data(), size);
        ggml_backend_tensor_set(tensor, f16.data(), 0, size * sizeof(ggml_fp16_t));
    } else if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> idata(size);
        for (size_t i = 0; i < size; i++) idata[i] = (int32_t)data[i];
        ggml_backend_tensor_set(tensor, idata.data(), 0, size * sizeof(int32_t));
    } else {
        fprintf(stderr, "init_tensor_uniform: unsupported type %s\n", ggml_type_name(tensor->type));
        exit(1);
    }
}

static void init_tensor_q8_0(ggml_tensor * tensor) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) data[i] = dist(rng);

    const size_t nrows = size / tensor->ne[0];
    std::vector<uint8_t> dataq(nrows * ggml_row_size(tensor->type, tensor->ne[0]));
    std::vector<float> imatrix(tensor->ne[0], 1.0f);
    struct quantize_user_data qdata = { false, false };
    ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, nrows, tensor->ne[0], imatrix.data(), &qdata);
    ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
}

// generic quantized-tensor init (any type ggml_quantize_chunk supports)
static void init_tensor_quant(ggml_tensor * tensor) {
    init_tensor_q8_0(tensor);
}

static double max_abs_diff(const float * a, const float * b, size_t n) {
    double max = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (std::isinf(a[i]) && std::isinf(b[i]) && std::signbit(a[i]) == std::signbit(b[i])) {
            continue;
        }
        max = std::max(max, (double)std::abs(a[i] - b[i]));
    }
    return max;
}

static int64_t max_abs_diff_i32(const int32_t * a, const int32_t * b, size_t n) {
    int64_t max = 0;
    for (size_t i = 0; i < n; i++) {
        max = std::max(max, (int64_t)std::llabs((int64_t)a[i] - (int64_t)b[i]));
    }
    return max;
}

static int n_failures = 0;

static void copy_tensors_by_name(ggml_context * ctx_cpu, ggml_context * ctx_tgt) {
    for (ggml_tensor * t_c = ggml_get_first_tensor(ctx_cpu); t_c != NULL; t_c = ggml_get_next_tensor(ctx_cpu, t_c)) {
        if (t_c->data == nullptr || t_c->name[0] == '\0') continue;
        if (t_c->view_src != nullptr || t_c->op == GGML_OP_VIEW) continue;
        ggml_tensor * t_t = ggml_get_tensor(ctx_tgt, t_c->name);
        if (t_t != nullptr) {
            ggml_backend_tensor_copy(t_c, t_t);
        }
    }
}

static void check_float(const char * name, ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_context * ctx_cpu, ggml_context * ctx_tgt, ggml_tensor * out_cpu, ggml_tensor * out_tgt,
        double max_err) {
    ggml_cgraph * gf_cpu = ggml_new_graph(ctx_cpu);
    ggml_build_forward_expand(gf_cpu, out_cpu);
    ggml_cplan plan = ggml_graph_plan(gf_cpu, 4);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }
    ggml_graph_compute(gf_cpu, &plan);
    free(plan.work_data);

    ggml_cgraph * gf_tgt = ggml_new_graph(ctx_tgt);
    ggml_build_forward_expand(gf_tgt, out_tgt);
    fprintf(stderr, "[check] computing %s with backend %s\n", name, ggml_backend_name(backend_tgt));
    fflush(stderr);
    if (ggml_backend_graph_compute(backend_tgt, gf_tgt) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL %s: backend compute failed\n", name);
        n_failures++;
        return;
    }

    const size_t nelements = ggml_nelements(out_cpu);
    std::vector<float> a(nelements), b(nelements);
    if (out_cpu->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> fa(nelements), fb(nelements);
        ggml_backend_tensor_get(out_cpu, fa.data(), 0, nelements * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(out_tgt, fb.data(), 0, nelements * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < nelements; i++) {
            a[i] = ggml_fp16_to_fp32(fa[i]);
            b[i] = ggml_fp16_to_fp32(fb[i]);
        }
    } else {
        ggml_backend_tensor_get(out_cpu, a.data(), 0, nelements * sizeof(float));
        ggml_backend_tensor_get(out_tgt, b.data(), 0, nelements * sizeof(float));
    }

    double err = max_abs_diff(a.data(), b.data(), nelements);
    if (err > max_err) {
        fprintf(stderr, "FAIL %s: max abs diff = %g > %g\n", name, err, max_err);
        for (size_t i = 0; i < nelements && i < 32; i++) {
            fprintf(stderr, "  [%zu] cpu=%g tgt=%g\n", i, a[i], b[i]);
        }
        n_failures++;
    } else {
        printf("OK   %s (max abs diff = %g)\n", name, err);
    }
}

static void check_i32(const char * name, ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_context * ctx_cpu, ggml_context * ctx_tgt, ggml_tensor * out_cpu, ggml_tensor * out_tgt) {
    ggml_cgraph * gf_cpu = ggml_new_graph(ctx_cpu);
    ggml_build_forward_expand(gf_cpu, out_cpu);
    ggml_cplan plan = ggml_graph_plan(gf_cpu, 4);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }
    ggml_graph_compute(gf_cpu, &plan);
    free(plan.work_data);

    ggml_cgraph * gf_tgt = ggml_new_graph(ctx_tgt);
    ggml_build_forward_expand(gf_tgt, out_tgt);
    fprintf(stderr, "[check] computing %s with backend %s\n", name, ggml_backend_name(backend_tgt));
    fflush(stderr);
    if (ggml_backend_graph_compute(backend_tgt, gf_tgt) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL %s: backend compute failed\n", name);
        n_failures++;
        return;
    }

    const size_t nelements = ggml_nelements(out_cpu);
    std::vector<int32_t> a(nelements), b(nelements);
    ggml_backend_tensor_get(out_cpu, a.data(), 0, nelements * sizeof(int32_t));
    ggml_backend_tensor_get(out_tgt, b.data(), 0, nelements * sizeof(int32_t));

    int64_t err = max_abs_diff_i32(a.data(), b.data(), nelements);
    if (err != 0) {
        fprintf(stderr, "FAIL %s: max abs diff = %ld\n", name, (long)err);
        for (size_t i = 0; i < nelements && i < 32; i++) {
            fprintf(stderr, "  [%zu] cpu=%d tgt=%d\n", i, a[i], b[i]);
        }
        n_failures++;
    } else {
        printf("OK   %s (exact)\n", name);
    }
}

// Compare a quantized output tensor between the CPU reference and the target
// backend. The GPU quantizer may differ from the CPU one in the last ULP (f16
// scale rounding, FMA contraction), so a bit-exact memcmp is tried first and a
// dequantized comparison with tolerance is used as the fallback.
static void check_quant(const char * name, ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_context * ctx_cpu, ggml_context * ctx_tgt, ggml_tensor * out_cpu, ggml_tensor * out_tgt,
        double max_err) {
    ggml_cgraph * gf_cpu = ggml_new_graph(ctx_cpu);
    ggml_build_forward_expand(gf_cpu, out_cpu);
    ggml_cplan plan = ggml_graph_plan(gf_cpu, 4);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }
    ggml_graph_compute(gf_cpu, &plan);
    free(plan.work_data);

    ggml_cgraph * gf_tgt = ggml_new_graph(ctx_tgt);
    ggml_build_forward_expand(gf_tgt, out_tgt);
    fprintf(stderr, "[check] computing %s with backend %s\n", name, ggml_backend_name(backend_tgt));
    fflush(stderr);
    if (ggml_backend_graph_compute(backend_tgt, gf_tgt) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL %s: backend compute failed\n", name);
        n_failures++;
        return;
    }

    const size_t nbytes = ggml_nbytes(out_cpu);
    std::vector<uint8_t> a(nbytes), b(nbytes);
    ggml_backend_tensor_get(out_cpu, a.data(), 0, nbytes);
    ggml_backend_tensor_get(out_tgt, b.data(), 0, nbytes);

    if (memcmp(a.data(), b.data(), nbytes) == 0) {
        printf("OK   %s (bytes exact)\n", name);
        return;
    }

    // fall back to comparing the dequantized values
    const int64_t ne0 = out_cpu->ne[0];
    const size_t nrows = nbytes / ggml_row_size(out_cpu->type, ne0);
    std::vector<float> fa(ggml_nelements(out_cpu)), fb(ggml_nelements(out_cpu));
    const ggml_type_traits_t & tq = ggml_internal_get_type_traits(out_cpu->type);
    for (size_t r = 0; r < nrows; r++) {
        tq.to_float(a.data() + r * ggml_row_size(out_cpu->type, ne0), fa.data() + r * ne0, ne0);
        tq.to_float(b.data() + r * ggml_row_size(out_cpu->type, ne0), fb.data() + r * ne0, ne0);
    }
    double err = max_abs_diff(fa.data(), fb.data(), fa.size());
    if (err > max_err) {
        fprintf(stderr, "FAIL %s: dequantized max abs diff = %g > %g\n", name, err, max_err);
        for (size_t i = 0; i < fa.size() && i < 32; i++) {
            fprintf(stderr, "  [%zu] cpu=%g tgt=%g\n", i, fa[i], fb[i]);
        }
        n_failures++;
    } else {
        printf("OK   %s (dequantized max abs diff = %g)\n", name, err);
    }
}

// The KV-cache quantized write: ggml_cpy of an F32 [D, T*H] tensor into a
// strided 2d view of a larger quantized cache (mirrors llm_build_kv_store's
// k_cache_view), plus the dequantizing read-back (cpy quant -> F32).
static void test_cpy_kv_write(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, ggml_type cache_type,
        int D, int n_head_kv, int n_cache_rows, int kv_head, int n_tokens) {
    char name[256];
    snprintf(name, sizeof(name), "cpy_kv_write cache=%s D=%d hkv=%d rows=%d head=%d T=%d",
            ggml_type_name(cache_type), D, n_head_kv, n_cache_rows, kv_head, n_tokens);

    // two graphs per ctx are created (write + readback checks)
    ggml_init_params params = { ggml_tensor_overhead()*64 + 4*ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    const size_t row_size = ggml_row_size(cache_type, D);

    auto build = [&](ggml_context * ctx) {
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, cache_type, D, n_cache_rows);
        ggml_set_name(cache, "cache");
        ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tokens*n_head_kv);
        ggml_set_name(src, "src");

        ggml_tensor * view = ggml_view_2d(ctx, cache, D, n_tokens*n_head_kv,
                row_size, row_size*n_head_kv*kv_head);
        ggml_tensor * cpy = ggml_cpy(ctx, src, view);
        ggml_set_name(cpy, "cpy");

        // dequantizing read-back of the written region (cpy quant -> F32)
        ggml_tensor * rb = ggml_cpy(ctx, view, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tokens*n_head_kv));
        ggml_set_name(rb, "rb");
        return cpy;
    };

    ggml_tensor * cpy_c = build(ctx_cpu);
    ggml_tensor * cpy_t = build(ctx_tgt);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    // zero the caches so untouched rows compare equal
    {
        const size_t nbytes = ggml_nbytes(ggml_get_tensor(ctx_cpu, "cache"));
        std::vector<uint8_t> z(nbytes, 0);
        ggml_backend_tensor_set(ggml_get_tensor(ctx_cpu, "cache"), z.data(), 0, nbytes);
        ggml_backend_tensor_set(ggml_get_tensor(ctx_tgt, "cache"), z.data(), 0, nbytes);
    }

    init_tensor_uniform(ggml_get_tensor(ctx_cpu, "src"), -2.0f, 2.0f);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    // the quantized write (checks the cache bytes through the cpy view)
    check_quant(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, cpy_c, cpy_t, 0.2);

    // the read-back runs after both caches have been written
    {
        char rbname[300];
        snprintf(rbname, sizeof(rbname), "%s readback", name);
        check_float(rbname, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt,
                ggml_get_tensor(ctx_cpu, "rb"), ggml_get_tensor(ctx_tgt, "rb"), 0.2);
    }

    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

// INDEXER_TOPK only defines the *set* of selected rows (the CPU bucket top-k
// deliberately returns most buckets in original order); compare sorted rows.
static void check_i32_set(const char * name, ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_context * ctx_cpu, ggml_context * ctx_tgt, ggml_tensor * out_cpu, ggml_tensor * out_tgt) {
    ggml_cgraph * gf_cpu = ggml_new_graph(ctx_cpu);
    ggml_build_forward_expand(gf_cpu, out_cpu);
    ggml_cplan plan = ggml_graph_plan(gf_cpu, 4);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }
    ggml_graph_compute(gf_cpu, &plan);
    free(plan.work_data);

    ggml_cgraph * gf_tgt = ggml_new_graph(ctx_tgt);
    ggml_build_forward_expand(gf_tgt, out_tgt);
    fprintf(stderr, "[check] computing %s with backend %s\n", name, ggml_backend_name(backend_tgt));
    fflush(stderr);
    if (ggml_backend_graph_compute(backend_tgt, gf_tgt) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL %s: backend compute failed\n", name);
        n_failures++;
        return;
    }

    const int64_t ne0 = out_cpu->ne[0];
    const size_t nelements = ggml_nelements(out_cpu);
    std::vector<int32_t> a(nelements), b(nelements);
    ggml_backend_tensor_get(out_cpu, a.data(), 0, nelements * sizeof(int32_t));
    ggml_backend_tensor_get(out_tgt, b.data(), 0, nelements * sizeof(int32_t));

    int64_t err = 0;
    for (size_t r = 0; r + ne0 <= nelements; r += ne0) {
        std::vector<int32_t> ar(a.begin() + r, a.begin() + r + ne0);
        std::vector<int32_t> br(b.begin() + r, b.begin() + r + ne0);
        std::sort(ar.begin(), ar.end());
        std::sort(br.begin(), br.end());
        for (int64_t i = 0; i < ne0; i++) {
            err = std::max(err, (int64_t)std::llabs((int64_t)ar[i] - (int64_t)br[i]));
        }
    }
    if (err != 0) {
        fprintf(stderr, "FAIL %s: max abs diff = %ld\n", name, (long)err);
        for (size_t i = 0; i < nelements && i < 32; i++) {
            fprintf(stderr, "  [%zu] cpu=%d tgt=%d\n", i, a[i], b[i]);
        }
        n_failures++;
    } else {
        printf("OK   %s (set exact)\n", name);
    }
}

static void test_sinkhorn(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, int S, int T, bool transposed) {
    char name[256];
    snprintf(name, sizeof(name), "sinkhorn S=%d T=%d transposed=%d", S, T, transposed ? 1 : 0);
    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * a_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, S * S, T);
    ggml_set_name(a_c, "a");
    ggml_tensor * out_c = ggml_sinkhorn(ctx_cpu, a_c, S, 4, 1e-6f, transposed);

    ggml_tensor * a_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, S * S, T);
    ggml_set_name(a_t, "a");
    ggml_tensor * out_t = ggml_sinkhorn(ctx_tgt, a_t, S, 4, 1e-6f, transposed);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    init_tensor_uniform(a_c, -1.0f, 1.0f);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 1e-4);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_hc_pre(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, int S, int T, int iters = 4) {
    char name[256];
    snprintf(name, sizeof(name), "hc_pre S=%d T=%d", S, T);
    const int ntot = S * S + 2 * S;
    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * x_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, ntot, T);
    ggml_set_name(x_c, "x");
    ggml_tensor * scale_c = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, 3);
    ggml_set_name(scale_c, "scale");
    ggml_tensor * bias_c = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, ntot);
    ggml_set_name(bias_c, "bias");
    ggml_tensor * out_c = ggml_hc_pre(ctx_cpu, x_c, scale_c, bias_c, S, iters, 1e-6f);

    ggml_tensor * x_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, ntot, T);
    ggml_set_name(x_t, "x");
    ggml_tensor * scale_t = ggml_new_tensor_1d(ctx_tgt, GGML_TYPE_F32, 3);
    ggml_set_name(scale_t, "scale");
    ggml_tensor * bias_t = ggml_new_tensor_1d(ctx_tgt, GGML_TYPE_F32, ntot);
    ggml_set_name(bias_t, "bias");
    ggml_tensor * out_t = ggml_hc_pre(ctx_tgt, x_t, scale_t, bias_t, S, iters, 1e-6f);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    init_tensor_uniform(x_c, -1.0f, 1.0f);
    init_tensor_uniform(scale_c, -0.5f, 0.5f);
    init_tensor_uniform(bias_c, -0.5f, 0.5f);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 1e-4);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_hc_post(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, int ne0, int S, int T) {
    char name[256];
    snprintf(name, sizeof(name), "hc_post ne0=%d S=%d T=%d", ne0, S, T);
    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * x_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, ne0, T);
    ggml_set_name(x_c, "x");
    ggml_tensor * post_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, S, T);
    ggml_set_name(post_c, "post");
    ggml_tensor * res_c = ggml_new_tensor_3d(ctx_cpu, GGML_TYPE_F32, ne0, S, T);
    ggml_set_name(res_c, "res");
    ggml_tensor * comb_c = ggml_new_tensor_3d(ctx_cpu, GGML_TYPE_F32, S, S, T);
    ggml_set_name(comb_c, "comb");
    ggml_tensor * out_c = ggml_hc_post(ctx_cpu, x_c, post_c, res_c, comb_c);

    ggml_tensor * x_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, ne0, T);
    ggml_set_name(x_t, "x");
    ggml_tensor * post_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, S, T);
    ggml_set_name(post_t, "post");
    ggml_tensor * res_t = ggml_new_tensor_3d(ctx_tgt, GGML_TYPE_F32, ne0, S, T);
    ggml_set_name(res_t, "res");
    ggml_tensor * comb_t = ggml_new_tensor_3d(ctx_tgt, GGML_TYPE_F32, S, S, T);
    ggml_set_name(comb_t, "comb");
    ggml_tensor * out_t = ggml_hc_post(ctx_tgt, x_t, post_t, res_t, comb_t);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    init_tensor_uniform(x_c, -1.0f, 1.0f);
    init_tensor_uniform(post_c, -1.0f, 1.0f);
    init_tensor_uniform(res_c, -1.0f, 1.0f);
    init_tensor_uniform(comb_c, -1.0f, 1.0f);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 1e-4);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_ds4_comp(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, int type, int ne0_state, int nblock, int ratio) {
    char name[256];
    snprintf(name, sizeof(name), "ds4_comp type=%d ne0_state=%d nblock=%d ratio=%d", type, ne0_state, nblock, ratio);

    const int dst_ne0 = type == 0 ? ne0_state / 2 : ne0_state;
    const int idx_len  = type == 0 ? 2 * ratio * nblock : ratio * nblock;
    const int nrows_state = type == 0 ? 2 * ratio * nblock : ratio * nblock;

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * state_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, ne0_state, nrows_state);
    ggml_set_name(state_c, "state");
    ggml_tensor * score_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, ne0_state, nrows_state);
    ggml_set_name(score_c, "score");
    ggml_tensor * idx_c = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_I32, idx_len);
    ggml_set_name(idx_c, "idx");
    ggml_tensor * out_c = ggml_ds4_comp(ctx_cpu, state_c, score_c, idx_c, ratio, type);

    ggml_tensor * state_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, ne0_state, nrows_state);
    ggml_set_name(state_t, "state");
    ggml_tensor * score_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, ne0_state, nrows_state);
    ggml_set_name(score_t, "score");
    ggml_tensor * idx_t = ggml_new_tensor_1d(ctx_tgt, GGML_TYPE_I32, idx_len);
    ggml_set_name(idx_t, "idx");
    ggml_tensor * out_t = ggml_ds4_comp(ctx_tgt, state_t, score_t, idx_t, ratio, type);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    init_tensor_uniform(state_c, -1.0f, 1.0f);
    init_tensor_uniform(score_c, -1.0f, 1.0f);

    std::vector<int32_t> idx_data(idx_len);
    std::mt19937 rng(12345);
    for (int i = 0; i < idx_len; i++) {
        idx_data[i] = (int32_t)(rng() % nrows_state);
    }
    ggml_backend_tensor_set(idx_c, idx_data.data(), 0, idx_len * sizeof(int32_t));

    copy_tensors_by_name(ctx_cpu, ctx_tgt);
    ggml_backend_tensor_set(idx_t, idx_data.data(), 0, idx_len * sizeof(int32_t));

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 1e-4);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_mask_to_idx(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, ggml_type type, int ne00, int nrows, int max_row_size) {
    char name[256];
    snprintf(name, sizeof(name), "mask_to_idx type=%s ne00=%d nrows=%d max=%d", ggml_type_name(type), ne00, nrows, max_row_size);
    const int ne0 = std::min(ne00, max_row_size);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * m_c = ggml_new_tensor_2d(ctx_cpu, type, ne00, nrows);
    ggml_set_name(m_c, "m");
    ggml_tensor * out_c = ggml_mask_to_index(ctx_cpu, m_c, max_row_size);

    ggml_tensor * m_t = ggml_new_tensor_2d(ctx_tgt, type, ne00, nrows);
    ggml_set_name(m_t, "m");
    ggml_tensor * out_t = ggml_mask_to_index(ctx_tgt, m_t, max_row_size);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    // Fill with -inf, then zero out `ne0` positions per row (sorted, unique).
    const size_t size = ggml_nelements(m_c);
    std::vector<float> data(size, -INFINITY);
    std::mt19937 rng(777);
    for (int r = 0; r < nrows; r++) {
        std::vector<int> positions(ne00);
        for (int i = 0; i < ne00; i++) positions[i] = i;
        std::shuffle(positions.begin(), positions.end(), rng);
        for (int k = 0; k < ne0; k++) {
            data[r * ne00 + positions[k]] = 0.0f;
        }
    }
    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(m_c, data.data(), 0, size * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> f16(size);
        ggml_fp32_to_fp16_row(data.data(), f16.data(), size);
        ggml_backend_tensor_set(m_c, f16.data(), 0, size * sizeof(ggml_fp16_t));
    }
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check_i32(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_mask_topk(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, ggml_type type, int ne0, int ne1, int ntopk) {
    char name[256];
    snprintf(name, sizeof(name), "mask_topk type=%s ne0=%d ne1=%d ntopk=%d", ggml_type_name(type), ne0, ne1, ntopk);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * m_c = ggml_new_tensor_2d(ctx_cpu, type, ne0, ne1);
    ggml_set_name(m_c, "m");
    ggml_tensor * topk_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_I32, ntopk, ne1);
    ggml_set_name(topk_c, "topk");
    ggml_tensor * out_c = ggml_indexer_mask(ctx_cpu, m_c, topk_c);

    ggml_tensor * m_t = ggml_new_tensor_2d(ctx_tgt, type, ne0, ne1);
    ggml_set_name(m_t, "m");
    ggml_tensor * topk_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_I32, ntopk, ne1);
    ggml_set_name(topk_t, "topk");
    ggml_tensor * out_t = ggml_indexer_mask(ctx_tgt, m_t, topk_t);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    init_tensor_uniform(m_c, -2.0f, 2.0f);

    std::vector<int32_t> idx_data(ntopk * ne1);
    std::mt19937 rng(999);
    for (int i = 0; i < ntopk * ne1; i++) {
        idx_data[i] = (int32_t)(rng() % ne0);
    }
    ggml_backend_tensor_set(topk_c, idx_data.data(), 0, idx_data.size() * sizeof(int32_t));

    copy_tensors_by_name(ctx_cpu, ctx_tgt);
    ggml_backend_tensor_set(topk_t, idx_data.data(), 0, idx_data.size() * sizeof(int32_t));

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 0.0);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_indexer_topk(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, ggml_type ktype, int Dk, int n_kv, int n_head, int n_rows, int n_top_k) {
    char name[256];
    snprintf(name, sizeof(name), "indexer_topk k=%s Dk=%d n_kv=%d n_head=%d n_rows=%d topk=%d",
            ggml_type_name(ktype), Dk, n_kv, n_head, n_rows, n_top_k);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * k_c = ggml_new_tensor_2d(ctx_cpu, ktype, Dk, n_kv);
    ggml_set_name(k_c, "k");
    ggml_tensor * q_c = ggml_new_tensor_3d(ctx_cpu, GGML_TYPE_F32, Dk, n_head, n_rows);
    ggml_set_name(q_c, "q");
    ggml_tensor * w_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, n_head, n_rows);
    ggml_set_name(w_c, "w");
    ggml_tensor * m_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, n_kv, n_rows);
    ggml_set_name(m_c, "m");
    ggml_tensor * out_c = ggml_indexer_topk(ctx_cpu, k_c, q_c, w_c, m_c, GGML_UNARY_OP_RELU, n_top_k);

    ggml_tensor * k_t = ggml_new_tensor_2d(ctx_tgt, ktype, Dk, n_kv);
    ggml_set_name(k_t, "k");
    ggml_tensor * q_t = ggml_new_tensor_3d(ctx_tgt, GGML_TYPE_F32, Dk, n_head, n_rows);
    ggml_set_name(q_t, "q");
    ggml_tensor * w_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, n_head, n_rows);
    ggml_set_name(w_t, "w");
    ggml_tensor * m_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, n_kv, n_rows);
    ggml_set_name(m_t, "m");
    ggml_tensor * out_t = ggml_indexer_topk(ctx_tgt, k_t, q_t, w_t, m_t, GGML_UNARY_OP_RELU, n_top_k);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    init_tensor_uniform(k_c, -1.0f, 1.0f);
    init_tensor_uniform(q_c, -1.0f, 1.0f);
    init_tensor_uniform(w_c, -1.0f, 1.0f);
    init_tensor_uniform(m_c, -2.0f, 2.0f);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check_i32_set(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_set_rows(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, ggml_type dst_type, int ncols, int nrows_cache, int nrows_src, bool i64) {
    char name[256];
    snprintf(name, sizeof(name), "set_rows dst=%s ncols=%d cache=%d src=%d i64=%d",
            ggml_type_name(dst_type), ncols, nrows_cache, nrows_src, i64 ? 1 : 0);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * a_c = ggml_new_tensor_2d(ctx_cpu, dst_type, ncols, nrows_cache);
    ggml_set_name(a_c, "a");
    ggml_tensor * b_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, ncols, nrows_src);
    ggml_set_name(b_c, "b");
    ggml_tensor * c_c = ggml_new_tensor_1d(ctx_cpu, i64 ? GGML_TYPE_I64 : GGML_TYPE_I32, nrows_src);
    ggml_set_name(c_c, "c");
    ggml_tensor * out_c = ggml_set_rows(ctx_cpu, a_c, b_c, c_c);

    ggml_tensor * a_t = ggml_new_tensor_2d(ctx_tgt, dst_type, ncols, nrows_cache);
    ggml_set_name(a_t, "a");
    ggml_tensor * b_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, ncols, nrows_src);
    ggml_set_name(b_t, "b");
    ggml_tensor * c_t = ggml_new_tensor_1d(ctx_tgt, i64 ? GGML_TYPE_I64 : GGML_TYPE_I32, nrows_src);
    ggml_set_name(c_t, "c");
    ggml_tensor * out_t = ggml_set_rows(ctx_tgt, a_t, b_t, c_t);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    init_tensor_uniform(b_c, -2.0f, 2.0f);
    if (ggml_is_quantized(dst_type)) {
        init_tensor_quant(a_c);
    } else {
        init_tensor_uniform(a_c, -2.0f, 2.0f);
    }

    std::vector<int32_t> i32(nrows_src);
    std::vector<int64_t> i64_data(nrows_src);
    std::mt19937 rng(31337);
    for (int i = 0; i < nrows_src; i++) {
        i32[i] = (int32_t)(rng() % nrows_cache);
        i64_data[i] = i32[i];
    }
    if (i64) {
        ggml_backend_tensor_set(c_c, i64_data.data(), 0, nrows_src * sizeof(int64_t));
    } else {
        ggml_backend_tensor_set(c_c, i32.data(), 0, nrows_src * sizeof(int32_t));
    }

    copy_tensors_by_name(ctx_cpu, ctx_tgt);
    if (i64) {
        ggml_backend_tensor_set(c_t, i64_data.data(), 0, nrows_src * sizeof(int64_t));
    } else {
        ggml_backend_tensor_set(c_t, i32.data(), 0, nrows_src * sizeof(int32_t));
    }

    if (ggml_is_quantized(dst_type)) {
        // GPU quantizers may differ in the last ULP (f16 scale rounding, FMA
        // contraction); compare the quantized caches with a dequantized fallback.
        check_quant(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 0.2);
    } else {
        check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 0.0);
    }
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_latent_attn(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, ggml_type ctype, int Dk, int T, int H, int N, int P, int topk, bool indexed) {
    char name[256];
    snprintf(name, sizeof(name), "latent_attn cache=%s indexed=%d Dk=%d T=%d H=%d N=%d P=%d topk=%d",
            ggml_type_name(ctype), indexed ? 1 : 0, Dk, T, H, N, P, topk);
    const int dv = 32;
    const int dv_off = 0;

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    auto build = [&](ggml_context * ctx, ggml_tensor ** ind_out) {
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dk, T, H);
        ggml_set_name(q, "q");
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, ctype, Dk, N);
        ggml_set_name(cache, "cache");
        ggml_tensor * pk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Dk, P);
        ggml_set_name(pk, "pk");
        ggml_tensor * pv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P, dv);
        ggml_set_name(pv, "pv");
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, T);
        ggml_set_name(mask, "mask");
        ggml_tensor * indices = nullptr;
        if (indexed) {
            indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, T);
            ggml_set_name(indices, "indices");
        }
        ggml_tensor * out;
        if (indexed) {
            out = ggml_latent_attn_indexed_ext(ctx, q, cache, pk, pv, mask, indices, dv, dv_off, 0.5f, 0.0f);
        } else {
            out = ggml_latent_attn_prefix_ext(ctx, q, cache, pk, pv, mask, dv, dv_off, 0.5f, 0.0f);
        }
        *ind_out = indices;
        return out;
    };

    ggml_tensor * indices_c = nullptr;
    ggml_tensor * indices_t = nullptr;
    ggml_tensor * out_c = build(ctx_cpu, &indices_c);
    ggml_tensor * out_t = build(ctx_tgt, &indices_t);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_cpu); t != NULL; t = ggml_get_next_tensor(ctx_cpu, t)) {
        if (t->data == nullptr || t->view_src != nullptr || t->op == GGML_OP_VIEW) continue;
        if (strcmp(t->name, "cache") == 0 && ctype == GGML_TYPE_Q8_0) {
            init_tensor_q8_0(t);
        } else if (strcmp(t->name, "indices") == 0) {
            continue;
        } else if (t->type == GGML_TYPE_I32) {
            continue;
        } else {
            init_tensor_uniform(t, -1.0f, 1.0f);
        }
    }

    if (indexed) {
        std::vector<int32_t> idx_data(topk * T);
        std::mt19937 rng(4242);
        for (int i = 0; i < topk * T; i++) {
            idx_data[i] = (int32_t)(rng() % N);
        }
        ggml_backend_tensor_set(indices_c, idx_data.data(), 0, idx_data.size() * sizeof(int32_t));
    }

    copy_tensors_by_name(ctx_cpu, ctx_tgt);
    if (indexed) {
        std::vector<int32_t> idx_data(topk * T);
        ggml_backend_tensor_get(indices_c, idx_data.data(), 0, idx_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(indices_t, idx_data.data(), 0, idx_data.size() * sizeof(int32_t));
    }

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 5e-3);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

// Indexed flash attention (GGML_OP_FLASH_ATTN_EXT with src[5] == indices and an
// optional per-head sink in src[4]), the DSV4 sparse-attention path. The CPU
// reference routes through iqk_flash_attn_noalibi, which requires F32 q, F16 mask,
// max_bias == 0, single KV head/batch, top_k < n_kv, and a top_k whose "last valid
// + 1" rounds up to a multiple of 32 (so keep topk a multiple of 32).
static void test_flash_attn_indexed(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_type kvtype, int Dk, int Dv, int T, int H, int KV, int topk, bool with_sinks, bool with_padding) {
    char name[256];
    snprintf(name, sizeof(name), "flash_attn_indexed kv=%s Dk=%d Dv=%d T=%d H=%d KV=%d topk=%d sinks=%d pad=%d",
            ggml_type_name(kvtype), Dk, Dv, T, H, KV, topk, with_sinks ? 1 : 0, with_padding ? 1 : 0);

    GGML_ASSERT(topk % 32 == 0);
    GGML_ASSERT(topk < KV);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    const int64_t mask_T = GGML_PAD(T, GGML_KQ_MASK_PAD);

    auto build = [&](ggml_context * ctx) {
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dk, T, H);
        ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_2d(ctx, kvtype, Dk, KV);
        ggml_set_name(k, "k");
        ggml_tensor * v = ggml_new_tensor_2d(ctx, kvtype, Dv, KV);
        ggml_set_name(v, "v");
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, KV, mask_T);
        ggml_set_name(mask, "mask");
        ggml_tensor * sinks = nullptr;
        if (with_sinks) {
            sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            ggml_set_name(sinks, "sinks");
        }
        ggml_tensor * indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, T);
        ggml_set_name(indices, "indices");

        const float scale = 1.0f / sqrtf((float) Dk);
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (with_sinks) {
            ggml_flash_attn_ext_add_sinks(out, sinks);
        }
        out->src[5] = indices;
        return out;
    };

    ggml_tensor * out_c = build(ctx_cpu);
    ggml_tensor * out_t = build(ctx_tgt);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_cpu); t != NULL; t = ggml_get_next_tensor(ctx_cpu, t)) {
        if (t->data == nullptr || t->view_src != nullptr || t->op == GGML_OP_VIEW) continue;
        if (strcmp(t->name, "indices") == 0 || t->type == GGML_TYPE_I32) continue;
        if ((strcmp(t->name, "k") == 0 || strcmp(t->name, "v") == 0) && kvtype == GGML_TYPE_Q8_0) {
            init_tensor_q8_0(t);
        } else if (strcmp(t->name, "mask") == 0) {
            std::vector<ggml_fp16_t> m(KV * mask_T, ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(t, m.data(), 0, m.size() * sizeof(ggml_fp16_t));
        } else {
            init_tensor_uniform(t, -1.0f, 1.0f);
        }
    }

    std::vector<int32_t> idx_data(topk * T);
    std::mt19937 rng(12345);
    for (int i = 0; i < topk * T; i++) {
        idx_data[i] = (int32_t)(rng() % KV);
    }
    if (with_padding) {
        // trailing -1 per row; last_found+1 must still round to a multiple of 32
        GGML_ASSERT(topk >= 32);
        for (int t = 0; t < T; t++) {
            for (int j = topk - 16; j < topk; j++) {
                idx_data[t*topk + j] = -1;
            }
        }
    }
    ggml_backend_tensor_set(ggml_get_tensor(ctx_cpu, "indices"), idx_data.data(), 0, idx_data.size() * sizeof(int32_t));

    copy_tensors_by_name(ctx_cpu, ctx_tgt);
    {
        std::vector<int32_t> idx2(topk * T);
        ggml_backend_tensor_get(ggml_get_tensor(ctx_cpu, "indices"), idx2.data(), 0, idx2.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ggml_get_tensor(ctx_tgt, "indices"), idx2.data(), 0, idx2.size() * sizeof(int32_t));
    }

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 5e-3);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

// Dense FA with a quantized KV cache: q is forced onto an exact Q8 grid (each
// 32-element block has amax = 1.0 and values n/127), so the CPU reference's
// integer vec_dot (which quantizes q to Q8_2) matches the GPU's f32 dot of the
// format-dequantized K/V to f32 rounding. This validates the Vulkan kernels
// against the CPU integer kernels, which are exact for these legacy quants.
static void init_tensor_q_grid(ggml_tensor * tensor) {
    std::mt19937 rng(4242);
    std::uniform_int_distribution<int> dist(-127, 127);
    const size_t ne0 = tensor->ne[0];
    const size_t nrows = ggml_nelements(tensor) / ne0;
    GGML_ASSERT(ne0 % 32 == 0);
    std::vector<float> data(ggml_nelements(tensor));
    for (size_t r = 0; r < nrows; r++) {
        for (size_t i = 0; i < ne0; i++) {
            data[r*ne0 + i] = (float)dist(rng) / 127.0f;
        }
        // force every 32-element block to have amax = 1.0 (Q8_2 d = 1/127)
        for (size_t b = 0; b < ne0 / 32; b++) {
            data[r*ne0 + b*32] = 1.0f;
        }
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
}

// Dense flash attention with per-head sinks (src[4], no indexer), the DSV4
// full-attention layer path. The CPU reference uses the generic FA path
// (op_params[4] = IQK_DISABLED) which handles sinks.
static void test_flash_attn_dense_sinks(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_type kvtype, int Dk, int Dv, int T, int H, int KV, bool with_sinks) {
    char name[256];
    snprintf(name, sizeof(name), "flash_attn_dense_sinks kv=%s Dk=%d Dv=%d T=%d H=%d KV=%d sinks=%d",
            ggml_type_name(kvtype), Dk, Dv, T, H, KV, with_sinks ? 1 : 0);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    const int64_t mask_T = GGML_PAD(T, GGML_KQ_MASK_PAD);

    auto build = [&](ggml_context * ctx) {
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dk, T, H);
        ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_2d(ctx, kvtype, Dk, KV);
        ggml_set_name(k, "k");
        ggml_tensor * v = ggml_new_tensor_2d(ctx, kvtype, Dv, KV);
        ggml_set_name(v, "v");
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, KV, mask_T);
        ggml_set_name(mask, "mask");
        ggml_tensor * sinks = nullptr;
        if (with_sinks) {
            sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            ggml_set_name(sinks, "sinks");
        }

        const float scale = 1.0f / sqrtf((float) Dk);
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ((int32_t *) out->op_params)[4] = GGML_FLASH_ATTN_EXT_IQK_DISABLED;
        if (with_sinks) {
            ggml_flash_attn_ext_add_sinks(out, sinks);
        }
        return out;
    };

    ggml_tensor * out_c = build(ctx_cpu);
    ggml_tensor * out_t = build(ctx_tgt);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_cpu); t != NULL; t = ggml_get_next_tensor(ctx_cpu, t)) {
        if (t->data == nullptr || t->view_src != nullptr || t->op == GGML_OP_VIEW) continue;
        if ((strcmp(t->name, "k") == 0 || strcmp(t->name, "v") == 0) && kvtype != GGML_TYPE_F16) {
            init_tensor_quant(t);
        } else if (strcmp(t->name, "q") == 0 && ggml_is_quantized(kvtype)) {
            init_tensor_q_grid(t);
        } else if (strcmp(t->name, "mask") == 0) {
            std::vector<ggml_fp16_t> m(KV * mask_T, ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(t, m.data(), 0, m.size() * sizeof(ggml_fp16_t));
        } else {
            init_tensor_uniform(t, -1.0f, 1.0f);
        }
    }

    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 5e-3);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

int main(int argc, char ** argv) {
    const char * tgt_name = argc > 1 ? argv[1] : "CPU";

    ggml_backend_load_all();

    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    ggml_backend_t backend_tgt = nullptr;
    if (strcmp(tgt_name, "CPU") == 0) {
        backend_tgt = backend_cpu;
    } else {
        backend_tgt = ggml_backend_reg_init_backend_from_str(tgt_name);
    }
    if (!backend_tgt) {
        fprintf(stderr, "failed to init backend %s\n", tgt_name);
        return 1;
    }
    printf("target backend: %s\n", ggml_backend_name(backend_tgt));

    for (int S = 1; S <= 8; S++) {
        test_sinkhorn(backend_cpu, backend_tgt, S, 5, false);
        test_sinkhorn(backend_cpu, backend_tgt, S, 3, true);
    }

    for (int S = 1; S <= 8; S++) {
        test_hc_pre(backend_cpu, backend_tgt, S, 3);
    }

    test_hc_pre(backend_cpu, backend_tgt, 4, 1, 20);
    test_hc_pre(backend_cpu, backend_tgt, 4, 8, 20);
    test_hc_post(backend_cpu, backend_tgt, 4096, 4, 1);
    test_hc_post(backend_cpu, backend_tgt, 4096, 4, 8);

    test_hc_post(backend_cpu, backend_tgt, 16, 4, 3);
    test_hc_post(backend_cpu, backend_tgt, 16, 4, 1);
    test_hc_post(backend_cpu, backend_tgt, 32, 8, 2);

    test_ds4_comp(backend_cpu, backend_tgt, 0, 64, 2, 2);
    test_ds4_comp(backend_cpu, backend_tgt, 0, 64, 3, 4);
    test_ds4_comp(backend_cpu, backend_tgt, 1, 32, 3, 4);
    test_ds4_comp(backend_cpu, backend_tgt, 1, 64, 2, 5);
    test_ds4_comp(backend_cpu, backend_tgt, 0, 512, 2, 4);
    test_ds4_comp(backend_cpu, backend_tgt, 1, 512, 1, 128);

    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        test_mask_to_idx(backend_cpu, backend_tgt, t, 64, 3, 8);
        test_mask_to_idx(backend_cpu, backend_tgt, t, 128, 2, 16);
        test_mask_topk(backend_cpu, backend_tgt, t, 64, 3, 8);
        test_mask_topk(backend_cpu, backend_tgt, t, 64, 5, 16);
    }

    test_latent_attn(backend_cpu, backend_tgt, GGML_TYPE_F32, 64, 2, 2, 4, 2, 3, false);
    test_latent_attn(backend_cpu, backend_tgt, GGML_TYPE_F32, 64, 1, 3, 4, 2, 3, false);
    test_latent_attn(backend_cpu, backend_tgt, GGML_TYPE_F32, 64, 2, 2, 4, 2, 3, true);
    test_latent_attn(backend_cpu, backend_tgt, GGML_TYPE_F16, 64, 2, 2, 4, 2, 3, false);
    test_latent_attn(backend_cpu, backend_tgt, GGML_TYPE_F16, 64, 2, 2, 4, 2, 3, true);
    test_latent_attn(backend_cpu, backend_tgt, GGML_TYPE_Q8_0, 64, 2, 2, 4, 2, 3, false);
    test_latent_attn(backend_cpu, backend_tgt, GGML_TYPE_Q8_0, 64, 2, 2, 4, 2, 3, true);

    // Indexed flash attention + sinks (DSV4 sparse attention).
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_F16, 64, 64, 2, 3, 128, 32, false, false);
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_F16, 64, 64, 2, 3, 128, 32, true,  false);
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_F16, 64, 64, 1, 4, 128, 32, true,  false);
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_F16, 64, 64, 2, 3, 128, 64, true,  true);
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_F16, 128, 128, 1, 2, 256, 32, true, false);
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_Q8_0, 64, 64, 2, 3, 128, 32, true, false);
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_Q8_0, 64, 64, 1, 4, 128, 32, true, false);
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_F16, 512, 512, 1, 2, 1024, 32, true, false);

    // DSV4 real shapes: 512 head, 64 query heads, top-k 256/512 with sinks.
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_F16, 512, 512, 1, 64, 2048, 256, true, true);
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_F16, 512, 512, 1, 64, 2048, 512, true, true);
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_F16, 512, 512, 4, 64, 2048, 512, true, false);
    test_flash_attn_indexed(backend_cpu, backend_tgt, GGML_TYPE_F16, 512, 512, 4, 64, 2048, 256, false, true);

    // Dense FA + sinks (DSV4 full-attention layers).
    test_flash_attn_dense_sinks(backend_cpu, backend_tgt, GGML_TYPE_F16, 512, 512, 1, 64, 2048, true);
    test_flash_attn_dense_sinks(backend_cpu, backend_tgt, GGML_TYPE_F16, 512, 512, 5, 64, 2048, true);
    test_flash_attn_dense_sinks(backend_cpu, backend_tgt, GGML_TYPE_F16, 512, 512, 1, 64, 2048, false);
    test_flash_attn_dense_sinks(backend_cpu, backend_tgt, GGML_TYPE_F16, 64, 64, 2, 3, 128, true);

    // Dense FA with a quantized KV cache (Q6_0/Q8_0 decode + prompt paths).
    test_flash_attn_dense_sinks(backend_cpu, backend_tgt, GGML_TYPE_Q6_0, 64, 64, 1, 3, 128, false);
    test_flash_attn_dense_sinks(backend_cpu, backend_tgt, GGML_TYPE_Q6_0, 64, 64, 1, 8, 512, true);
    test_flash_attn_dense_sinks(backend_cpu, backend_tgt, GGML_TYPE_Q6_0, 128, 128, 5, 2, 256, false);
    test_flash_attn_dense_sinks(backend_cpu, backend_tgt, GGML_TYPE_Q6_0, 256, 256, 32, 4, 1024, false);
    test_flash_attn_dense_sinks(backend_cpu, backend_tgt, GGML_TYPE_Q8_0, 64, 64, 1, 3, 128, false);
    test_flash_attn_dense_sinks(backend_cpu, backend_tgt, GGML_TYPE_Q8_0, 128, 128, 5, 2, 256, false);

    // Quantized KV-cache write (llm_build_kv_store's ggml_cpy into a strided
    // cache view) and the dequantizing read-back.
    test_cpy_kv_write(backend_cpu, backend_tgt, GGML_TYPE_Q6_0, 128, 4, 256, 3, 7);
    test_cpy_kv_write(backend_cpu, backend_tgt, GGML_TYPE_Q6_0, 256, 1, 128, 1, 1);
    test_cpy_kv_write(backend_cpu, backend_tgt, GGML_TYPE_Q6_0, 64, 8, 512, 5, 32);
    test_cpy_kv_write(backend_cpu, backend_tgt, GGML_TYPE_Q8_0, 128, 4, 256, 3, 7);

    test_indexer_topk(backend_cpu, backend_tgt, GGML_TYPE_F32, 32, 128, 4, 4, 8);
    test_indexer_topk(backend_cpu, backend_tgt, GGML_TYPE_F32, 16, 64, 3, 2, 6);
    test_indexer_topk(backend_cpu, backend_tgt, GGML_TYPE_F16, 32, 128, 4, 4, 8);
    test_indexer_topk(backend_cpu, backend_tgt, GGML_TYPE_F16, 16, 64, 3, 2, 6);

    test_set_rows(backend_cpu, backend_tgt, GGML_TYPE_F16, 64, 128, 5, false);
    test_set_rows(backend_cpu, backend_tgt, GGML_TYPE_F16, 128, 256, 8, true);
    test_set_rows(backend_cpu, backend_tgt, GGML_TYPE_F32, 32, 64, 4, false);
    test_set_rows(backend_cpu, backend_tgt, GGML_TYPE_Q6_0, 128, 256, 8, false);
    test_set_rows(backend_cpu, backend_tgt, GGML_TYPE_Q6_0, 64, 128, 5, true);
    test_set_rows(backend_cpu, backend_tgt, GGML_TYPE_Q8_0, 128, 256, 8, false);
    test_set_rows(backend_cpu, backend_tgt, GGML_TYPE_Q8_0, 64, 128, 5, true);

    if (backend_tgt != backend_cpu) {
        ggml_backend_free(backend_tgt);
    }
    ggml_backend_free(backend_cpu);

    printf("%s: %d failures\n", tgt_name, n_failures);
    return n_failures == 0 ? 0 : 1;
}
