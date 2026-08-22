// Correctness test for the DSV4 "misc" ops that were CPU-fallbacks on Vulkan:
//   GGML_OP_MUL_MULTI_ADD (hyper-connection mixing / fused MoE weighted sum),
//   GGML_OP_HADAMARD (normalized Hadamard transform),
//   GGML_OP_FILL (constant fill),
//   GGML_UNARY_OP_SQRT_SOFTPLUS (DeepSeek-V4 MoE gate).
//
// Usage: test-dsv4-misc <backend>   (e.g. "Vulkan0", "CUDA0", "CPU")
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
        for (size_t i = 0; i < nelements && i < 16; i++) {
            fprintf(stderr, "  [%zu] cpu=%g tgt=%g\n", i, a[i], b[i]);
        }
        n_failures++;
    } else {
        printf("OK   %s (max abs diff = %g)\n", name, err);
    }
}

// dst[k, ir] = sum_j src0[k, j, ir] * src1[0, j, ir] (* scales[ids[j, ir]])
static void test_mul_multi_add(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        int64_t ne00, int64_t ne01, int64_t nrows, bool with_scales) {
    char name[256];
    snprintf(name, sizeof(name), "mul_multi_add %ldx%ldx%ld%s",
            (long)ne00, (long)ne01, (long)nrows, with_scales ? " +scales" : "");

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    auto build = [&](ggml_context * ctx) {
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne00, ne01, nrows);
        ggml_set_name(a, "a");
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, ne01, nrows);
        ggml_set_name(w, "w");
        ggml_tensor * out = ggml_mul_multi_add(ctx, a, w);
        if (with_scales) {
            const int64_t n_scales = 7;
            ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_scales);
            ggml_set_name(scales, "scales");
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ne01, nrows);
            ggml_set_name(ids, "ids");
            out->src[2] = scales;
            out->src[3] = ids;
        }
        return out;
    };
    ggml_tensor * out_c = build(ctx_cpu);
    ggml_tensor * out_t = build(ctx_tgt);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 1e-4);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_hadamard(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_type src_type, int64_t ne0, int64_t nrows, int nh) {
    char name[256];
    snprintf(name, sizeof(name), "hadamard %s ne0=%ld rows=%ld nh=%d",
            ggml_type_name(src_type), (long)ne0, (long)nrows, nh);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * a_c = ggml_new_tensor_2d(ctx_cpu, src_type, ne0, nrows);
    ggml_set_name(a_c, "a");
    ggml_tensor * out_c = ggml_hadamard(ctx_cpu, a_c, nh);

    ggml_tensor * a_t = ggml_new_tensor_2d(ctx_tgt, src_type, ne0, nrows);
    ggml_set_name(a_t, "a");
    ggml_tensor * out_t = ggml_hadamard(ctx_tgt, a_t, nh);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    // f16 sources dequantize differently on each backend; allow f16 rounding slack
    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t,
            src_type == GGML_TYPE_F16 ? 2e-2 : 1e-4);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_fill(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_type type, int64_t nelements, float value) {
    char name[256];
    snprintf(name, sizeof(name), "fill %s n=%ld value=%g", ggml_type_name(type), (long)nelements, value);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * a_c = ggml_new_tensor_1d(ctx_cpu, type, nelements);
    ggml_set_name(a_c, "a");
    ggml_tensor * out_c = ggml_fill(ctx_cpu, a_c, value);

    ggml_tensor * a_t = ggml_new_tensor_1d(ctx_tgt, type, nelements);
    ggml_set_name(a_t, "a");
    ggml_tensor * out_t = ggml_fill(ctx_tgt, a_t, value);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 1e-6);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_get_rows_dim0(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_type type, int64_t ne0, int64_t ne1, int64_t n_gather) {
    char name[256];
    snprintf(name, sizeof(name), "get_rows_dim0 %s ne0=%ld ne1=%ld gather=%ld",
            ggml_type_name(type), (long)ne0, (long)ne1, (long)n_gather);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    auto build = [&](ggml_context * ctx) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        ggml_set_name(a, "a");
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_gather, ne1);
        ggml_set_name(b, "b");
        return ggml_get_rows_ext(ctx, a, b, true, true);
    };
    ggml_tensor * out_c = build(ctx_cpu);
    ggml_tensor * out_t = build(ctx_tgt);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    // ids: deterministic per-(j, row) gathers within [0, ne0)
    {
        std::vector<int32_t> ids(n_gather * ne1);
        std::mt19937 rng(1234);
        for (int64_t r = 0; r < ne1; ++r) {
            for (int64_t j = 0; j < n_gather; ++j) {
                ids[r * n_gather + j] = int32_t(rng() % ne0);
            }
        }
        ggml_backend_tensor_set(ggml_get_tensor(ctx_cpu, "b"), ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ggml_get_tensor(ctx_tgt, "b"), ids.data(), 0, ids.size() * sizeof(int32_t));
    }

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 0.0);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_sqrt_softplus(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, int64_t n) {
    char name[128];
    snprintf(name, sizeof(name), "sqrt_softplus n=%ld", (long)n);

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * a_c = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, n);
    ggml_set_name(a_c, "a");
    ggml_tensor * out_c = ggml_sqrt_softplus(ctx_cpu, a_c);

    ggml_tensor * a_t = ggml_new_tensor_1d(ctx_tgt, GGML_TYPE_F32, n);
    ggml_set_name(a_t, "a");
    ggml_tensor * out_t = ggml_sqrt_softplus(ctx_tgt, a_t);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    // Fill with a wide range so both the linear (x > 20) and log1p(exp) branches run
    init_tensor_uniform(a_c, -30.0f, 30.0f);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check_float(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 2e-5);
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

    // DSV4 shapes: hc_attn_pre is [4096, 4, n_tokens]; the MoE weighted sum is
    // [n_embd, n_expert_used, n_tokens] with per-expert scales.
    test_mul_multi_add(backend_cpu, backend_tgt, 4096, 4, 1, false);
    test_mul_multi_add(backend_cpu, backend_tgt, 4096, 4, 3, false);
    test_mul_multi_add(backend_cpu, backend_tgt, 2048, 6, 2, true);
    test_mul_multi_add(backend_cpu, backend_tgt, 256, 3, 5, false);

    // Hadamard block sizes used by DSV4 (indexer head 128, latent state 64, q/kv 512)
    for (int nh : {64, 128, 256, 512}) {
        test_hadamard(backend_cpu, backend_tgt, GGML_TYPE_F32, 512, 5, nh);
        test_hadamard(backend_cpu, backend_tgt, GGML_TYPE_F32, 128, 3, nh == 128 ? nh : 64);
    }
    test_hadamard(backend_cpu, backend_tgt, GGML_TYPE_F16, 512, 4, 512);
    test_hadamard(backend_cpu, backend_tgt, GGML_TYPE_F16, 128, 8, 64);

    test_fill(backend_cpu, backend_tgt, GGML_TYPE_F32, 4097, -INFINITY);
    test_fill(backend_cpu, backend_tgt, GGML_TYPE_F32, 1024, 0.0f);
    test_fill(backend_cpu, backend_tgt, GGML_TYPE_F16, 512, -INFINITY);

    test_get_rows_dim0(backend_cpu, backend_tgt, GGML_TYPE_F32, 1024, 1, 512);
    test_get_rows_dim0(backend_cpu, backend_tgt, GGML_TYPE_F16, 768, 2, 512);
    test_get_rows_dim0(backend_cpu, backend_tgt, GGML_TYPE_F16, 256, 3, 64);

    test_sqrt_softplus(backend_cpu, backend_tgt, 256);

    printf("%s: %d failures\n", ggml_backend_name(backend_tgt), n_failures);
    return n_failures ? 1 : 0;
}
