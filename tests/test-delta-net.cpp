// Standalone correctness test for the gated delta-net Vulkan ops:
// GGML_OP_SSM_CONV, GGML_OP_L2_NORM, GGML_UNARY_OP_SOFTPLUS, GGML_OP_DELTA_NET.
//
// Usage: test-delta-net <backend>   (e.g. "Vulkan0", "CUDA0", "CPU")
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
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
    if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> idata(size);
        for (size_t i = 0; i < size; i++) idata[i] = (int32_t)data[i];
        ggml_backend_tensor_set(tensor, idata.data(), 0, size * sizeof(int32_t));
    } else {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    }
}

static double max_abs_diff(const float * a, const float * b, size_t n) {
    double max = 0.0;
    for (size_t i = 0; i < n; i++) {
        max = std::max(max, (double)std::abs(a[i] - b[i]));
    }
    return max;
}

static int n_failures = 0;

static void check(const char * name, ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
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
    ggml_backend_tensor_get(out_cpu, a.data(), 0, nelements * sizeof(float));
    ggml_backend_tensor_get(out_tgt, b.data(), 0, nelements * sizeof(float));

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

static void test_l2_norm(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, int64_t n, int64_t rows) {
    char name[256];
    snprintf(name, sizeof(name), "l2_norm n=%ld rows=%ld", (long)n, (long)rows);
    ggml_init_params params = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * a_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, n, rows);
    ggml_set_name(a_c, "a");
    ggml_tensor * out_c = ggml_l2_norm(ctx_cpu, a_c, 1e-6f);

    ggml_tensor * a_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, n, rows);
    ggml_set_name(a_t, "a");
    ggml_tensor * out_t = ggml_l2_norm(ctx_tgt, a_t, 1e-6f);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    init_tensor_uniform(a_c);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 1e-4);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_softplus(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, int64_t n) {
    char name[256];
    snprintf(name, sizeof(name), "softplus n=%ld", (long)n);
    ggml_init_params params = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * a_c = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, n);
    ggml_set_name(a_c, "a");
    ggml_tensor * out_c = ggml_softplus(ctx_cpu, a_c);

    ggml_tensor * a_t = ggml_new_tensor_1d(ctx_tgt, GGML_TYPE_F32, n);
    ggml_set_name(a_t, "a");
    ggml_tensor * out_t = ggml_softplus(ctx_tgt, a_t);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    init_tensor_uniform(a_c, -1.0f, 1.0f);
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 1e-4);
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_ssm_conv(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt, int64_t nc, int64_t nr, int64_t n_t) {
    char name[256];
    snprintf(name, sizeof(name), "ssm_conv nc=%ld nr=%ld n_t=%ld", (long)nc, (long)nr, (long)n_t);
    ggml_init_params params = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    ggml_tensor * s_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, nc - 1, nr);
    ggml_set_name(s_c, "s");
    ggml_tensor * x_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, nr, n_t);
    ggml_set_name(x_c, "x");
    ggml_tensor * c_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, nc, nr);
    ggml_set_name(c_c, "c");
    ggml_tensor * sq_c = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_I32, 1, n_t);
    ggml_set_name(sq_c, "sq");
    ggml_tensor * saved_c = ggml_new_tensor_1d(ctx_cpu, GGML_TYPE_F32, (nc - 1) * nr * n_t);
    ggml_set_name(saved_c, "saved");
    ggml_tensor * raw_c = ggml_ssm_conv(ctx_cpu, s_c, x_c, c_c, sq_c, saved_c);
    // the real graph always feeds ssm_conv into a view + silu; keep that shape so the
    // CPU's nc==4 fast path has valid following nodes to look at
    ggml_tensor * conv_c = ggml_view_2d(ctx_cpu, raw_c, nr, n_t, ggml_row_size(GGML_TYPE_F32, nr), 0);
    ggml_tensor * out_c = ggml_silu(ctx_cpu, conv_c);

    ggml_tensor * s_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, nc - 1, nr);
    ggml_set_name(s_t, "s");
    ggml_tensor * x_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, nr, n_t);
    ggml_set_name(x_t, "x");
    ggml_tensor * c_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, nc, nr);
    ggml_set_name(c_t, "c");
    ggml_tensor * sq_t = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_I32, 1, n_t);
    ggml_set_name(sq_t, "sq");
    ggml_tensor * saved_t = ggml_new_tensor_1d(ctx_tgt, GGML_TYPE_F32, (nc - 1) * nr * n_t);
    ggml_set_name(saved_t, "saved");
    ggml_tensor * raw_t = ggml_ssm_conv(ctx_tgt, s_t, x_t, c_t, sq_t, saved_t);
    ggml_tensor * conv_t = ggml_view_2d(ctx_tgt, raw_t, nr, n_t, ggml_row_size(GGML_TYPE_F32, nr), 0);
    ggml_tensor * out_t = ggml_silu(ctx_tgt, conv_t);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);
    init_tensor_uniform(s_c);
    init_tensor_uniform(x_c);
    init_tensor_uniform(c_c);
    std::vector<int32_t> sq_data(n_t, 0);
    ggml_backend_tensor_set(sq_c, sq_data.data(), 0, n_t * sizeof(int32_t));
    copy_tensors_by_name(ctx_cpu, ctx_tgt);
    ggml_backend_tensor_set(sq_t, sq_data.data(), 0, n_t * sizeof(int32_t));

    // compare the raw ssm_conv result (conv output + saved conv state)
    {
        ggml_cgraph * gf_cpu = ggml_new_graph(ctx_cpu);
        ggml_build_forward_expand(gf_cpu, out_c);
        ggml_cplan plan = ggml_graph_plan(gf_cpu, 4);
        if (plan.work_size > 0) plan.work_data = (uint8_t *)malloc(plan.work_size);
        ggml_graph_compute(gf_cpu, &plan);
        free(plan.work_data);

        ggml_cgraph * gf_tgt = ggml_new_graph(ctx_tgt);
        ggml_build_forward_expand(gf_tgt, out_t);
        fprintf(stderr, "[check] computing %s with backend %s\n", name, ggml_backend_name(backend_tgt));
        fflush(stderr);
        if (ggml_backend_graph_compute(backend_tgt, gf_tgt) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL %s: backend compute failed\n", name);
            n_failures++;
        } else {
            const size_t ne = ggml_nelements(raw_c);
            std::vector<float> a(ne), b(ne);
            ggml_backend_tensor_get(raw_c, a.data(), 0, ne * sizeof(float));
            ggml_backend_tensor_get(raw_t, b.data(), 0, ne * sizeof(float));
            double err = max_abs_diff(a.data(), b.data(), ne);
            if (err > 1e-4) {
                fprintf(stderr, "FAIL %s: max abs diff = %g > %g\n", name, err, 1e-4);
                for (size_t i = 0; i < ne && i < 32; i++) {
                    fprintf(stderr, "  [%zu] cpu=%g tgt=%g\n", i, a[i], b[i]);
                }
                n_failures++;
            } else {
                printf("OK   %s (max abs diff = %g)\n", name, err);
            }

            // per-step conv checkpoint
            const size_t ns = ggml_nelements(saved_c);
            std::vector<float> sa(ns), sb(ns);
            ggml_backend_tensor_get(saved_c, sa.data(), 0, ns * sizeof(float));
            ggml_backend_tensor_get(saved_t, sb.data(), 0, ns * sizeof(float));
            double serr = max_abs_diff(sa.data(), sb.data(), ns);
            if (serr > 1e-4) {
                fprintf(stderr, "FAIL %s saved: max abs diff = %g > %g\n", name, serr, 1e-4);
                for (size_t i = 0; i < ns && i < 32; i++) {
                    fprintf(stderr, "  [%zu] cpu=%g tgt=%g\n", i, sa[i], sb[i]);
                }
                n_failures++;
            } else {
                printf("OK   %s saved (max abs diff = %g)\n", name, serr);
            }
        }
    }
    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

// Mirror build_fused_delta_net()'s input layout: permuted v/g/beta views and
// l2_norm'ed q/k (or their decode-time permuted equivalents).
static void test_delta_net(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        int64_t S, int64_t H_k, int64_t H_v, int64_t n_tok, int repeat_type) {
    char name[256];
    snprintf(name, sizeof(name), "delta_net S=%ld H_k=%ld H_v=%ld n_tok=%ld rep=%d",
            (long)S, (long)H_k, (long)H_v, (long)n_tok, repeat_type);
    const int64_t qkv_dim = 2 * S * H_k + S * H_v;

    ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    auto build = [&](ggml_context * ctx, ggml_tensor ** saved_out) {
        // conv output (silu'd) [qkv_dim, n_tok] laid out q | k | v along dim0
        ggml_tensor * conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qkv_dim, n_tok);
        ggml_set_name(conv, "conv");

        ggml_tensor * q = ggml_view_4d(ctx, conv, S, H_k, n_tok, 1,
                ggml_row_size(GGML_TYPE_F32, S), ggml_row_size(GGML_TYPE_F32, qkv_dim),
                ggml_row_size(GGML_TYPE_F32, qkv_dim) * n_tok, 0);
        ggml_tensor * k = ggml_view_4d(ctx, conv, S, H_k, n_tok, 1,
                ggml_row_size(GGML_TYPE_F32, S), ggml_row_size(GGML_TYPE_F32, qkv_dim),
                ggml_row_size(GGML_TYPE_F32, qkv_dim) * n_tok,
                ggml_row_size(GGML_TYPE_F32, S * H_k));
        ggml_tensor * v = ggml_view_4d(ctx, conv, S, H_v, n_tok, 1,
                ggml_row_size(GGML_TYPE_F32, S), ggml_row_size(GGML_TYPE_F32, qkv_dim),
                ggml_row_size(GGML_TYPE_F32, qkv_dim) * n_tok,
                ggml_row_size(GGML_TYPE_F32, 2 * S * H_k));

        if (n_tok > 1) {
            q = ggml_permute(ctx, q, 0, 2, 1, 3);
            k = ggml_permute(ctx, k, 0, 2, 1, 3);
            q = ggml_l2_norm(ctx, q, 1e-6f);
            k = ggml_l2_norm(ctx, k, 1e-6f);
        } else {
            q = ggml_l2_norm(ctx, q, 1e-6f);
            k = ggml_l2_norm(ctx, k, 1e-6f);
            q = ggml_permute(ctx, q, 0, 2, 1, 3);
            k = ggml_permute(ctx, k, 0, 2, 1, 3);
        }

        v = ggml_permute(ctx, v, 0, 2, 1, 3);

        // gate [H_v, n_tok] -> permute to [n_tok, 1, H_v, 1]
        ggml_tensor * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H_v, n_tok);
        ggml_set_name(gate, "gate");
        ggml_tensor * g = ggml_permute(ctx, gate, 2, 0, 3, 1);

        // beta [H_v, 1, n_tok] -> permute to [1, n_tok, H_v, 1]
        ggml_tensor * beta = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H_v, 1, n_tok);
        ggml_set_name(beta, "beta");
        ggml_tensor * b = ggml_permute(ctx, beta, 2, 0, 1, 3);

        ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, S * H_v);
        ggml_set_name(state, "state");

        ggml_tensor * saved = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (n_tok - 1) * S * S * H_v);
        ggml_set_name(saved, "saved");

        ggml_tensor * out = ggml_delta_net(ctx, q, k, v, g, b, state, saved);
        out->op_params[0] = repeat_type;
        *saved_out = saved;
        return out;
    };

    ggml_tensor * saved_c = nullptr;
    ggml_tensor * saved_t = nullptr;
    ggml_tensor * out_c = build(ctx_cpu, &saved_c);
    ggml_tensor * out_t = build(ctx_tgt, &saved_t);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_cpu); t != NULL; t = ggml_get_next_tensor(ctx_cpu, t)) {
        if (t->data == nullptr || t->view_src != nullptr || t->op == GGML_OP_VIEW) continue;
        init_tensor_uniform(t);
    }
    copy_tensors_by_name(ctx_cpu, ctx_tgt);

    check(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, 0.05);

    if (n_tok > 1) {
        const size_t ns = ggml_nelements(saved_c);
        std::vector<float> sa(ns), sb(ns);
        ggml_backend_tensor_get(saved_c, sa.data(), 0, ns * sizeof(float));
        ggml_backend_tensor_get(saved_t, sb.data(), 0, ns * sizeof(float));
        double serr = max_abs_diff(sa.data(), sb.data(), ns);
        if (serr > 0.05) {
            fprintf(stderr, "FAIL %s saved: max abs diff = %g > %g\n", name, serr, 0.05);
            for (size_t i = 0; i < ns && i < 32; i++) {
                fprintf(stderr, "  [%zu] cpu=%g tgt=%g\n", i, sa[i], sb[i]);
            }
            n_failures++;
        } else {
            printf("OK   %s saved (max abs diff = %g)\n", name, serr);
        }
    }
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

    test_l2_norm(backend_cpu, backend_tgt, 128, 8);
    test_softplus(backend_cpu, backend_tgt, 1024);
    test_ssm_conv(backend_cpu, backend_tgt, 4, 256, 1);
    test_ssm_conv(backend_cpu, backend_tgt, 4, 256, 5);

    // the qwen35 delta-net decode shape: H_k=16, H_v=48, gqa=3, repeat_type=1
    test_delta_net(backend_cpu, backend_tgt, 128, 16, 48, 1, 1);

    for (int repeat_type : { 0, 1 }) {
        test_delta_net(backend_cpu, backend_tgt, 16, 2, 4, 1, repeat_type);
        test_delta_net(backend_cpu, backend_tgt, 32, 2, 4, 1, repeat_type);
        // Multi-token: the CPU reference's fallback path assumes contiguous v, so only
        // exercise it for head_dim 64/128 where iqk_fused_delta_net handles the strides.
        test_delta_net(backend_cpu, backend_tgt, 64, 2, 4, 3, repeat_type);
        test_delta_net(backend_cpu, backend_tgt, 128, 2, 4, 1, repeat_type);
        test_delta_net(backend_cpu, backend_tgt, 128, 2, 4, 3, repeat_type);
        test_delta_net(backend_cpu, backend_tgt, 128, 2, 4, 5, repeat_type);
    }

    printf("%s: %d failures\n", tgt_name, n_failures);
    return n_failures == 0 ? 0 : 1;
}
