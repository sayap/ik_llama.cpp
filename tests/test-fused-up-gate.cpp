// Standalone correctness test for GGML_OP_FUSED_UP_GATE / GGML_OP_MOE_FUSED_UP_GATE:
// builds the ops, computes them with the CPU backend and a target backend, and compares.
//
// Usage: test-fused-up-gate <backend>   (e.g. "Vulkan0", "CUDA0", "CPU")
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
    size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) data[i] = dist(rng);

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> datah(size);
        for (size_t i = 0; i < size; i++) datah[i] = ggml_fp32_to_fp16(data[i]);
        ggml_backend_tensor_set(tensor, datah.data(), 0, size * sizeof(ggml_fp16_t));
    } else if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> datab(size);
        for (size_t i = 0; i < size; i++) datab[i] = ggml_fp32_to_bf16(data[i]);
        ggml_backend_tensor_set(tensor, datab.data(), 0, size * sizeof(ggml_bf16_t));
    } else if (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(size % ggml_blck_size(tensor->type) == 0);
        // rows may carry a per-row scale header (row_meta_size), so the total size is
        // nrows * row_size(ne0), not ggml_row_size(type, nelements)
        const size_t nrows = size / tensor->ne[0];
        std::vector<uint8_t> dataq(nrows * ggml_row_size(tensor->type, tensor->ne[0]));
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        struct quantize_user_data qdata = { false, false };
        ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, nrows, tensor->ne[0], imatrix.data(), &qdata);
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else {
        GGML_ABORT("unsupported test type");
    }
}

static void init_ids(ggml_tensor * ids, int n_mats) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    for (int64_t r = 0; r < ggml_nrows(ids); r++) {
        std::vector<int32_t> data(ids->ne[0]);
        for (int i = 0; i < ids->ne[0]; i++) data[i] = i % n_mats;
        std::shuffle(data.begin(), data.end(), rng);
        ggml_backend_tensor_set(ids, data.data(), r * ids->nb[1], ids->ne[0] * sizeof(int32_t));
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

// The KT-family CPU vec_dot kernels carry a calibration factor (1.01/1.05) that
// the other paths do not, so the CPU reference diverges for those types (same as
// in test-iqk-quants).
static double test_iqk_tolerance(ggml_type type_a) {
    switch (type_a) {
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
            return 5.0;
        default:
            return 0.0;
    }
}

// The CPU reference is only trustworthy for a subset of the IQK/KT families:
// - the CPU optimized iqk kernels diverge from the format's scalar dequant for several
//   experimental types (see docs/Vulkan.md), so the CPU-vs-target comparison is restricted
//   to the base IQK types (IQ2_K..IQ6_K, row_meta_size = 0) plus the well-established
//   types; the KS/KL/KT row-meta types are only checked fused-vs-non-fused on the same
//   target backend
static bool test_cpu_reference_ok(ggml_type type_a) {
    switch (type_a) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
            return true;
        default:
            return false;
    }
}

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
        for (size_t i = 0; i < nelements && i < 16; i++) {
            fprintf(stderr, "  [%zu] cpu=%g tgt=%g\n", i, a[i], b[i]);
        }
        n_failures++;
    } else {
        printf("OK   %s (max abs diff = %g)\n", name, err);
    }
}

static void test_dense(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_type type_a, int64_t k, int64_t m, int64_t n_tokens, ggml_unary_op op) {
    char name[256];
    snprintf(name, sizeof(name), "fused_up_gate %s k=%ld m=%ld n=%ld op=%d",
            ggml_type_name(type_a), (long)k, (long)m, (long)n_tokens, (int)op);

    ggml_init_params params = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    // up/gate/b must be set on the CPU backend and copied to the target
    ggml_tensor * up_c   = ggml_new_tensor_2d(ctx_cpu, type_a, k, m);
    ggml_set_name(up_c, "up");
    ggml_tensor * gate_c = ggml_new_tensor_2d(ctx_cpu, type_a, k, m);
    ggml_set_name(gate_c, "gate");
    ggml_tensor * b_c    = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, k, n_tokens);
    ggml_set_name(b_c, "b");
    ggml_tensor * out_c  = ggml_fused_up_gate(ctx_cpu, up_c, gate_c, b_c, op);

    ggml_tensor * up_t   = ggml_new_tensor_2d(ctx_tgt, type_a, k, m);
    ggml_set_name(up_t, "up");
    ggml_tensor * gate_t = ggml_new_tensor_2d(ctx_tgt, type_a, k, m);
    ggml_set_name(gate_t, "gate");
    ggml_tensor * b_t    = ggml_new_tensor_2d(ctx_tgt, GGML_TYPE_F32, k, n_tokens);
    ggml_set_name(b_t, "b");
    ggml_tensor * out_t  = ggml_fused_up_gate(ctx_tgt, up_t, gate_t, b_t, op);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    init_tensor_uniform(up_c);
    init_tensor_uniform(gate_c);
    init_tensor_uniform(b_c);

    // copy the same data to the target backend tensors (by name)
    for (ggml_tensor * t_c = ggml_get_first_tensor(ctx_cpu); t_c != NULL; t_c = ggml_get_next_tensor(ctx_cpu, t_c)) {
        if (t_c->data == nullptr || t_c->name[0] == '\0') continue;
        ggml_tensor * t_t = ggml_get_tensor(ctx_tgt, t_c->name);
        if (t_t != nullptr) {
            ggml_backend_tensor_copy(t_c, t_t);
        }
    }

    // The CPU-vs-target comparison is only done when the CPU reference is trustworthy for
    // the type; the fused graph is always computed on the target backend because the
    // fused-vs-non-fused reference below needs the result.
    const bool cpu_ok = test_cpu_reference_ok(type_a);
    const double cpu_tol = type_a == GGML_TYPE_IQ4_XS ? 5.0 : 0.2; // f16acc precision (GPU dependent)
    if (cpu_ok) {
        check(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, std::max(cpu_tol, test_iqk_tolerance(type_a)));
    } else {
        ggml_cgraph * gf_tgt = ggml_new_graph(ctx_tgt);
        ggml_build_forward_expand(gf_tgt, out_t);
        fprintf(stderr, "[check] computing %s with backend %s\n", name, ggml_backend_name(backend_tgt));
        fflush(stderr);
        if (ggml_backend_graph_compute(backend_tgt, gf_tgt) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL %s: backend compute failed\n", name);
            n_failures++;
            ggml_free(ctx_cpu);
            ggml_free(ctx_tgt);
            return;
        }
    }

    // reference: non-fused graph (mul_mat + fused_mul_unary) on the same target backend
    // -> should match the fused result to within ~1e-4 if the fused path is correct
    {
        ggml_context * ctx_ref = ggml_init(params);
        ggml_tensor * up_r   = ggml_new_tensor_2d(ctx_ref, type_a, k, m);
        ggml_tensor * gate_r = ggml_new_tensor_2d(ctx_ref, type_a, k, m);
        ggml_tensor * b_r    = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, k, n_tokens);
        ggml_tensor * u_r    = ggml_mul_mat(ctx_ref, up_r, b_r);
        ggml_tensor * g_r    = ggml_mul_mat(ctx_ref, gate_r, b_r);
        ggml_tensor * out_r  = ggml_fused_mul_unary(ctx_ref, g_r, u_r, op);
        ggml_backend_alloc_ctx_tensors(ctx_ref, backend_tgt);
        ggml_backend_tensor_copy(up_c, up_r);
        ggml_backend_tensor_copy(gate_c, gate_r);
        ggml_backend_tensor_copy(b_c, b_r);

        ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
        ggml_build_forward_expand(gf_ref, out_r);
        if (ggml_backend_graph_compute(backend_tgt, gf_ref) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL %s: reference compute failed\n", name);
            n_failures++;
        } else {
            const size_t ne = ggml_nelements(out_t);
            std::vector<float> vf(ne), vr(ne);
            ggml_backend_tensor_get(out_t, vf.data(), 0, ne * sizeof(float));
            ggml_backend_tensor_get(out_r, vr.data(), 0, ne * sizeof(float));
            double err = max_abs_diff(vf.data(), vr.data(), ne);
            if (err > 1e-4) {
                fprintf(stderr, "FAIL %s: fused vs non-fused diff = %g\n", name, err);
                n_failures++;
            } else {
                printf("OK   %s (fused vs non-fused diff = %g)\n", name, err);
            }
        }
        ggml_free(ctx_ref);
    }

    ggml_free(ctx_cpu);
    ggml_free(ctx_tgt);
}

static void test_moe(ggml_backend_t backend_cpu, ggml_backend_t backend_tgt,
        ggml_type type_a, int n_mats, int n_used, int64_t m, int64_t k, int64_t n_tokens,
        bool fused, bool bias, ggml_unary_op op) {
    char name[256];
    snprintf(name, sizeof(name), "moe_fused_up_gate %s mats=%d used=%d m=%ld k=%ld n=%ld %s %s op=%d",
            ggml_type_name(type_a), n_mats, n_used, (long)m, (long)k, (long)n_tokens,
            fused ? "fused" : "sep", bias ? "bias" : "nobias", (int)op);

    ggml_init_params params = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx_cpu = ggml_init(params);
    ggml_context * ctx_tgt = ggml_init(params);

    auto build = [&](ggml_context * ctx) {
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, n_tokens);
        ggml_set_name(ids, "ids_parent");
        if (n_used != n_mats) {
            ids = ggml_view_2d(ctx, ids, n_used, n_tokens, ids->nb[1], 0);
        }
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
        ggml_set_name(b, "b");
        if (fused) {
            ggml_tensor * as   = ggml_new_tensor_3d(ctx, type_a, k, 2*m, n_mats);
            ggml_set_name(as, "as");
            ggml_tensor * as_b = bias ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*m, n_mats) : nullptr;
            if (as_b) ggml_set_name(as_b, "as_b");
            return ggml_moe_up_gate_ext(ctx, as, nullptr, b, ids, as_b, nullptr, op);
        }
        ggml_tensor * up     = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_set_name(up, "up");
        ggml_tensor * gate   = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_set_name(gate, "gate");
        ggml_tensor * up_b   = bias ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m, n_mats) : nullptr;
        if (up_b) ggml_set_name(up_b, "up_b");
        ggml_tensor * gate_b = bias ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m, n_mats) : nullptr;
        if (gate_b) ggml_set_name(gate_b, "gate_b");
        return ggml_moe_up_gate_ext(ctx, up, gate, b, ids, up_b, gate_b, op);
    };

    ggml_tensor * out_c = build(ctx_cpu);
    ggml_tensor * out_t = build(ctx_tgt);

    ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    ggml_backend_alloc_ctx_tensors(ctx_tgt, backend_tgt);

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_cpu); t != NULL; t = ggml_get_next_tensor(ctx_cpu, t)) {
        if (t->type == GGML_TYPE_I32) {
            if (t->view_src != nullptr || t->op == GGML_OP_VIEW) continue;
            init_ids(t, n_mats);
        } else {
            init_tensor_uniform(t);
        }
    }

    // copy the same data to the target backend tensors (by name; skip views)
    for (ggml_tensor * t_c = ggml_get_first_tensor(ctx_cpu); t_c != NULL; t_c = ggml_get_next_tensor(ctx_cpu, t_c)) {
        if (t_c->data == nullptr || t_c->name[0] == '\0') continue;
        if (t_c->view_src != nullptr || t_c->op == GGML_OP_VIEW) continue;
        ggml_tensor * t_t = ggml_get_tensor(ctx_tgt, t_c->name);
        if (t_t != nullptr) {
            ggml_backend_tensor_copy(t_c, t_t);
        }
    }

    // The CPU-vs-target comparison is only done when the CPU reference is trustworthy for
    // the type; the fused graph is always computed on the target backend because the
    // fused-vs-non-fused reference below needs the result.
    const bool cpu_ok = test_cpu_reference_ok(type_a);
    const double cpu_tol = type_a == GGML_TYPE_IQ4_XS ? 5.0 : 0.2; // f16acc precision (GPU dependent)
    if (cpu_ok) {
        check(name, backend_cpu, backend_tgt, ctx_cpu, ctx_tgt, out_c, out_t, std::max(cpu_tol, test_iqk_tolerance(type_a)));
    } else {
        ggml_cgraph * gf_tgt = ggml_new_graph(ctx_tgt);
        ggml_build_forward_expand(gf_tgt, out_t);
        fprintf(stderr, "[check] computing %s with backend %s\n", name, ggml_backend_name(backend_tgt));
        fflush(stderr);
        if (ggml_backend_graph_compute(backend_tgt, gf_tgt) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL %s: backend compute failed\n", name);
            n_failures++;
            ggml_free(ctx_cpu);
            ggml_free(ctx_tgt);
            return;
        }
    }

    // reference: non-fused graph (mul_mat_id + fused_mul_unary) on the same target backend.
    // Only valid without biases (the reference cannot easily add per-expert biases).
    if (bias) {
        ggml_free(ctx_cpu);
        ggml_free(ctx_tgt);
        return;
    }
    {
        ggml_context * ctx_ref = ggml_init(params);
        ggml_tensor * ids_r = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_I32, n_mats, n_tokens);
        ggml_set_name(ids_r, "ids_parent");
        if (n_used != n_mats) {
            ids_r = ggml_view_2d(ctx_ref, ids_r, n_used, n_tokens, ids_r->nb[1], 0);
        }
        ggml_tensor * b_r = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, k, 1, n_tokens);
        ggml_set_name(b_r, "b");
        ggml_tensor * g_r, * u_r;
        if (fused) {
            // separate gate/up tensors (the Vulkan mul_mat_id cannot handle views into a
            // [k, 2m, n_mats] matrix because the per-expert stride differs from ne00*ne01)
            ggml_tensor * gate = ggml_new_tensor_3d(ctx_ref, type_a, k, m, n_mats);
            ggml_set_name(gate, "gate_ref");
            ggml_tensor * up = ggml_new_tensor_3d(ctx_ref, type_a, k, m, n_mats);
            ggml_set_name(up, "up_ref");
            g_r = ggml_mul_mat_id(ctx_ref, gate, b_r, ids_r);
            u_r = ggml_mul_mat_id(ctx_ref, up,   b_r, ids_r);
        } else {
            ggml_tensor * up = ggml_new_tensor_3d(ctx_ref, type_a, k, m, n_mats);
            ggml_set_name(up, "up");
            ggml_tensor * gate = ggml_new_tensor_3d(ctx_ref, type_a, k, m, n_mats);
            ggml_set_name(gate, "gate");
            g_r = ggml_mul_mat_id(ctx_ref, gate, b_r, ids_r);
            u_r = ggml_mul_mat_id(ctx_ref, up,   b_r, ids_r);
        }
        ggml_tensor * out_r = ggml_fused_mul_unary(ctx_ref, g_r, u_r, op);
        ggml_backend_alloc_ctx_tensors(ctx_ref, backend_tgt);
        for (ggml_tensor * t_c = ggml_get_first_tensor(ctx_cpu); t_c != NULL; t_c = ggml_get_next_tensor(ctx_cpu, t_c)) {
            if (t_c->data == nullptr || t_c->name[0] == '\0') continue;
            if (t_c->view_src != nullptr || t_c->op == GGML_OP_VIEW) continue;
            if (fused && (strcmp(t_c->name, "as") == 0)) {
                // split the fused matrix into gate (first half) and up (second half) reference tensors
                ggml_tensor * gate_r = ggml_get_tensor(ctx_ref, "gate_ref");
                ggml_tensor * up_r   = ggml_get_tensor(ctx_ref, "up_ref");
                std::vector<uint8_t> row(ggml_row_size(type_a, k));
                for (int e = 0; e < n_mats; e++) {
                    for (int64_t r = 0; r < m; r++) {
                        ggml_backend_tensor_get(t_c, row.data(), e*t_c->nb[2] + r*t_c->nb[1], row.size());
                        ggml_backend_tensor_set(gate_r, row.data(), e*gate_r->nb[2] + r*gate_r->nb[1], row.size());
                        ggml_backend_tensor_get(t_c, row.data(), e*t_c->nb[2] + (m+r)*t_c->nb[1], row.size());
                        ggml_backend_tensor_set(up_r, row.data(), e*up_r->nb[2] + r*up_r->nb[1], row.size());
                    }
                }
                continue;
            }
            ggml_tensor * t_r = ggml_get_tensor(ctx_ref, t_c->name);
            if (t_r != nullptr) {
                ggml_backend_tensor_copy(t_c, t_r);
            }
        }

        ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
        ggml_build_forward_expand(gf_ref, out_r);
        if (ggml_backend_graph_compute(backend_tgt, gf_ref) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL %s: reference compute failed\n", name);
            n_failures++;
        } else {
            const size_t ne = ggml_nelements(out_t);
            std::vector<float> vf(ne), vr(ne);
            ggml_backend_tensor_get(out_t, vf.data(), 0, ne * sizeof(float));
            ggml_backend_tensor_get(out_r, vr.data(), 0, ne * sizeof(float));
            double err = max_abs_diff(vf.data(), vr.data(), ne);
            if (err > 1e-4) {
                fprintf(stderr, "FAIL %s: fused vs non-fused diff = %g\n", name, err);
                n_failures++;
            } else {
                printf("OK   %s (fused vs non-fused diff = %g)\n", name, err);
            }
        }
        ggml_free(ctx_ref);
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

    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_F16, GGML_TYPE_BF16 };
    // IQK/KT (QK_K = 256) families: exercised via the vec (mul_mat_vec) path for single
    // token and the dequant-to-F16 path for batches
    const ggml_type types_iqk[] = {
        GGML_TYPE_IQ2_K, GGML_TYPE_IQ3_K, GGML_TYPE_IQ4_K, GGML_TYPE_IQ5_K, GGML_TYPE_IQ6_K,
        GGML_TYPE_IQ2_KS, GGML_TYPE_IQ3_KS, GGML_TYPE_IQ4_KS, GGML_TYPE_IQ4_KSS, GGML_TYPE_IQ5_KS,
        GGML_TYPE_IQ2_KL, GGML_TYPE_IQ1_KT, GGML_TYPE_IQ2_KT, GGML_TYPE_IQ3_KT, GGML_TYPE_IQ4_KT,
    };

    // dense: single-token decode and small batches
    for (ggml_type type_a : types) {
        for (ggml_unary_op op : { GGML_UNARY_OP_SILU, GGML_UNARY_OP_GELU, GGML_UNARY_OP_RELU }) {
            test_dense(backend_cpu, backend_tgt, type_a, 256, 512, 1, op);
            test_dense(backend_cpu, backend_tgt, type_a, 256, 512, 7, op);
        }
    }
    for (ggml_type type_a : types_iqk) {
        test_dense(backend_cpu, backend_tgt, type_a, 256, 512, 1, GGML_UNARY_OP_SILU);
        test_dense(backend_cpu, backend_tgt, type_a, 256, 512, 7, GGML_UNARY_OP_SILU);
    }

    // MoE: fused and separate weights, with and without bias, single and multi token
    for (ggml_type type_a : types) {
        for (ggml_unary_op op : { GGML_UNARY_OP_SILU, GGML_UNARY_OP_GELU }) {
            for (bool fused : { true, false }) {
                for (bool bias : { false, true }) {
                    test_moe(backend_cpu, backend_tgt, type_a, 8, 2, 256, 256, 1, fused, bias, op);
                    test_moe(backend_cpu, backend_tgt, type_a, 8, 2, 256, 256, 5, fused, bias, op);
                }
            }
        }
    }
    for (ggml_type type_a : types_iqk) {
        for (bool fused : { true, false }) {
            test_moe(backend_cpu, backend_tgt, type_a, 8, 2, 256, 256, 1, fused, false, GGML_UNARY_OP_SILU);
            test_moe(backend_cpu, backend_tgt, type_a, 8, 2, 256, 256, 5, fused, false, GGML_UNARY_OP_SILU);
        }
    }

    printf("%s: %d failures\n", tgt_name, n_failures);
    return n_failures == 0 ? 0 : 1;
}
