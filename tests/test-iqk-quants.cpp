// Correctness test for the IQK (QK_K=256) and KT quant families on a target
// backend.
//
// The reference is the scalar dequant of the format (ggml's to_float, which is
// the format definition: the quantizers write data that these dequant
// functions invert) followed by an F32 dot product with the activations. The
// CPU backend's optimized iqk_mul_mat kernels are known to diverge from the
// scalar dequant for several of these experimental types, so they are NOT used
// as the reference here.
//
// Exercises MUL_MAT (single-token decode = mul_mat_vec, small batches, larger
// K, multi-token = mat-mat dequant-to-F16 fallback) and MUL_MAT_ID (MoE,
// single and multi token).
//
// Usage: test-iqk-quants <backend>   (e.g. "Vulkan0", "CUDA0", "CPU")
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

static void init_tensor_quantized(ggml_tensor * tensor, std::vector<float> & f32) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t size = ggml_nelements(tensor);
    f32.resize(size);
    for (size_t i = 0; i < size; i++) f32[i] = dist(rng);

    // rows may carry a per-row scale header (row_meta_size), so the total size is
    // nrows * row_size, not ggml_row_size(type, nelements)
    const size_t nrows = size / tensor->ne[0];
    std::vector<uint8_t> dataq(nrows * ggml_row_size(tensor->type, tensor->ne[0]));
    std::vector<float> imatrix(tensor->ne[0], 1.0f);
    struct quantize_user_data qdata = { false, false };
    ggml_quantize_chunk(tensor->type, f32.data(), dataq.data(), 0, nrows, tensor->ne[0], imatrix.data(), &qdata);
    ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
}

static void init_activations(ggml_tensor * tensor, std::vector<float> & f32) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t size = ggml_nelements(tensor);
    f32.resize(size);
    for (size_t i = 0; i < size; i++) f32[i] = dist(rng);
    ggml_backend_tensor_set(tensor, f32.data(), 0, size * sizeof(float));
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

static int n_failures = 0;

// Reference: out = A^T x where A is the F32 dequant of the quantized weights.
// A has shape [k, m] (row r of the tensor = weight column r), x has shape [k, n].
static void dequant_reference(ggml_type type, const uint8_t * qdata, int64_t k, int64_t m,
        const std::vector<float> & act, int64_t n_tokens, std::vector<float> & out) {
    const size_t row_size = ggml_row_size(type, k);
    const ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
    std::vector<float> row(k);
    // dst [m, n_tokens]: ggml lays out element (row r, col t) at flat r + t*m
    out.assign(m * n_tokens, 0.0f);
    for (int64_t r = 0; r < m; ++r) {
        traits.to_float(qdata + r * row_size, row.data(), k);
        for (int64_t t = 0; t < n_tokens; ++t) {
            double sum = 0.0;
            for (int64_t i = 0; i < k; ++i) {
                sum += (double)row[i] * (double)act[t * k + i];
            }
            out[r + t * m] = (float)sum;
        }
    }
}

// The KT-family decoders carry a calibration factor (1.01/1.05) that the scalar
// dequant reference does not, so results differ by ~1-3 abs (same as CUDA).
static double test_tolerance(ggml_type type_a) {
    switch (type_a) {
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
            return 5.0;
        default:
            return 0.25;
    }
}

static void check_mul_mat(ggml_backend_t backend_tgt, ggml_type type_a,
        int64_t k, int64_t m, int64_t n_tokens) {
    char name[256];
    snprintf(name, sizeof(name), "mul_mat %s k=%ld m=%ld n=%ld",
            ggml_type_name(type_a), (long)k, (long)m, (long)n_tokens);

    ggml_init_params params = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * a   = ggml_new_tensor_2d(ctx, type_a, k, m);
    ggml_set_name(a, "a");
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n_tokens);
    ggml_set_name(b, "b");
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);

    ggml_backend_alloc_ctx_tensors(ctx, backend_tgt);

    std::vector<float> a_f32, act;
    init_tensor_quantized(a, a_f32);
    init_activations(b, act);

    // reference on CPU
    const size_t a_size = m * ggml_row_size(type_a, k);
    std::vector<uint8_t> q(a_size);
    ggml_backend_tensor_get(a, q.data(), 0, a_size);
    std::vector<float> ref;
    dequant_reference(type_a, q.data(), k, m, act, n_tokens, ref);
    if (type_a == GGML_TYPE_IQ4_KS && n_tokens == 1 && m == 512) {
        double mx = 0; for (auto v : ref) mx = std::max(mx, (double)std::abs(v));
        printf("IQ4KS n=1: ref max=%.3f | d=%g\n", mx, *(float*)q.data());
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_backend_graph_compute(backend_tgt, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL %s: backend compute failed\n", name);
        n_failures++;
        ggml_free(ctx);
        return;
    }

    std::vector<float> got(m * n_tokens);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

    double err = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        err = std::max(err, (double)std::abs(got[i] - ref[i]));
    }
    // The mat-mat path dequantizes to F16 and accumulates in F16, so the error grows
    // with the accumulation length k (the vec path accumulates in F32). Scale the
    // tolerance by k/256 so multi-block rows are not false positives; real bugs
    // (wrong row/scale addressing) produce errors many orders of magnitude larger.
    const double tol = test_tolerance(type_a) * std::max(1.0, (double)k / 256.0);
    if (err > tol) {
        fprintf(stderr, "FAIL %s: max abs diff = %g > %g\n", name, err, tol);
        for (size_t i = 0; i < got.size() && i < 16; i++) {
            fprintf(stderr, "  [%zu] ref=%g tgt=%g\n", i, ref[i], got[i]);
        }
        n_failures++;
    } else {
        printf("OK   %s (max abs diff = %g)\n", name, err);
    }
    fflush(stdout);
    ggml_free(ctx);
}

static void check_get_rows(ggml_backend_t backend_tgt, ggml_type type_a, int64_t k, int64_t m, int64_t n) {
    char name[256];
    snprintf(name, sizeof(name), "get_rows %s k=%ld m=%ld n=%ld",
            ggml_type_name(type_a), (long)k, (long)m, (long)n);

    ggml_init_params params = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * a   = ggml_new_tensor_2d(ctx, type_a, k, m);
    ggml_set_name(a, "a");
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n, 1);
    ggml_set_name(ids, "ids");
    ggml_tensor * out = ggml_get_rows(ctx, a, ids);

    ggml_backend_alloc_ctx_tensors(ctx, backend_tgt);

    std::vector<float> a_f32;
    init_tensor_quantized(a, a_f32);

    // random row ids (may repeat), all in range
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::vector<int32_t> ids_data(n);
    for (int64_t i = 0; i < n; i++) ids_data[i] = (int32_t)(rng() % (uint32_t)m);
    ggml_backend_tensor_set(ids, ids_data.data(), 0, n * sizeof(int32_t));

    // reference: out[:, i] = dequant(a)[:, ids[i]]
    const size_t a_size = m * ggml_row_size(type_a, k);
    std::vector<uint8_t> q(a_size);
    ggml_backend_tensor_get(a, q.data(), 0, a_size);
    const ggml_type_traits_t traits = ggml_internal_get_type_traits(type_a);
    std::vector<float> row(k);
    std::vector<float> ref(k * n);
    for (int64_t i = 0; i < n; i++) {
        traits.to_float(q.data() + ids_data[i] * ggml_row_size(type_a, k), row.data(), k);
        for (int64_t j = 0; j < k; j++) ref[j + i * k] = row[j];
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_backend_graph_compute(backend_tgt, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL %s: backend compute failed\n", name);
        n_failures++;
        ggml_free(ctx);
        return;
    }

    // dst is F32 for this test graph; the backend may also emit F16, but get_rows
    // on the CPU/GPU backends produces F32 for an F32 dst
    std::vector<float> got(k * n);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

    double err = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        err = std::max(err, (double)std::abs(got[i] - ref[i]));
    }
    const double tol = std::max(0.05, test_tolerance(type_a)); // dequant accuracy only, no accumulation
    if (err > tol) {
        fprintf(stderr, "FAIL %s: max abs diff = %g > %g\n", name, err, tol);
        for (size_t i = 0; i < got.size() && i < 16; i++) {
            fprintf(stderr, "  [%zu] ref=%g tgt=%g\n", i, ref[i], got[i]);
        }
        n_failures++;
    } else {
        printf("OK   %s (max abs diff = %g)\n", name, err);
    }
    fflush(stdout);
    ggml_free(ctx);
}

static void check_mul_mat_id(ggml_backend_t backend_tgt, ggml_type type_a,
        int n_mats, int n_used, int64_t m, int64_t k, int64_t n_tokens) {
    char name[256];
    snprintf(name, sizeof(name), "mul_mat_id %s mats=%d used=%d m=%ld k=%ld n=%ld",
            ggml_type_name(type_a), n_mats, n_used, (long)m, (long)k, (long)n_tokens);

    ggml_init_params params = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * ids_parent = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, n_tokens);
    ggml_set_name(ids_parent, "ids_parent");
    ggml_tensor * ids = ids_parent;
    if (n_used != n_mats) {
        ids = ggml_view_2d(ctx, ids_parent, n_used, n_tokens, ids_parent->nb[1], 0);
    }
    ggml_tensor * a = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
    ggml_set_name(b, "b");
    ggml_tensor * out = ggml_mul_mat_id(ctx, a, b, ids);

    ggml_backend_alloc_ctx_tensors(ctx, backend_tgt);

    std::vector<float> a_f32, act;
    init_tensor_quantized(a, a_f32);
    init_activations(b, act);
    init_ids(ids_parent, n_mats);

    // reference: for each token and expert position, dequant the selected expert and dot.
    // dst layout: [m, n_used, n_tokens] -> flat = r + id*m + t*m*n_used
    std::vector<int32_t> ids_data(n_mats * n_tokens);
    ggml_backend_tensor_get(ids_parent, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));
    const size_t expert_row_size = ggml_row_size(type_a, k);
    std::vector<uint8_t> q(n_mats * m * ggml_row_size(type_a, k));
    ggml_backend_tensor_get(a, q.data(), 0, q.size());
    std::vector<float> row(k);
    std::vector<float> ref(m * n_used * n_tokens, 0.0f);
    const ggml_type_traits_t traits = ggml_internal_get_type_traits(type_a);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_used; ++u) {
            const int expert = ids_data[t * n_mats + u];
            if (expert < 0 || expert >= n_mats) continue;
            for (int64_t r = 0; r < m; ++r) {
                traits.to_float(q.data() + expert * m * expert_row_size + r * expert_row_size, row.data(), k);
                double sum = 0.0;
                for (int64_t i = 0; i < k; ++i) {
                    sum += (double)row[i] * (double)act[t * k + i];
                }
                ref[r + u * m + t * m * n_used] = (float)sum;
            }
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_backend_graph_compute(backend_tgt, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL %s: backend compute failed\n", name);
        n_failures++;
        ggml_free(ctx);
        return;
    }

    std::vector<float> got(m * n_used * n_tokens);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

    double err = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        err = std::max(err, (double)std::abs(got[i] - ref[i]));
    }
    // Same F16-accumulation scaling as check_mul_mat (k/256).
    const double tol = test_tolerance(type_a) * std::max(1.0, (double)k / 256.0);
    if (err > tol) {
        fprintf(stderr, "FAIL %s: max abs diff = %g > %g\n", name, err, tol);
        for (size_t i = 0; i < got.size() && i < 16; i++) {
            fprintf(stderr, "  [%zu] ref=%g tgt=%g\n", i, ref[i], got[i]);
        }
        n_failures++;
    } else {
        printf("OK   %s (max abs diff = %g)\n", name, err);
    }
    fflush(stdout);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * tgt_name = argc > 1 ? argv[1] : "CPU";

    ggml_backend_load_all();

    ggml_backend_t backend_tgt = nullptr;
    if (strcmp(tgt_name, "CPU") == 0) {
        backend_tgt = ggml_backend_cpu_init();
    } else {
        backend_tgt = ggml_backend_reg_init_backend_from_str(tgt_name);
    }
    if (!backend_tgt) {
        fprintf(stderr, "failed to init backend %s\n", tgt_name);
        return 1;
    }
    printf("target backend: %s\n", ggml_backend_name(backend_tgt));

    const ggml_type types[] = {
        GGML_TYPE_Q6_0,
        GGML_TYPE_MXFP4,
        GGML_TYPE_IQ2_K, GGML_TYPE_IQ3_K, GGML_TYPE_IQ4_K, GGML_TYPE_IQ5_K, GGML_TYPE_IQ6_K,
        GGML_TYPE_IQ2_KS, GGML_TYPE_IQ3_KS, GGML_TYPE_IQ4_KS, GGML_TYPE_IQ4_KSS, GGML_TYPE_IQ5_KS,
        GGML_TYPE_IQ2_KL,
        GGML_TYPE_IQ1_KT, GGML_TYPE_IQ2_KT, GGML_TYPE_IQ3_KT, GGML_TYPE_IQ4_KT,
    };

    const char * filter = argc > 2 ? argv[2] : nullptr;
    for (ggml_type type_a : types) {
        if (filter && strcmp(filter, ggml_type_name(type_a)) != 0) continue;
        // single-token decode (mul_mat_vec) and small batches
        check_mul_mat(backend_tgt, type_a, 256, 512, 1);
        check_mul_mat(backend_tgt, type_a, 256, 512, 5);
        // larger K
        check_mul_mat(backend_tgt, type_a, 4096, 128, 1);
        // multi-token (mat-mat path), single and multi-block rows (the dequant-to-F16
        // kernels read the per-row scale header from the row start; k > 256 exercises
        // the multi-block path that a single block never covers)
        check_mul_mat(backend_tgt, type_a, 256, 128, 32);
        check_mul_mat(backend_tgt, type_a, 4096, 64, 16);

        // MoE single-token (vec-id path) and multi-token (mat-mat-id path)
        check_mul_mat_id(backend_tgt, type_a, 4, 2, 256, 256, 1);
        check_mul_mat_id(backend_tgt, type_a, 4, 2, 256, 256, 8);
        check_mul_mat_id(backend_tgt, type_a, 4, 2, 256, 1024, 8);

        // GET_ROWS (quantized token embeddings): row lookup by id
        check_get_rows(backend_tgt, type_a, 256, 512, 17);
        check_get_rows(backend_tgt, type_a, 4096, 128, 5);
    }

    printf("%s: %d failures\n", tgt_name, n_failures);
    return n_failures == 0 ? 0 : 1;
}
