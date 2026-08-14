// Standalone test for the Vulkan split buffer type and the host-side REDUCE op
// used by -sm graph. Does not require a model.
//
// Usage: test-vk-split
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int n_failures = 0;

static bool closef(float a, float b, float tol = 1e-5f) {
    float d = a > b ? a - b : b - a;
    return d <= tol;
}

static void test_split_roundtrip(ggml_backend_buffer_type_t split_buft, int split_dim, int64_t ne0, int64_t ne1) {
    GGML_ASSERT(split_dim >= -1 && split_dim <= 2);

    ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, "split");

    // Build the split tensor extra by hand (mirrors llama prepare_split_tensors).
    const int n_device = 2;
    static thread_local ggml_tensor * splits_buf[2];
    static thread_local ggml_split_tensor_t extra_buf;
    int64_t s0 = split_dim == 0 ? ne0 / 2 : ne0;
    int64_t s1 = split_dim == 1 ? ne1 / 2 : ne1;
    splits_buf[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, split_dim == 0 ? s0 : ne0, split_dim == 1 ? s1 : ne1);
    splits_buf[1] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, split_dim == 0 ? ne0 - s0 : ne0, split_dim == 1 ? ne1 - s1 : ne1);
    extra_buf.n_device = n_device;
    extra_buf.split_dim = split_dim;
    extra_buf.tensor = t;
    extra_buf.splits = splits_buf;
    t->extra = &extra_buf;

    // Allocate the parent tensor on the split buffer type (this calls init_tensor).
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(split_buft, ggml_backend_buft_get_alloc_size(split_buft, t));
    if (!buf) {
        fprintf(stderr, "FAIL split_dim=%d: buffer alloc failed\n", split_dim);
        n_failures++;
        ggml_free(ctx);
        return;
    }
    ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));

    const size_t n = ggml_nelements(t);
    std::vector<float> in(n);
    for (size_t i = 0; i < n; i++) in[i] = (float)(i + 1);
    ggml_backend_tensor_set(t, in.data(), 0, ggml_nbytes(t));

    // Read each split back and verify against the expected slice of `in`.
    std::vector<float> out(n);
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));

    bool ok = true;
    if (split_dim < 0) {
        for (size_t i = 0; i < n; i++) ok &= closef(out[i], in[i]);
    } else if (split_dim == 0) {
        for (int64_t i = 0; i < n; i++) ok &= closef(out[i], in[i]);
    } else if (split_dim == 1) {
        for (int64_t i = 0; i < n; i++) ok &= closef(out[i], in[i]);
    }

    if (!ok) {
        fprintf(stderr, "FAIL split_dim=%d ne0=%ld ne1=%ld\n", split_dim, (long)ne0, (long)ne1);
        for (int64_t i = 0; i < n && i < 16; i++) {
            fprintf(stderr, "  [%ld] in=%g out=%g\n", (long)i, in[i], out[i]);
        }
        n_failures++;
    } else {
        printf("OK   split_dim=%d ne0=%ld ne1=%ld\n", split_dim, (long)ne0, (long)ne1);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void test_split_row_meta(ggml_backend_buffer_type_t split_buft) {
    // Regression test: contiguous row split of a quant with a per-row scale header
    // (row_meta_size > 0). The block data must be offset past the row-meta header, not
    // just by (block_index * type_size).
    const ggml_type type = GGML_TYPE_IQ4_KS;
    auto tt = ggml_internal_get_type_traits(type);
    GGML_ASSERT(tt.row_meta_size > 0);

    const int64_t ne0 = tt.blck_size * 2; // 2 blocks per row
    const int64_t ne1 = 3;

    ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_name(t, "split_row_meta");

    static thread_local ggml_tensor * splits_buf[2];
    static thread_local ggml_split_tensor_t extra_buf;
    splits_buf[0] = ggml_new_tensor_2d(ctx, type, tt.blck_size, ne1);
    splits_buf[1] = ggml_new_tensor_2d(ctx, type, tt.blck_size, ne1);
    extra_buf.n_device = 2;
    extra_buf.split_dim = 0;
    extra_buf.tensor = t;
    extra_buf.splits = splits_buf;
    t->extra = &extra_buf;

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(split_buft, ggml_backend_buft_get_alloc_size(split_buft, t));
    if (!buf) {
        fprintf(stderr, "FAIL split row_meta: buffer alloc failed\n");
        n_failures++;
        ggml_free(ctx);
        return;
    }
    ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));

    const size_t row_size = ggml_row_size(type, ne0);
    std::vector<uint8_t> in(row_size * ne1);
    for (size_t i = 0; i < in.size(); i++) in[i] = (uint8_t)(i + 1);
    ggml_backend_tensor_set(t, in.data(), 0, ggml_nbytes(t));

    // Each split row is [row_meta_size header][one block of type_size bytes].
    bool ok = true;
    for (int id = 0; id < 2; id++) {
        ggml_tensor * split = extra_buf.splits[id];
        const size_t split_row_size = ggml_row_size(type, split->ne[0]);
        std::vector<uint8_t> got(ggml_nbytes(split));
        ggml_backend_tensor_get(split, got.data(), 0, got.size());
        for (int64_t i1 = 0; i1 < ne1; i1++) {
            const uint8_t * src_row = in.data() + i1 * row_size;
            const uint8_t * got_row = got.data() + i1 * split_row_size;
            if (memcmp(got_row, src_row, tt.row_meta_size) != 0) {
                ok = false;
                fprintf(stderr, "  row_meta dev=%d row=%ld: meta mismatch\n", id, (long)i1);
                break;
            }
            const size_t block_off = tt.row_meta_size + (size_t) id * tt.type_size;
            if (memcmp(got_row + tt.row_meta_size, src_row + block_off, tt.type_size) != 0) {
                ok = false;
                fprintf(stderr, "  row_meta dev=%d row=%ld: block data mismatch\n", id, (long)i1);
                break;
            }
        }
        if (!ok) break;
    }

    if (!ok) {
        fprintf(stderr, "FAIL split row_meta\n");
        n_failures++;
    } else {
        printf("OK   split row_meta (iq4_ks)\n");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void test_split_mixed(ggml_backend_buffer_type_t buft0, ggml_backend_buffer_type_t buft1) {
    // Directly exercises the backend-agnostic split buffer type with two different
    // backends in the per-slot map (here Vulkan0 + CPU), which is the mixed-device case.
    ggml_backend_buffer_type_t bufts[2] = { buft0, buft1 };
    ggml_backend_buffer_type_t split_buft = ggml_backend_split_buffer_type(bufts, 2);

    ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8);
    ggml_set_name(t, "split_mixed");

    static thread_local ggml_tensor * splits_buf[2];
    static thread_local ggml_split_tensor_t extra_buf;
    splits_buf[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 8);
    splits_buf[1] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 8);
    extra_buf.n_device = 2;
    extra_buf.split_dim = 0;
    extra_buf.tensor = t;
    extra_buf.splits = splits_buf;
    t->extra = &extra_buf;

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(split_buft, ggml_backend_buft_get_alloc_size(split_buft, t));
    if (!buf) {
        fprintf(stderr, "FAIL split mixed: buffer alloc failed\n");
        n_failures++;
        ggml_free(ctx);
        return;
    }
    ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));

    const size_t n = ggml_nelements(t);
    std::vector<float> in(n);
    for (size_t i = 0; i < n; i++) in[i] = (float)(i + 1);
    ggml_backend_tensor_set(t, in.data(), 0, ggml_nbytes(t));

    std::vector<float> out(n);
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));

    bool ok = true;
    for (size_t i = 0; i < n; i++) ok &= closef(out[i], in[i]);
    if (!ok) {
        fprintf(stderr, "FAIL split mixed (Vulkan0+CPU)\n");
        for (size_t i = 0; i < n && i < 16; i++) fprintf(stderr, "  [%zu] in=%g out=%g\n", i, in[i], out[i]);
        n_failures++;
    } else {
        printf("OK   split mixed (Vulkan0+CPU)\n");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void test_split_ranges(ggml_backend_buffer_type_t split_buft, int split_dim) {
    // Mirrors prepare_delta_split(): explicit (non-contiguous) per-device ranges stored
    // in tensor->op_params. This is the layout used by the gated delta-net weights under
    // -sm graph. We only verify the load side (set_tensor); get_tensor on the wrapper is
    // intentionally not implemented for the ranges form.
    const int64_t ne0 = split_dim == 0 ? 12 : 8;
    const int64_t ne1 = split_dim == 0 ? 4 : 12;

    ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, "split_ranges");

    static thread_local ggml_tensor * splits_buf[2];
    static thread_local ggml_split_tensor_t extra_buf;
    if (split_dim == 0) {
        splits_buf[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, ne1);
        splits_buf[1] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, ne1);
    } else {
        splits_buf[0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 8);
        splits_buf[1] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 4);
    }
    extra_buf.n_device = 2;
    extra_buf.split_dim = split_dim;
    extra_buf.tensor = t;
    extra_buf.splits = splits_buf;
    t->extra = &extra_buf;

    static thread_local std::vector<std::vector<std::pair<int, int>>> ranges;
    ranges.resize(2);
    if (split_dim == 0) {
        ranges[0] = { {0, 8} };
        ranges[1] = { {8, 4} };
    } else {
        ranges[0] = { {0, 4}, {8, 4} };
        ranges[1] = { {4, 4} };
    }
    auto * ranges_ptr = &ranges;
    memcpy(t->op_params, &ranges_ptr, sizeof(ranges_ptr));

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(split_buft, ggml_backend_buft_get_alloc_size(split_buft, t));
    if (!buf) {
        fprintf(stderr, "FAIL split ranges dim=%d: buffer alloc failed\n", split_dim);
        n_failures++;
        ggml_free(ctx);
        return;
    }
    ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));

    std::vector<float> in(ggml_nelements(t));
    for (size_t i = 0; i < in.size(); i++) in[i] = (float)(i + 1);
    ggml_backend_tensor_set(t, in.data(), 0, ggml_nbytes(t));

    // Read each split and verify it matches the expected contiguous slice of `in`.
    bool ok = true;
    for (int id = 0; id < 2; id++) {
        ggml_tensor * split = extra_buf.splits[id];
        std::vector<float> got(ggml_nelements(split));
        ggml_backend_tensor_get(split, got.data(), 0, ggml_nbytes(split));

        std::vector<float> want;
        want.reserve(ggml_nelements(split));
        if (split_dim == 0) {
            // Row split: element ranges within each row, rows are contiguous.
            for (int64_t i1 = 0; i1 < split->ne[1]; i1++) {
                for (auto & p : ranges[id]) {
                    for (int i = 0; i < p.second; i++) {
                        want.push_back(in[(p.first + i) + i1 * ne0]);
                    }
                }
            }
        } else {
            // Column split: whole columns, each column has ne0 elements.
            for (auto & p : ranges[id]) {
                for (int c = 0; c < p.second; c++) {
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        want.push_back(in[i0 + (p.first + c) * ne0]);
                    }
                }
            }
        }
        for (size_t i = 0; i < got.size(); i++) {
            if (!closef(got[i], want[i])) {
                ok = false;
                fprintf(stderr, "  dim=%d dev=%d [%zu] got=%g want=%g\n", split_dim, id, i, got[i], want[i]);
                break;
            }
        }
    }

    if (!ok) {
        fprintf(stderr, "FAIL split ranges dim=%d\n", split_dim);
        n_failures++;
    } else {
        printf("OK   split ranges dim=%d\n", split_dim);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void test_reduce_scheduler(ggml_backend_t backend0, ggml_backend_t backend1) {
    const int64_t K = 32, M = 16, N = 4;

    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[3] = { backend0, backend1, backend_cpu };
    ggml_backend_buffer_type_t bufts[3] = {
        ggml_backend_get_default_buffer_type(backend0),
        ggml_backend_get_default_buffer_type(backend1),
        ggml_backend_cpu_buffer_type(),
    };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 3, 4096, false);
    ggml_backend_sched_set_split_mode_graph(sched, true, false);

    ggml_init_params params = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx0 = ggml_init(params);
    ggml_context * ctx1 = ggml_init(params);
    ggml_context * ctx_x = ggml_init(params);
    ggml_context * ctx  = ggml_init(params);

    ggml_tensor * w0 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, K, M);
    ggml_set_name(w0, "w0");
    ggml_tensor * w1 = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, K, M);
    ggml_set_name(w1, "w1");

    ggml_backend_alloc_ctx_tensors(ctx0, backend0);
    ggml_backend_alloc_ctx_tensors(ctx1, backend1);
    ggml_backend_buffer_set_usage(w0->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(w1->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_tensor * x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, K, N);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_backend_alloc_ctx_tensors(ctx_x, backend_cpu);

    ggml_tensor * y0 = ggml_mul_mat(ctx, w0, x);
    ggml_set_name(y0, "y0");
    ggml_tensor * y1 = ggml_mul_mat(ctx, w1, x);
    ggml_set_name(y1, "y1");
    ggml_tensor * srcs[2] = { y0, y1 };
    ggml_tensor * r = ggml_reduce(ctx, srcs, 2, GGML_OP_ADD);
    ggml_set_name(r, "reduce");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);

    std::vector<float> w0_data(K * M), w1_data(K * M), x_data(K * N);
    for (int64_t m = 0; m < M; m++) {
        for (int64_t k = 0; k < K; k++) {
            w0_data[k + m * K] = (float)(k * M + m + 1);
            w1_data[k + m * K] = (float)(2 * (k * M + m) + 1);
        }
    }
    for (int64_t n = 0; n < N; n++) {
        for (int64_t k = 0; k < K; k++) {
            x_data[k + n * K] = (float)(k * N + n + 1);
        }
    }
    ggml_backend_tensor_set(w0, w0_data.data(), 0, ggml_nbytes(w0));
    ggml_backend_tensor_set(w1, w1_data.data(), 0, ggml_nbytes(w1));
    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "FAIL reduce scheduler: alloc_graph failed\n");
        n_failures++;
        ggml_backend_sched_free(sched);
        ggml_free(ctx0); ggml_free(ctx1); ggml_free(ctx);
        return;
    }
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL reduce scheduler: compute failed\n");
        n_failures++;
        ggml_backend_sched_free(sched);
        ggml_free(ctx0); ggml_free(ctx1); ggml_free(ctx);
        return;
    }

    // Reference: r = w0@x + w1@x on CPU.
    std::vector<float> ref(M * N, 0.0f);
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                s += (w0_data[k + m * K] + w1_data[k + m * K]) * x_data[k + n * K];
            }
            ref[m + n * M] = s;
        }
    }

    std::vector<float> out(M * N);
    ggml_backend_tensor_get(r, out.data(), 0, ggml_nbytes(r));
    {
        std::vector<float> y0o(M * N), y1o(M * N);
        ggml_backend_tensor_get(y0, y0o.data(), 0, ggml_nbytes(y0));
        ggml_backend_tensor_get(y1, y1o.data(), 0, ggml_nbytes(y1));
        float w0v, w1v, xv;
        ggml_backend_tensor_get(w0, &w0v, 0, sizeof(float));
        ggml_backend_tensor_get(w1, &w1v, 0, sizeof(float));
        ggml_backend_tensor_get(x, &xv, 0, sizeof(float));
        fprintf(stderr, "[debug] w0[0]=%g w1[0]=%g x[0]=%g y0[0]=%g y1[0]=%g r[0]=%g ref[0]=%g\n", w0v, w1v, xv, y0o[0], y1o[0], out[0], ref[0]);
    }
    bool ok = true;
    for (size_t i = 0; i < ref.size(); i++) ok &= closef(out[i], ref[i], 1e-3f);
    if (!ok) {
        fprintf(stderr, "FAIL reduce scheduler\n");
        for (size_t i = 0; i < ref.size(); i++) fprintf(stderr, "  [%zu] got=%g want=%g\n", i, out[i], ref[i]);
        n_failures++;
    } else {
        printf("OK   reduce scheduler (cross-device)\n");
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_cpu);
    ggml_free(ctx0); ggml_free(ctx1); ggml_free(ctx_x); ggml_free(ctx);
}

static void test_mul_mat_direct(ggml_backend_t backend) {
    const int64_t K = 32, M = 16, N = 4;
    ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_name(w, "w");
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(x, "x");
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> wd(K * M), xd(K * N);
    for (int64_t m = 0; m < M; m++) for (int64_t k = 0; k < K; k++) wd[k + m * K] = (float)(k * M + m + 1);
    for (int64_t n = 0; n < N; n++) for (int64_t k = 0; k < K; k++) xd[k + n * K] = (float)(k * N + n + 1);
    ggml_backend_tensor_set(w, wd.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, xd.data(), 0, ggml_nbytes(x));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL mul_mat direct: compute failed\n");
        n_failures++;
        ggml_free(ctx);
        return;
    }
    std::vector<float> yo(M * N);
    ggml_backend_tensor_get(y, yo.data(), 0, ggml_nbytes(y));
    float ref0 = 0.0f;
    for (int64_t k = 0; k < K; k++) ref0 += wd[k + 0 * K] * xd[k + 0 * K];
    fprintf(stderr, "[debug] mul_mat direct y[0]=%g ref[0]=%g\n", yo[0], ref0);
    if (!closef(yo[0], ref0, 1e-3f)) {
        fprintf(stderr, "FAIL mul_mat direct\n");
        n_failures++;
    } else {
        printf("OK   mul_mat direct\n");
    }
    ggml_free(ctx);
}

static void test_reduce_single_device(ggml_backend_t backend) {
    ggml_init_params params = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_set_name(b, "b");

    ggml_tensor * srcs[2] = { a, b };
    ggml_tensor * r = ggml_reduce(ctx, srcs, 2, GGML_OP_ADD);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    const size_t n = ggml_nelements(a);
    std::vector<float> va(n), vb(n);
    for (size_t i = 0; i < n; i++) { va[i] = (float)i; vb[i] = (float)(2*i); }
    ggml_backend_tensor_set(a, va.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(b, vb.data(), 0, n * sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL reduce single-device: compute failed\n");
        n_failures++;
        ggml_free(ctx);
        return;
    }

    std::vector<float> out(n);
    ggml_backend_tensor_get(r, out.data(), 0, n * sizeof(float));
    bool ok = true;
    for (size_t i = 0; i < n; i++) ok &= closef(out[i], va[i] + vb[i], 1e-4f);
    if (!ok) {
        fprintf(stderr, "FAIL reduce single-device\n");
        for (size_t i = 0; i < n && i < 16; i++) fprintf(stderr, "  [%zu] got=%g want=%g\n", i, out[i], va[i] + vb[i]);
        n_failures++;
    } else {
        printf("OK   reduce single-device\n");
    }
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();

    if (ggml_backend_vk_get_device_count() < 2) {
        fprintf(stderr, "need at least 2 Vulkan devices\n");
        return 1;
    }

    ggml_backend_buffer_type_t split_buft = ggml_backend_vk_split_buffer_type(nullptr);
    printf("split buffer type: %s\n", ggml_backend_buft_name(split_buft));

    test_split_roundtrip(split_buft, -1, 32, 8);
    test_split_roundtrip(split_buft, 0, 64, 8);
    test_split_roundtrip(split_buft, 1, 32, 8);
    test_split_row_meta(split_buft);
    test_split_ranges(split_buft, 0);
    test_split_ranges(split_buft, 1);

    ggml_backend_t backend0 = ggml_backend_reg_init_backend_from_str("Vulkan0");
    ggml_backend_t backend1 = ggml_backend_reg_init_backend_from_str("Vulkan1");
    if (backend0) {
        test_mul_mat_direct(backend0);
        test_reduce_single_device(backend0);
    } else {
        fprintf(stderr, "failed to init Vulkan0\n");
        n_failures++;
    }
    if (backend0 && backend1) {
        test_reduce_scheduler(backend0, backend1);
    } else {
        fprintf(stderr, "skipping cross-device reduce (need Vulkan0 and Vulkan1)\n");
    }
    if (backend0) {
        test_split_mixed(ggml_backend_get_default_buffer_type(backend0), ggml_backend_cpu_buffer_type());
    }
    if (backend0) ggml_backend_free(backend0);
    if (backend1) ggml_backend_free(backend1);

    printf("test-vk-split: %d failures\n", n_failures);
    return n_failures == 0 ? 0 : 1;
}
