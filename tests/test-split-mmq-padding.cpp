// Regression test for the generic split buffer type (ggml_backend_split_buffer_type,
// the "GGML_SPLIT" weight buffer used by -sm graph / -sm attn in GGML_BACKEND_DL and
// mixed-backend builds).
//
// Backends may require quantized weight rows to be padded and zero-filled past the
// tensor's logical size: the CUDA buffer type's get_alloc_size() rounds quantized rows
// up to MATRIX_ROW_PADDING (512) elements because the MMQ kernels over-read each row
// up to that boundary. The generic split buffer allocated every slice with the bare
// ggml_nbytes(), so any slice whose ne0 % 512 != 0 (the common case for -sm graph
// K-splits such as blk.N.ssm_out.weight.X or blk.N.ffn_down.weight.X) was left
// unpadded: MMQ read past the end of the slice allocation -> "CUDA error: an illegal
// memory access was encountered" (when the tail happened to border an unmapped page,
// e.g. Qwen3.8-27B + --spec-type mtp on 3x CUDA -sm graph), and any garbage in the
// over-read tail could leak into the dot products.
//
// The same crash chain had a second memory-safety hole in the MMVQ (single-/few-token)
// path: k_mul_mat_vec_q and friends process rows in pairs (rows_per_cuda_block == 2
// for ncols_y >= 4) without guarding the phantom row of an odd-nrows matrix, reading
// one row past the end of the weight tensor. With a per-slice allocation under
// -sm graph this is again a real out-of-bounds access (observed on the 15-row
// blk.N.ssm_beta.weight.X slices of Qwen3.8-27B during MTP verify batches).
//
// This test mirrors quantized weights across N CUDA devices through the generic split
// buffer and runs mul_mat per device, checking:
//
//   1. (white-box) each slice's buffer honors the per-backend padded alloc size,
//   2. the results match a CPU reference,
//   3. the results match a plain per-device CUDA-buffer copy of the same weight.
//
// Case A: q6_0 [640 x 4096], batch 10  -> MMQ path (batch > MMVQ_MAX_BATCH_SIZE == 8)
// Case B: q8_0 [640 x 15],   batch 5   -> MMVQ path, odd row count (phantom row)
//
// Usage: test-split-mmq-padding [backend-prefix] [n_devices]
//   backend-prefix defaults to "CUDA", n_devices to 2 (or the number found).
// Standalone runs check the above; run under compute-sanitizer memcheck
// (tests/test-split-mmq-padding.sh) to also catch the out-of-bounds reads.
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct dev_info {
    ggml_backend_t             backend;
    ggml_backend_buffer_type_t buft;
};

// K is deliberately not a multiple of the CUDA MATRIX_ROW_PADDING (512), so the MMQ
// row over-read crosses the slice tail. Case A's row count makes each slice multi-MB,
// so slices get their own cudaMalloc mappings with gaps between them (like real model
// weights) — the over-read then lands in unmapped memory and compute-sanitizer memcheck
// flags it deterministically instead of silently reaching a neighbouring allocation.
// Case B's odd row count triggers the phantom-row read of the MMVQ kernels.
struct case_params {
    const char * label;
    ggml_type    type;
    int64_t      k;
    int64_t      n_rows;
    int64_t      batch;   // 10 -> MMQ, 5 -> MMVQ with rows_per_cuda_block == 2
};

static const case_params k_cases[] = {
    { "q6_0 [640x4096] batch 10 (MMQ)",  GGML_TYPE_Q6_0, 640, 4096, 10 },
    { "q8_0 [640x15]   batch 5  (MMVQ)", GGML_TYPE_Q8_0, 640,   15,  5 },
};

static bool find_devices(const char * prefix, int want, std::vector<dev_info> & out) {
    const size_t prefix_len = strlen(prefix);
    const size_t n = ggml_backend_reg_get_count();
    for (size_t i = 0; i < n && (int) out.size() < want; i++) {
        const char * name = ggml_backend_reg_get_name(i);
        if (strncmp(name, prefix, prefix_len) != 0) continue;
        if (!isdigit((unsigned char) name[prefix_len])) continue; // CUDA0, CUDA1, ...
        ggml_backend_t backend = ggml_backend_reg_init_backend(i, nullptr);
        if (!backend) continue;
        ggml_backend_buffer_type_t buft = ggml_backend_reg_get_default_buffer_type(i);
        if (!buft) continue;
        out.push_back({backend, buft});
    }
    return !out.empty();
}

static std::vector<uint8_t> make_weights(ggml_type type, int64_t k, int64_t n_rows) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> f((size_t) k * n_rows);
    for (auto & v : f) v = dist(rng);

    std::vector<uint8_t> q(n_rows * ggml_row_size(type, k));
    std::vector<float> imatrix(k, 1.0f);
    struct quantize_user_data qdata = { false, false };
    const size_t written = ggml_quantize_chunk(type, f.data(), q.data(), 0, n_rows, k,
            imatrix.data(), &qdata);
    GGML_ASSERT(written == q.size());
    return q;
}

static std::vector<float> make_input(int64_t k, int64_t batch) {
    std::mt19937 rng(4321);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x((size_t) k * batch);
    for (auto & v : x) v = dist(rng);
    return x;
}

// Allocates x and out on `buft` (plus a plain W when W_prealloc is null) and computes
// out = W @ x. W_prealloc is a tensor that is already allocated on `buft` (e.g. a
// split-buffer slice).
static std::vector<float> run_mul_mat(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
        ggml_tensor * W_prealloc, const std::vector<uint8_t> & w_bytes, ggml_type type,
        int64_t k, int64_t n_rows, int64_t batch, const std::vector<float> & x_host) {
    ggml_init_params ip = { 1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    GGML_ASSERT(ctx);

    ggml_tensor * W = W_prealloc;
    if (!W) {
        W = ggml_new_tensor_2d(ctx, type, k, n_rows);
        ggml_set_name(W, "W_plain");
    }

    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, batch);
    ggml_tensor * out = ggml_mul_mat(ctx, W, x);

    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft) != nullptr);

    if (!W_prealloc) {
        // plain per-device weight, same bytes (fresh allocations are zero-filled by the
        // driver, which covers the row padding tail)
        ggml_backend_tensor_set(W, w_bytes.data(), 0, ggml_nbytes(W));
    }
    ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> result((size_t) n_rows * batch);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_free(ctx);
    return result;
}

static bool check_close(const std::vector<float> & a, const std::vector<float> & b,
        const char * what, float atol, float rtol) {
    double max_abs = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double diff = std::fabs((double) a[i] - (double) b[i]);
        const double tol  = atol + rtol * std::fabs((double) b[i]);
        if (diff > tol) {
            fprintf(stderr, "test-split-mmq-padding: %s mismatch at %zu: %g vs %g (diff %g > tol %g)\n",
                    what, i, (double) a[i], (double) b[i], diff, tol);
            return false;
        }
        max_abs = std::max(max_abs, diff);
    }
    printf("  %-28s max_abs_diff = %.3g\n", what, max_abs);
    return true;
}

static bool run_case(const std::vector<dev_info> & devs, const case_params & cp) {
    printf("case: %s\n", cp.label);

    const std::vector<uint8_t> w_bytes = make_weights(cp.type, cp.k, cp.n_rows);
    const std::vector<float>   x_host  = make_input(cp.k, cp.batch);

    // 1) mirror W across the devices through the generic split buffer
    ggml_init_params ip = { 1024*1024, nullptr, true };
    ggml_context * ctx_w = ggml_init(ip);
    GGML_ASSERT(ctx_w);

    std::vector<ggml_tensor *> splits(devs.size(), nullptr);
    ggml_tensor * W = ggml_new_tensor_2d(ctx_w, cp.type, cp.k, cp.n_rows);
    ggml_set_name(W, "W");
    for (size_t d = 0; d < devs.size(); d++) {
        splits[d] = ggml_new_tensor_2d(ctx_w, cp.type, cp.k, cp.n_rows);
        ggml_format_name(splits[d], "W.%zu", d);
    }
    ggml_split_tensor_t split_extra = {};
    split_extra.n_device  = (int) devs.size();
    split_extra.split_dim = -1; // mirrored
    split_extra.tensor    = W;
    split_extra.splits    = splits.data();
    W->extra = &split_extra;

    std::vector<ggml_backend_buffer_type_t> bufts;
    for (auto & d : devs) bufts.push_back(d.buft);
    ggml_backend_buffer_type_t split_buft = ggml_backend_split_buffer_type(bufts.data(), bufts.size());
    GGML_ASSERT(split_buft != nullptr);

    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, split_buft) != nullptr);

    // White-box: each slice must honor the per-backend padded allocation size and carry
    // a zeroed tail (the padding bug allocated exactly ggml_nbytes()). Recorded as a
    // failure but does not abort, so the compute phase (and the compute-sanitizer run)
    // still exercises the kernels on the mis-sized allocation.
    const size_t nbytes = ggml_nbytes(W);
    bool ok = true;
    for (size_t d = 0; d < devs.size(); d++) {
        const size_t want_padded = ggml_backend_buft_get_alloc_size(devs[d].buft, splits[d]);
        const size_t have        = ggml_backend_buffer_get_size(splits[d]->buffer);
        const char * buft_name   = ggml_backend_buft_name(devs[d].buft);
        printf("  slice %zu (%s): logical %zu, padded %zu, allocated %zu\n",
                d, buft_name, nbytes, want_padded, have);
        if (have != want_padded) {
            fprintf(stderr, "test-split-mmq-padding: slice %zu allocated %zu bytes, but %s requires %zu "
                    "(quantized row padding ignored)\n", d, have, buft_name, want_padded);
            ok = false;
        }
    }

    ggml_backend_tensor_set(W, w_bytes.data(), 0, nbytes);

    // 2) CPU reference
    std::vector<float> ref;
    {
        ggml_context * ctx = ggml_init({ 1024*1024, nullptr, true });
        GGML_ASSERT(ctx);
        ggml_tensor * Wc = ggml_new_tensor_2d(ctx, cp.type, cp.k, cp.n_rows);
        ggml_tensor * xc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cp.k, cp.batch);
        ggml_tensor * oc = ggml_mul_mat(ctx, Wc, xc);
        ggml_backend_t cpu = ggml_backend_cpu_init();
        GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type()) != nullptr);
        ggml_backend_tensor_set(Wc, w_bytes.data(), 0, nbytes);
        ggml_backend_tensor_set(xc, x_host.data(), 0, x_host.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, oc);
        GGML_ASSERT(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu);
        ref.resize((size_t) cp.n_rows * cp.batch);
        ggml_backend_tensor_get(oc, ref.data(), 0, ref.size() * sizeof(float));
        ggml_free(ctx);
    }

    // 3) compute on each device and compare
    for (size_t d = 0; d < devs.size(); d++) {
        printf("  device %zu (%s)\n", d, ggml_backend_name(devs[d].backend));

        const std::vector<float> r_split = run_mul_mat(devs[d].backend, devs[d].buft,
                splits[d], w_bytes, cp.type, cp.k, cp.n_rows, cp.batch, x_host);

        // control: plain per-device weight buffer with the same bytes
        const std::vector<float> r_ctrl = run_mul_mat(devs[d].backend, devs[d].buft,
                nullptr, w_bytes, cp.type, cp.k, cp.n_rows, cp.batch, x_host);

        // loose vs the CPU reference: the CPU path quantizes the input to the weight's
        // vec_dot type while CUDA quantizes to q8_1, so a few percent of input-
        // quantization noise is expected; the tight regression check is the comparison
        // against the plain per-device weight below
        ok = check_close(r_split, ref,   "split vs cpu",   5e-2f, 5e-2f) && ok;
        ok = check_close(r_split, r_ctrl, "split vs plain", 1e-4f, 1e-4f) && ok;
    }

    ggml_free(ctx_w);
    return ok;
}

int main(int argc, char ** argv) {
    const char * prefix = argc > 1 ? argv[1] : "CUDA";
    int want = argc > 2 ? atoi(argv[2]) : 2;

    // deterministic kernel launches (this is about the eager MMQ/MMVQ dispatch)
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);

    ggml_backend_load_all();

    std::vector<dev_info> devs;
    if (!find_devices(prefix, want, devs)) {
        printf("SKIP: no '%s<id>' backends found\n", prefix);
        return 0;
    }
    printf("devices: %zu\n", devs.size());

    bool ok = true;
    for (const auto & cp : k_cases) {
        ok = run_case(devs, cp) && ok;
    }

    if (!ok) {
        fprintf(stderr, "test-split-mmq-padding: FAIL\n");
        return 1;
    }
    printf("test-split-mmq-padding: PASS\n");
    return 0;
}
