// FFN-shaped matmul timing for IQK types: [k, m] x [k, n].
// Usage: test-vec-time <backend> <type> <k> <m> [n]
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    const char * tgt_name = argc > 1 ? argv[1] : "Vulkan0";
    const char * type_name = argc > 2 ? argv[2] : "iq4_kt";
    const int64_t k = argc > 3 ? atoll(argv[3]) : 152064;
    const int64_t m = argc > 4 ? atoll(argv[4]) : 5120;
    const int64_t n = argc > 5 ? atoll(argv[5]) : 512;

    ggml_backend_load_all();
    ggml_backend_t backend_tgt = nullptr;
    if (strcmp(tgt_name, "CPU") == 0) backend_tgt = ggml_backend_cpu_init();
    else backend_tgt = ggml_backend_reg_init_backend_from_str(tgt_name);
    if (!backend_tgt) { fprintf(stderr, "no backend\n"); return 1; }

    ggml_type type = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * tn = ggml_type_name((ggml_type)t);
        if (tn && strcmp(tn, type_name) == 0) { type = (ggml_type)t; break; }
    }
    if (type == GGML_TYPE_COUNT) { fprintf(stderr, "bad type\n"); return 1; }

    ggml_init_params params = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_backend_alloc_ctx_tensors(ctx, backend_tgt);

    fprintf(stderr, "quantizing...\n");
    {
        std::vector<float> f32(m * k);
        for (size_t i = 0; i < f32.size(); ++i) f32[i] = (float)((i * 7) % 13) * 0.05f;
        std::vector<uint8_t> dataq(m * ggml_row_size(type, k));
        std::vector<float> imatrix(k, 1.0f);
        struct quantize_user_data qdata = { false, false };
        ggml_quantize_chunk(type, f32.data(), dataq.data(), 0, m, k, imatrix.data(), &qdata);
        ggml_backend_tensor_set(a, dataq.data(), 0, dataq.size());
    }
    std::vector<float> act(k * n, 0.05f);
    ggml_backend_tensor_set(b, act.data(), 0, act.size() * sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    for (int i = 0; i < 20; ++i) ggml_backend_graph_compute(backend_tgt, gf);  // warmup (shader compile + clocks)
    int reps = 50;
    auto t0 = ggml_time_us();
    for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(backend_tgt, gf);
    double ms = (double)(ggml_time_us() - t0) / 1000.0 / reps;
    {
        double bytes = (double)ggml_nbytes(a) + (double)ggml_nbytes(b);
        printf("TIME %s [%lld,%lld]x[%lld,%lld]: %.2f ms (%.0f GB/s)\n", ggml_type_name(type),
               (long long)k, (long long)m, (long long)k, (long long)n, ms, bytes / (ms * 1e-3) / 1e9);
    }
    return 0;
}
