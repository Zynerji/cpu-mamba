// selective_scan_cpu.cpp — Mamba S6 selective scan, CPU only.
//
// y[b, t, d] = sum_s C[b, t, s] * h[b, d, s] + D[d] * x[b, t, d]
// h[b, d, s] = exp(dt[b, t, d] * A[d, s]) * h_prev[b, d, s] + dt[b, t, d] * B[b, t, s] * x[b, t, d]
// h0 = 0
//
// Parallelized over (b, d) — each independent scan's state lives in d_state floats
// (small, fits in registers/L1). Inner timestep loop runs sequentially per (b, d).
//
// Build: torch.utils.cpp_extension.load(name='selective_scan_cpu',
//         sources=['selective_scan_cpu.cpp'],
//         extra_cflags=['-O3', '-fopenmp', '-march=native', '-ffast-math'],
//         extra_ldflags=['-fopenmp'])

#include <torch/extension.h>
#include <cmath>
#include <cstdint>

torch::Tensor selective_scan_cpu(
    torch::Tensor x,    // (B, T, d_inner) float32, contiguous
    torch::Tensor dt,   // (B, T, d_inner) float32, contiguous (softplus'd)
    torch::Tensor A,    // (d_inner, d_state) float32, contiguous (negative)
    torch::Tensor B,    // (B, T, d_state) float32, contiguous
    torch::Tensor C,    // (B, T, d_state) float32, contiguous
    torch::Tensor D     // (d_inner,) float32, contiguous
) {
    TORCH_CHECK(x.is_contiguous() && dt.is_contiguous() && A.is_contiguous()
                && B.is_contiguous() && C.is_contiguous() && D.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "fp32 only in this build");
    TORCH_CHECK(x.dim() == 3 && dt.dim() == 3 && A.dim() == 2 && B.dim() == 3
                && C.dim() == 3 && D.dim() == 1, "shape mismatch");

    const int64_t Bsz = x.size(0);
    const int64_t T = x.size(1);
    const int64_t d_inner = x.size(2);
    const int64_t d_state = A.size(1);

    TORCH_CHECK(d_state <= 256, "d_state > 256 not supported (stack limit)");

    auto y = torch::zeros_like(x);

    const float* __restrict__ x_p = x.data_ptr<float>();
    const float* __restrict__ dt_p = dt.data_ptr<float>();
    const float* __restrict__ A_p = A.data_ptr<float>();
    const float* __restrict__ B_p = B.data_ptr<float>();
    const float* __restrict__ C_p = C.data_ptr<float>();
    const float* __restrict__ D_p = D.data_ptr<float>();
    float* __restrict__ y_p = y.data_ptr<float>();

    // Strides (in elements):
    //   x[b,t,d] = b * (T*d_inner) + t * d_inner + d
    //   B[b,t,s] = b * (T*d_state) + t * d_state + s
    //   A[d,s]   = d * d_state + s

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < Bsz; ++b) {
        for (int64_t d = 0; d < d_inner; ++d) {
            float h[256];  // d_state-wide hidden state
            for (int64_t s = 0; s < d_state; ++s) h[s] = 0.0f;

            const float* A_row = A_p + d * d_state;
            const float D_val = D_p[d];

            const int64_t bt_dinner_base = b * T * d_inner + d;
            const int64_t bt_dstate_base = b * T * d_state;

            for (int64_t t = 0; t < T; ++t) {
                const float dt_v = dt_p[bt_dinner_base + t * d_inner];
                const float x_v = x_p[bt_dinner_base + t * d_inner];
                const float* B_row = B_p + bt_dstate_base + t * d_state;
                const float* C_row = C_p + bt_dstate_base + t * d_state;

                float y_v = 0.0f;
                #pragma GCC ivdep
                for (int64_t s = 0; s < d_state; ++s) {
                    const float dA = std::exp(dt_v * A_row[s]);
                    const float dBx = (dt_v * B_row[s]) * x_v;
                    h[s] = dA * h[s] + dBx;
                    y_v += h[s] * C_row[s];
                }
                y_p[bt_dinner_base + t * d_inner] = y_v + D_val * x_v;
            }
        }
    }
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selective_scan_cpu", &selective_scan_cpu,
          "Mamba selective scan on CPU (OpenMP parallel over (B, d_inner))");
}
