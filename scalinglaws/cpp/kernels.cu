#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include "common.h"
#include <ATen/cuda/CUDAContext.h>

at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}

__global__ void solve_policy_kernel(
    TP2D::PTA pi, TP2D::PTA q, TP1D::PTA lambda_n,
    TP1D::PTA alpha_star) {

    const auto B = pi.size(0);
    const auto A = pi.size(1);
    const int b = blockIdx.x;

    const auto lambda = lambda_n[b];

    float alpha = 0.f;
    for (int a = 0; a < A; a++) {
        float gap = fmaxf(lambda*pi[b][a], 1.e-6f);
        alpha = fmaxf(alpha, q[b][a] + gap);
    }

    float error = CUDART_INF_F;
    float new_error = CUDART_INF_F;
    // Typical problems converge in 10 steps. Hypothetically 100 might be 
    // hit sometimes, but it's worth risking it for how utterly awful it'd 
    // be debugging an infinite loop in the kernel.
    for (int s=0; s<100; s++) {
        float S = 0.f; 
        float g = 0.f;
        for (int a=0; a<A; a++) {
            S += lambda*pi[b][a]/(alpha - q[b][a]);
            g += -lambda*pi[b][a]/powf(alpha - q[b][a], 2);
        }
        new_error = S - 1.f;
        if ((new_error < 1e-3f) || (error == new_error)) {
            alpha_star[b] = alpha;
            break;
        } else {
            alpha -= new_error/g;
            error = new_error;
        }
    }
}

__host__ TT solve_policy(const TT pi, const TT q, const TT lambda_n) {
    const uint B = pi.size(0);
    const uint A = pi.size(1);

    auto alpha_star(TP1D::empty({B}));
    alpha_star.t.fill_(NAN);

    solve_policy_kernel<<<{B}, {1}, 0, stream()>>>(
        TP2D(pi).pta(), TP2D(q).pta(), TP1D(lambda_n).pta(),
        alpha_star.pta());

    return alpha_star.t;
}