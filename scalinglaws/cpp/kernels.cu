#include <math_constants.h>
#include <ATen/ATen.h>
#include "common.h"
#include <ATen/cuda/CUDAContext.h>

at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}

__global__ void solve_policy_kernel(
    TP2D::PTA pi, TP2D::PTA q, TP1D::PTA lambda_n,
    TP2D::PTA policy, TP1D::PTA alpha_star) {

    const auto B = pi.size(0);
    const auto A = pi.size(1);
    const int b = blockIdx.x;

    const auto lambda = lambda_n[b];

    float alpha = 0.f;
    for (int a = 0; a < A; a++) {
        float gap = fmaxf(lambda*pi[b][a], 1.e-6f);
        alpha = fmaxf(alpha, q[b][a]);
    }

    float error = 1.f;
    float new_error = 1.f;
    for (int s=0; s<100; s++) {
        float S = 0.f; 
        float g = 0.f;
        for (int a=0; a<A; a++) {
            S += lambda*pi[b][a]/(alpha - q[b][a]);
            g += -lambda*pi[b][a]/powf(alpha - q[b][a], 2);
        }
        new_error = S - 1;
        if ((new_error < 1e-3f) || (error == new_error)) {
            break;
        } else {
            alpha -= new_error/g;
            error = new_error;
        }
    }
}

__host__ Solution solve_policy(const TT pi, const TT q, const TT lambda_n) {
    const uint B = pi.size(0);
    const uint A = pi.size(1);

    Solution soln(B, A);

    solve_policy_kernel<<<{B}, {1}, 0, stream()>>>(
        TP2D(pi).pta(), TP2D(q).pta(), TP1D(lambda_n).pta(),
        TP2D(soln.policy).pta(), TP1D(soln.alpha_star).pta());

    return soln;
}