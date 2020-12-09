#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include "common.h"
#include <ATen/cuda/CUDAContext.h>

const uint BLOCK = 8;

at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}

__global__ void solve_policy_kernel(
    TP2D::PTA pi, TP2D::PTA q, TP1D::PTA lambda_n,
    TP1D::PTA alpha_star) {

    const auto B = pi.size(0);
    const auto A = pi.size(1);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }

    extern __shared__ float shared[];
    float *qb = (float*)&shared[threadIdx.x*2*A];
    float *pib = (float*)&shared[threadIdx.x*2*A+A];
    for (int a = 0; a < A; a++) {
        qb[a] = q[b][a];
        pib[a] = pi[b][a];
    }
    __syncthreads();

    const auto lambda = lambda_n[b];

    float alpha = 0.f;
    for (int a = 0; a < A; a++) {
        float gap = fmaxf(lambda*pib[a], 1.e-6f);
        alpha = fmaxf(alpha, qb[a] + gap);
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
            float top = lambda*pib[a];
            float bot = alpha - qb[a];
            S += top/bot;
            g += -top/powf(bot, 2);
        }
        new_error = S - 1.f;
        // printf("%d: alpha: %.2f, S: %.2f, e: %.2f, g: %.2f\n", b, alpha, S, new_error, g);
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

    //TODO: Replace this with a hardware dependent test
    assert (BLOCK*2*A*sizeof(float) < 64*1024);

    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    solve_policy_kernel<<<{n_blocks}, {BLOCK}, BLOCK*2*A*sizeof(float), stream()>>>(
        TP2D(pi).pta(), TP2D(q).pta(), TP1D(lambda_n).pta(),
        alpha_star.pta());

    return alpha_star.t;
}