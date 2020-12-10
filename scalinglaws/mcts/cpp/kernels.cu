#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include "common.h"
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_reduce.cuh>  
#include <cub/block/block_load.cuh>  

#define BLOCK_THREADS 128

namespace cg = cooperative_groups;

at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}

__global__ void solve_policy_kernel(
    TP2D::PTA pi, TP2D::PTA q, TP1D::PTA lambda_n,
    TP1D::PTA alpha_star) {

    const auto B = pi.size(0);
    const auto A = pi.size(1);
    const int b = blockIdx.x;
    const int a = threadIdx.x;

    typedef cub::BlockLoad<float, BLOCK_THREADS, 1> BlockLoad;
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;

    __shared__ typename BlockLoad::TempStorage tempload;
    __shared__ typename BlockReduce::TempStorage tempreduce;
    __shared__ float alpha[1];
    __shared__ float S[1];
    __shared__ float g[1];

    float qb[1];
    float pib[1];

    BlockLoad(tempload).Load(q[b].data(), qb, A, 0.f);
    __syncthreads();
    BlockLoad(tempload).Load(pi[b].data(), pib, A, 0.f);
    __syncthreads();

    const auto lambda = lambda_n[b];

    float gap_a = fmaxf(lambda*pib[0], 1.e-6f);
    float alpha_a = BlockReduce(tempreduce).Reduce(qb[0] + gap_a, cub::Max(), A);
    if (a == 0) {
        alpha[0] = alpha_a;
    }
    __syncthreads();

    float error = CUDART_INF_F;
    float new_error = CUDART_INF_F;
    // Typical problems converge in 10 steps. Hypothetically 100 might be 
    // hit sometimes, but it's worth risking it for how utterly awful it'd 
    // be debugging an infinite loop in the kernel.
    for (int s=0; s<100; s++) {
        float top = lambda*pib[0];
        float bot = alpha[0] - qb[0];

        float S_a = BlockReduce(tempreduce).Sum(top/bot, A);
        __syncthreads();
        float g_a = BlockReduce(tempreduce).Sum(-top/powf(bot, 2), A);
        __syncthreads();

        if (a == 0) {
            S[0] = S_a;
            g[0] = g_a;
        }
        __syncthreads();

        new_error = S[0] - 1.f;
        // printf("%d: alpha: %.2f, S: %.2f, e: %.2f, g: %.2f\n", b, alpha, S, new_error, g);
        if ((new_error < 1e-3f) || (error == new_error)) {
            break;
        } else {
            alpha[0] -= new_error/g[0];
            error = new_error;
        }
    }

    if (a == 0) {
        alpha_star[b] = alpha[0];
    }

}

__host__ TT solve_policy(const TT pi, const TT q, const TT lambda_n) {
    const uint B = pi.size(0);
    const uint A = pi.size(1);

    TP1D alpha_star(pi.new_empty({B}));
    alpha_star.t.fill_(NAN);

    assert (A < BLOCK_THREADS);
    solve_policy_kernel<<<{B}, {BLOCK_THREADS}, 0, stream()>>>(
        TP2D(pi).pta(), TP2D(q).pta(), TP1D(lambda_n).pta(),
        alpha_star.pta());

    return alpha_star.t;
}