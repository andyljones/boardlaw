#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include "common.h"
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define TILESZ 32

namespace cg = cooperative_groups;

at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}

__device__ float reduce_sum(cg::thread_block_tile<32> tile, cg::coalesced_group strided, float x) {
    float tilesum = cg::reduce(tile, x, cg::plus<float>()); 
    float blocksum = cg::reduce(strided, tilesum, cg::plus<float>());
    return blocksum;
}

__global__ void solve_policy_kernel(
    TP2D::PTA pi, TP2D::PTA q, TP1D::PTA lambda_n,
    TP1D::PTA alpha_star) {

    const auto B = pi.size(0);
    const auto A = pi.size(1);
    const int b = blockIdx.x;

    extern __shared__ float shared[];
    float *qb = (float*)&shared[0];
    float *pib = (float*)&shared[A];

    auto block = cg::this_thread_block();
    auto a = block.thread_rank();

    auto tile = cg::tiled_partition<32>(block);
    auto strided = cg::labeled_partition(block, tile.thread_rank());

    if (a < A) {
        qb[a] = q[b][a];
        pib[a] = pi[b][a];
    }
    block.sync();

    const auto lambda = lambda_n[b];

    float alpha_a = -CUDART_INF_F;
    if (a < A) {
        float gap_a = fmaxf(lambda*pib[a], 1.e-6f);
        float alpha_a = qb[a] + gap_a;
    }  
    float alpha = cg::reduce(tile, alpha_a, cg::greater<float>());
    block.sync();
    float alpha = cg::reduce(strided, alpha, cg::greater<float>());
    block.sync();

    float error = CUDART_INF_F;
    float new_error = CUDART_INF_F;
    // Typical problems converge in 10 steps. Hypothetically 100 might be 
    // hit sometimes, but it's worth risking it for how utterly awful it'd 
    // be debugging an infinite loop in the kernel.
    for (int s=0; s<100; s++) {
        float S_a = 0.;
        float g_a = 0.;
        if (a < A) {
            float top = lambda*pib[a];
            float bot = alpha - qb[a];
            S_a = top/bot;
            g_a = -top/powf(bot, 2);
        }

        float S = cg::reduce(tile, S_a, cg::plus<float>());
        block.sync();
        float S = cg::reduce(tile, S_a, cg::plus<float>());
        float g = cg::reduce(group, g_a, cg::plus<float>());
        group.sync();

        new_error = S - 1.f;
        // printf("%d: alpha: %.2f, S: %.2f, e: %.2f, g: %.2f\n", b, alpha, S, new_error, g);
        if ((new_error < 1e-3f) || (error == new_error)) {
            break;
        } else {
            alpha -= new_error/g;
            error = new_error;
        }
    }

    if (a == 0) {
        alpha_star[b] = alpha;
    }

}

__host__ TT solve_policy(const TT pi, const TT q, const TT lambda_n) {
    const uint B = pi.size(0);
    const uint A = pi.size(1);

    TP1D alpha_star(pi.new_empty({B}));
    alpha_star.t.fill_(NAN);

    const uint n_threads = TILESZ*((A + TILESZ - 1)/TILESZ);
    solve_policy_kernel<<<{B}, {n_threads}, 2*A*sizeof(float), stream()>>>(
        TP2D(pi).pta(), TP2D(q).pta(), TP1D(lambda_n).pta(),
        alpha_star.pta());

    return alpha_star.t;
}