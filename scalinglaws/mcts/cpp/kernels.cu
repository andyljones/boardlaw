#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include "common.h"
#include <ATen/cuda/CUDAContext.h>

const uint BLOCK = 8;


at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}

__device__ float newton_search(float* pi, float* q, int A, float lambda_n) {
    // Find the initial alpha
    float alpha = 0.f;
    for (int a = 0; a < A; a++) {
        float gap = fmaxf(lambda_n*pi[a], 1.e-6f);
        alpha = fmaxf(alpha, q[a] + gap);
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
            float top = lambda_n*pi[a];
            float bot = alpha - q[a];
            S += top/bot;
            g += -top/powf(bot, 2);
        }
        new_error = S - 1.f;
        // printf("%d: alpha: %.2f, S: %.2f, e: %.2f, g: %.2f\n", b, alpha, S, new_error, g);
        if ((new_error < 1e-3f) || (error == new_error)) {
            break;
        } else {
            alpha -= new_error/g;
            error = new_error;
        }
    }

    return alpha;
}

__global__ void descend_kernel(
    MCTSPTA s, F3D::PTA pi, F3D::PTA q, F2D::PTA rands, DescentPTA descent) {

    const uint B = pi.size(0);
    const uint A = pi.size(2);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    if (b >= B) return;

    extern __shared__ float shared[];
    float *pis = (float*)&shared[threadIdx.x*2*A];
    float *qs = (float*)&shared[threadIdx.x*2*A+A];

    float c = s.c_puct[b];

    int t = 0;
    int parent = 0;
    int action = -1;
    int valid = -1;
    while (true) {
        if (s.terminal[b][t]) break;
        if (t == -1) break;

        int N = 0;
        auto seat = s.seats[b][t];
        for (int a=0; a<A; a++) {
            auto child = s.children[b][t][a];
            if (child > -1) {
                qs[a] = q[b][child][seat];
                pis[a] = pi[b][t][a];
                N += s.n[b][child];
            } else {
                qs[a] = 0.f;
                pis[a] = pi[b][t][a];
                N += 1;
            }
        }
        __syncthreads(); // memory barrier

        float lambda_n = c*float(N)/float(N +A);
        float alpha = newton_search(pis, qs, A, lambda_n);

        float rand = rands[b][t];
        float total = 0.f;
        // This is a bit of a mess. Intent is to handle the edge 
        // case of rand being 1, and the probabilities not summing
        // to that. Then we need to fall back to a 'valid' value, 
        // ie one that has a positive probability.
        action = -1; 
        valid = -1;
        for (int a=0; a<A; a++) {
            float prob = lambda_n*pis[a]/(alpha - qs[a]);
            total += prob;
            if ((prob > 0) && (total >= rand)) {
                action = a;
                break;
            } else if (prob > 0) {
                valid = a;
            }
        }
        parent = t;
        t = s.children[b][t][action];
    }

    descent.parents[b] = parent;
    descent.actions[b] = (action >= 0)? action : valid;
}

__host__ Descent descend(MCTS s) {
    const uint B = s.logits.size(0);
    const uint A = s.logits.size(2);

    auto q = s.w.t/(s.n.t.unsqueeze(-1) + 1e-6);
    q = (q - q.min())/(q.max() - q.min() + 1e-6);

    auto pi = s.logits.t.exp();

    auto rands = at::rand_like(s.logits.t.select(2, 0));

    Descent descent{
        s.seats.t.new_empty({B}),
        s.seats.t.new_empty({B})};
    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    descend_kernel<<<{n_blocks}, {BLOCK}, BLOCK*2*A*sizeof(float), stream()>>>(
        s.pta(), F3D(pi).pta(), F3D(q).pta(), F2D(rands).pta(), descent.pta());

    return descent;
}