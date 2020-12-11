#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include "common.h"
#include <ATen/cuda/CUDAContext.h>

const uint BLOCK = 8;

using F1D = TensorProxy<float, 1>;
using F2D = TensorProxy<float, 2>;
using F3D = TensorProxy<float, 3>;
using I1D = TensorProxy<int, 1>;
using I2D = TensorProxy<int, 2>;
using I3D = TensorProxy<int, 3>;
using B1D = TensorProxy<bool, 1>;
using B2D = TensorProxy<bool, 2>;

at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}

__global__ void solve_policy_kernel(
    F2D::PTA pi, F2D::PTA q, F1D::PTA lambda_n,
    F1D::PTA alpha_star) {

    const auto B = pi.size(0);
    const auto A = pi.size(1);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }

    // Copy data into shared memory
    extern __shared__ float shared[];
    float *qb = (float*)&shared[threadIdx.x*2*A];
    float *pib = (float*)&shared[threadIdx.x*2*A+A];
    for (int a = 0; a < A; a++) {
        qb[a] = q[b][a];
        pib[a] = pi[b][a];
    }
    __syncthreads();

    const auto lambda = lambda_n[b];

    // Find the initial alpha
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

    F1D alpha_star(pi.new_empty({B}));
    alpha_star.t.fill_(NAN);

    //TODO: Replace this with a hardware dependent test
    assert (BLOCK*2*A*sizeof(float) < 64*1024);

    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    solve_policy_kernel<<<{n_blocks}, {BLOCK}, BLOCK*2*A*sizeof(float), stream()>>>(
        F2D(pi).pta(), F2D(q).pta(), F1D(lambda_n).pta(),
        alpha_star.pta());

    return alpha_star.t;
}

__device__ float solve(float* pi, float* q, int A, float lambda_n) {
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
    F3D::PTA pi, F3D::PTA q, I2D::PTA n, F1D::PTA c_puct,
    I2D::PTA seats, B2D::PTA terminal, I3D::PTA children,
    F2D::PTA rands, I1D::PTA parents, I1D::PTA actions) {

    const uint B = pi.size(0);
    const uint A = pi.size(2);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    if (b >= B) return;

    extern __shared__ float shared[];
    float *pis = (float*)&shared[threadIdx.x*2*A];
    float *qs = (float*)&shared[threadIdx.x*2*A+A];

    float c = c_puct[b];

    int t = 0;
    int parent = 0;
    int action = -1;
    while (true) {
        if (terminal[b][t]) break;
        if (t == -1) break;

        int N = 0;
        auto seat = seats[b][t];
        for (int a=0; a<A; a++) {
            auto child = children[b][t][a];
            if (child > -1) {
                qs[a] = q[b][child][seat];
                pis[a] = pi[b][t][a];
                N += n[b][child];
            } else {
                qs[a] = 0.f;
                pis[a] = pi[b][t][a];
                N += 1;
            }
        }
        __syncthreads(); // memory barrier

        float lambda_n = c*float(N)/float(N +A);
        float alpha = solve(pis, qs, A, lambda_n);

        float rand = rands[b][t];
        float total = 0.f;
        action = A-1; // handles numerical imprecision
        for (int a=0; a<A; a++) {
            total += lambda_n*pis[a]/(alpha - qs[a]);
            if (total >= rand) {
                action = a;
                break;
            }
        }
        parent = t;
        t = children[b][t][action];
    }

    parents[b] = parent;
    actions[b] = action;
}

__host__ DescentResult descend(
    const TT logits, const TT w, const TT n, const TT c_puct,
    const TT seats, const TT terminal, const TT children) {

    const uint B = logits.size(0);
    const uint A = logits.size(2);

    auto q = w/(n.unsqueeze(-1) + 1e-6);
    q = (q - q.min())/(q.max() - q.min() + 1e-6);

    auto pi = logits.exp();

    auto rands = at::rand_like(logits.select(2, 0));

    auto parents = seats.new_empty({B});
    auto actions = seats.new_empty({B});
    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    descend_kernel<<<{n_blocks}, {BLOCK}, BLOCK*2*A*sizeof(float), stream()>>>(
        F3D(pi).pta(), F3D(q).pta(), I2D(n).pta(), F1D(c_puct).pta(),
        I2D(seats).pta(), B2D(terminal).pta(), I3D(children).pta(), 
        F2D(rands).pta(), I1D(parents).pta(), I1D(actions).pta());

    return {parents, actions};
}