#include "../../cpp/kernels.cu"
#include "common.h"

const uint BLOCK = 8;

struct Policy {
    int A;
    float* pi;
    float* q;
    float lambda_n;
    float alpha;

    __device__ Policy(int A) :
        A(A)
    {
        extern __shared__ float shared[];
        pi = (float*)&shared[(2*threadIdx.x+0)*A];
        q  = (float*)&shared[(2*threadIdx.x+1)*A];
    }

    __device__ float prob(int a) {
        return lambda_n*pi[a]/(alpha - q[a]);
    }
    
    __host__ static uint memory(uint A) {
        uint mem = BLOCK*2*A*sizeof(float);
        TORCH_CHECK(mem < 64*1024, "Too much shared memory per block")
        return mem;
    }

};

__device__ float newton_search(Policy p) {
    // Find the initial alpha
    float alpha = 0.f;
    for (int a = 0; a<p.A; a++) {
        float gap = fmaxf(p.lambda_n*p.pi[a], 1.e-4f);
        alpha = fmaxf(alpha, p.q[a] + gap);
    }

    float error = CUDART_INF_F;
    float new_error = CUDART_INF_F;
    // Typical problems converge in 10 steps. Hypothetically 100 might be 
    // hit sometimes, but it's worth risking it for how utterly awful it'd 
    // be debugging an infinite loop in the kernel.
    for (int t=0; t<100; t++) {
        float S = 0.f; 
        float g = 0.f;
        for (int a=0; a<p.A; a++) {
            float top = p.lambda_n*p.pi[a];
            float bot = alpha - p.q[a];
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

__device__ Policy policy(MCTSPTA m, H3D::PTA q, int t) {

    const uint A = m.logits.size(2);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    Policy p(A);

    int N = 0;
    auto seat = m.seats[b][t];

    for (int a=0; a<p.A; a++) {
        auto child = m.children[b][t][a];

        if (child > -1) {
            p.q[a] = q[b][child][seat];
            p.pi[a] = expf(m.logits[b][t][a]);
            N += m.n[b][child];
        } else {
            p.q[a] = 1.f;
            p.pi[a] = expf(m.logits[b][t][a]);
            N += 1;
        }
    }
    __syncthreads(); // memory barrier

    p.lambda_n = m.c_puct[b]*float(N)/float(N +A);
    p.alpha = newton_search(p);

    return p;
}

__host__ TT transition_q(MCTS m) {
    auto q = m.w.t/(m.n.t.unsqueeze(-1) + 1.e-4f);
    q = (q - q.min())/(q.max() - q.min() + 1.e-4f);
    return q.to(at::kHalf);
}

__global__ void root_kernel(MCTSPTA m, H3D::PTA q, H2D::PTA probs) {
    const uint B = m.logits.size(0);
    const uint A = m.logits.size(2);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= B) return;

    auto p = policy(m, q, 0);

    for (int a=0; a<A; a++) {
        probs[b][a] = p.prob(a);
    }
}

__host__ TT root(MCTS m) {
    c10::cuda::CUDAGuard g(m.logits.t.device());

    const uint B = m.logits.size(0);
    const uint A = m.logits.size(2);

    auto q = transition_q(m);

    auto probs = at::empty_like(m.logits.t.select(1, 0));

    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    root_kernel<<<{n_blocks}, {BLOCK}, Policy::memory(A), stream()>>>(
        m.pta(), H3D(q).pta(), H2D(probs).pta());
    C10_CUDA_CHECK(cudaGetLastError());

    return probs;
}

__global__ void descend_kernel(
    MCTSPTA m, H3D::PTA q, H2D::PTA rands, DescentPTA descent) {

    const uint B = m.logits.size(0);
    const uint A = m.logits.size(2);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    if (b >= B) return;

    int t = 0;
    int parent = 0;
    int action = -1;
    int valid = -1;
    while (true) {
        if (t == -1) break;
        if (m.terminal[b][t]) break;

        auto p = policy(m, q, t);

        float rand = rands[b][t];
        float total = 0.f;
        // This is a bit of a mess. Intent is to handle the edge 
        // case of rand being 1, and the probabilities not summing
        // to that. Then we need to fall back to a 'valid' value, 
        // ie one that has a positive probability.
        action = -1; 
        valid = -1;
        for (int a=0; a<A; a++) {
            float prob = p.prob(a);
            total += prob;
            if ((prob > 0) && (total >= rand)) {
                action = a;
                break;
            } else if (prob > 0) {
                valid = a;
            }
        }
        action = (action >= 0)? action : valid;
        parent = t;
        t = m.children[b][t][action];
    }

    descent.parents[b] = parent;
    descent.actions[b] = action;
}

__host__ Descent descend(MCTS m) {
    c10::cuda::CUDAGuard g(m.logits.t.device());

    const uint B = m.logits.size(0);
    const uint A = m.logits.size(2);

    auto q = transition_q(m);
    auto rands = at::rand_like(m.logits.t.select(2, 0));

    Descent descent{
        m.seats.t.new_empty({B}),
        m.seats.t.new_empty({B})};

    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    descend_kernel<<<{n_blocks}, {BLOCK}, Policy::memory(A), stream()>>>(
        m.pta(), H3D(q).pta(), H2D(rands).pta(), descent.pta());
    C10_CUDA_CHECK(cudaGetLastError());

    return descent;
}

__global__ void backup_kernel(BackupPTA bk, S1D::PTA leaves) {
    const uint B = bk.v.size(0);
    const uint S = bk.v.size(2);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    if (b >= B) return;

    extern __shared__ float shared[];
    float* v = (float*)&shared[threadIdx.x*S];

    int current = leaves[b];
    for (int s=0; s<S; s++) {
        v[s] = bk.v[b][current][s];
    }
    while (true) {
        if (current == -1) break;

        //TODO: Should invert this loop for memory locality buuuuut
        // it's not gonna be the bottleneck anyway. 
        for (int s=0; s<S; s++) {
            if (bk.terminal[b][current]) {
                v[s] = 0.f;
            }
            v[s] += bk.rewards[b][current][s];

            bk.n[b][current] += 1;
            bk.w[b][current][s] += v[s];
        }

        current = bk.parents[b][current]; 
    }
}

__host__ void backup(Backup b, TT leaves) {
    c10::cuda::CUDAGuard c(leaves.device());

    const uint B = b.v.size(0);
    const uint S = b.v.size(2);

    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    backup_kernel<<<{n_blocks}, {BLOCK}, BLOCK*S*sizeof(float), stream()>>>(
        b.pta(), S1D(leaves).pta());
    C10_CUDA_CHECK(cudaGetLastError());
}