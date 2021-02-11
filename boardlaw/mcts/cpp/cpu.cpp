#include "common.h"
#include <math.h>

namespace mctscpu {
const uint BLOCK = 8;

struct Policy {
    int A;
    float* pi;
    float* q;
    float lambda_n;
    float alpha;

    Policy(int A):
        A(A)
    {
        pi = (float*)malloc(A*sizeof(float));
        q  = (float*)malloc(A*sizeof(float));
    }

    ~Policy() {
        // free(pi);
        // free(q);
    }

    float prob(int a) {
        return lambda_n*pi[a]/(alpha - q[a]);
    }
    
    static uint memory(uint A) {
        uint mem = BLOCK*2*A*sizeof(float);
        TORCH_CHECK(mem < 64*1024, "Too much shared memory per block")
        return mem;
    }

};

float newton_search(Policy p) {
    // Find the initial alpha
    float alpha = 0.f;
    for (int a = 0; a<p.A; a++) {
        float gap = fmaxf(p.lambda_n*p.pi[a], 1.e-4f);
        alpha = fmaxf(alpha, p.q[a] + gap);
    }

    float error = INFINITY;
    float new_error = INFINITY;
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

Policy policy(MCTSTA m, H3D::TA q, int t, int b) {

    const uint A = m.logits.size(2);

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
            p.q[a] = 0.f;
            p.pi[a] = expf(m.logits[b][t][a]);
            N += 1;
        }
    }

    p.lambda_n = m.c_puct[b]*float(N)/float(N +A);
    p.alpha = newton_search(p);

    return p;
}

TT transition_q(MCTS m) {
    auto q = m.w.t/(m.n.t.unsqueeze(-1) + 1.e-4f);
    q = (q - q.min())/(q.max() - q.min() + 1.e-4f);
    return q.to(at::kHalf);
}

void root_kernel(MCTSTA m, H3D::TA q, H2D::TA probs, int b) {
    const uint B = m.logits.size(0);
    const uint A = m.logits.size(2);
    if (b >= B) return;

    auto p = policy(m, q, 0, b);

    for (int a=0; a<A; a++) {
        probs[b][a] = p.prob(a);
    }
}

TT root(MCTS m) {
    const uint B = m.logits.size(0);
    const uint A = m.logits.size(2);

    auto q = transition_q(m);

    auto probs = at::empty_like(m.logits.t.select(1, 0));

    for (int b=0; b<B; b++) {
        root_kernel(m.ta(), H3D(q).ta(), H2D(probs).ta(), b);
    } 

    return probs;
}

void descend_kernel(
    MCTSTA m, H3D::TA q, H2D::TA rands, DescentTA descent, int b) {

    const uint B = m.logits.size(0);
    const uint A = m.logits.size(2);

    if (b >= B) return;

    int t = 0;
    int parent = 0;
    int action = -1;
    int valid = -1;
    while (true) {
        if (t == -1) break;
        if (m.terminal[b][t]) break;

        auto p = policy(m, q, t, b);

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

Descent descend(MCTS m) {
    const uint B = m.logits.size(0);
    const uint A = m.logits.size(2);

    auto q = transition_q(m);
    auto rands = at::rand_like(m.logits.t.select(2, 0));

    Descent descent{
        m.seats.t.new_empty({B}),
        m.seats.t.new_empty({B})};

    for (int b=0; b<B; b++) {
        descend_kernel(m.ta(), H3D(q).ta(), H2D(rands).ta(), descent.ta(), b);
    }

    return descent;
}

void backup_kernel(BackupTA bk, S1D::TA leaves, uint b) {
    const uint B = bk.v.size(0);
    const uint S = bk.v.size(2);

    if (b >= B) return;

    float* v = (float*)malloc(S*sizeof(float));

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

    free(v);
}

void backup(Backup bk, TT leaves) {
    const uint B = bk.v.size(0);
    const uint S = bk.v.size(2);

    for (int b=0; b<B; b++) {
        backup_kernel(bk.ta(), S1D(leaves).ta(), b);
    }
}

}