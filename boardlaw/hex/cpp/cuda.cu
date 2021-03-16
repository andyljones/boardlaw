#include "../../cpp/kernels.cu"
#include "common.h"

namespace hexcuda {

const uint BLOCK = 8;

enum {
    EMPTY,
    BLACK,
    WHITE,
    TOP,
    BOT, 
    LEFT,
    RIGHT,
    BLACK_WIN,
    WHITE_WIN
};

__device__ void flood(C3D::PTA board, int row, int col, uint8_t new_val) {
    const uint S = board.size(1);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    const int neighbours[6][2] = {{-1, 0}, {-1, +1}, {0, -1}, {0, +1}, {+1, -1}, {+1, 0}};

    // If we don't need to flood the value, break
    if (new_val < TOP) { return; }

    uint8_t old_val = board[b][row][col];

    extern __shared__ uint8_t colors[];

    // Set up a queue to keep track of which cells need exploring
    int start = 0;
    int end = 1;
    uint8_t *queue = (uint8_t*)&colors[(3*threadIdx.x+0)*S*S];
    queue[0] = row;
    queue[1] = col;

    // Set up a mask to keep track of which cells we've already seen
    uint8_t *seen = (uint8_t*)&colors[(3*threadIdx.x+2)*S*S];
    for (int s=0; s<S*S; s++) { seen[s] = 0; }

    while (true) {
        // See if there's anything left in the queue
        if (start == end) { break; }

        // Pull a value out of the queue
        int r0 = queue[2*start+0];
        int c0 = queue[2*start+1];        
        start += 1;
        uint8_t cell_val = board[b][r0][c0];

        // If the old and new vals are the same, continue flooding!
        if (cell_val == old_val) {
            // Put the new value into place
            board[b][r0][c0] = new_val;
            // and add the neighbours to the queue
            for (int n=0; n<6; n++) {
                int r = r0 + neighbours[n][0];
                int c = c0 + neighbours[n][1];
                // but only if they're not over the edge
                if ((0 <= r) && (r < S) && (0 <= c) && (c < S)) {
                    // and we haven't seen them already
                    if (!seen[r*S+c]) {
                        queue[2*end+0] = r;
                        queue[2*end+1] = c;
                        end += 1;

                        seen[r*S+c] = 1;
                    }
                }
            }
        }
    }
}

__global__ void step_kernel(
    C3D::PTA board, I1D::PTA seats, I1D::PTA actions, F2D::PTA results) {

    const uint B = board.size(0);
    const uint S = board.size(1);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    if (b >= B) return;

    const int seat = seats[b]; 

    // Swap rows and cols if we're playing white
    const int action = actions[b];
    int row, col;
    if (seat == 0) { row = action / S, col = action % S; }
    else           { row = action % S, col = action / S; }

    // Set up the adjacency indicator
    bool adj[9];
    for (int a=0; a<9; a++) {
        adj[a] = false;
    }

    const int neighbours[6][2] = {{-1, 0}, {-1, +1}, {0, -1}, {0, +1}, {+1, -1}, {+1, 0}};
    // Populate the adjacency indicator
    for (int n=0; n<6; n++) {
        int r = row + neighbours[n][0];
        int c = col + neighbours[n][1];

        if      (r <  0) { adj[TOP] = true; }
        else if (r >= S) { adj[BOT] = true; }
        else if (c <  0) { adj[LEFT] = true; }
        else if (c >= S) { adj[RIGHT] = true; }
        else { adj[board[b][r][c]] = true; }
    }

    // Use the adjacency to decide what the new cell should be
    char new_val;
    if (seat) {
        if (adj[LEFT] && adj[RIGHT]) { 
            results[b][0] = -1.f;
            results[b][1] = +1.f;
            new_val = WHITE_WIN; } 
        else if (adj[LEFT]) { new_val = LEFT; } 
        else if (adj[RIGHT]) { new_val = RIGHT; } 
        else { new_val = WHITE; }
    } else {
        if (adj[TOP] && adj[BOT]) {
            results[b][0] = +1.f;
            results[b][1] = -1.f;
            new_val = BLACK_WIN;
        } else if (adj[TOP]) { new_val = TOP; } 
        else if (adj[BOT]) { new_val = BOT; } 
        else { new_val = BLACK; }
    }

    board[b][row][col] = seat? WHITE : BLACK;

    flood(board, row, col, new_val);
}

__host__ TT step(TT board, TT seats, TT actions) {
    c10::cuda::CUDAGuard g(board.device());
    const uint B = board.size(0);
    const uint S = board.size(1);

    TT results = board.new_zeros({B, 2}, at::kFloat);

    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    step_kernel<<<{n_blocks}, {BLOCK}, BLOCK*S*S*3*sizeof(uint8_t), stream()>>>(
        C3D(board).pta(), I1D(seats).pta(), I1D(actions).pta(), F2D(results).pta());
    C10_CUDA_CHECK(cudaGetLastError());

    return results;
}

__global__ void observe_kernel(C3D::PTA board, I1D::PTA seats, F4D::PTA obs) {
    const uint B = board.size(0);
    const uint S = board.size(1);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    if (b >= B) return;

    extern __shared__ uint8_t shared[];
    uint8_t* colors = (uint8_t*)&shared[threadIdx.x*S*S];

    // Copy the color into shared memory
    for (int i=0; i<S; i++) {
        for (int j=0; j<S; j++) {
            auto c = board[b][i][j];
            uint8_t color = 2;
            if ((c == BLACK) | (c == TOP) | (c == BOT) | (c == BLACK_WIN)) {
                color = 0;
            } else if ((c == WHITE) | (c == LEFT) | (c == RIGHT) | (c == WHITE_WIN)) {
                color = 1;
            }
            colors[i*S+j] = color;
        }
    }

    // Copy the colors to the obs 
    auto flip = seats[b] == 1;
    for (int i=0; i<S; i++) {
        for (int j=0; j<S; j++) {
            auto idx = flip? (j*S+i) : (i*S+j);
            auto c = colors[idx];
            // printf("%d/%d %d/%d %d/%d %d/%d\n", b, B, i, S, j, S, c, 2);

            if (c < 2) {
                if (flip){
                    obs[b][i][j][1-c] = 1.f;
                } else {
                    obs[b][i][j][c] = 1.f;
                }
            }
        }
    }
}

__host__ TT observe(TT board, TT seats) {
    c10::cuda::CUDAGuard g(board.device());

    auto flatboard = board.clone().view({-1, board.size(-1), board.size(-1)});
    auto flatseats = seats.clone().view({-1}).to(at::kInt);

    const uint B = flatboard.size(0);
    const uint S = flatboard.size(1);

    auto obs = flatboard.new_zeros({B, S, S, 2}, at::kFloat);

    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    observe_kernel<<<{n_blocks}, {BLOCK}, BLOCK*S*S*sizeof(uint8_t), stream()>>>(
        C3D(flatboard).pta(), I1D(flatseats).pta(), F4D(obs).pta());
    C10_CUDA_CHECK(cudaGetLastError());

    auto sizes = board.sizes().vec();
    sizes.push_back(2);
    return obs.view(sizes);

}

}