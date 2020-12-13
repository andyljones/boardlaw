#include "../../cpp/kernels.cu"
#include "common.h"

const uint BLOCK = 8;

enum {
    EMPTY,
    BLACK,
    WHITE,
    TOP,
    BOT, 
    LEFT,
    RIGHT
};

__device__ void flood(C3D::PTA board, int row, int col, uint8_t new_val) {
    const uint S = board.size(1);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    const int neighbours[6][2] = {{-1, 0}, {-1, +1}, {0, -1}, {0, +1}, {+1, -1}, {+1, 0}};

    // If we don't need to flood the value, break
    if (new_val < TOP) { return; }

    uint8_t old_val = board[b][row][col];

    extern __shared__ uint8_t shared[];

    // Set up a queue to keep track of which cells need exploring
    int start = 0;
    int end = 1;
    uint8_t *queue = (uint8_t*)&shared[(3*threadIdx.x+0)*S*S];
    queue[0] = row;
    queue[1] = col;
    printf("Flooding from (%d, %d)\n", row, col);

    // Set up a mask to keep track of which cells we've already seen
    uint8_t *seen = (uint8_t*)&shared[(3*threadIdx.x+2)*S*S];
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
            printf("Updating (%d, %d)\n", r0, c0);
            // and add the neighbours to the queue
            for (int n=0; n<6; n++) {
                int r = r0 + neighbours[n][0];
                int c = c0 + neighbours[n][1];
                // but only if they're not over the edge
                if ((0 <= r) && (r < S) && (0 <= c) && (c < S)) {
                    // and we haven't seen them already
                    if (!seen[r*S+c]) {
                        printf("Adding (%d, %d)\n", r, c);
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
        } 

        if (adj[LEFT]) { new_val = LEFT; } 
        else if (adj[RIGHT]) { new_val = RIGHT; } 
        else { new_val = WHITE; }
    } else {
        if (adj[TOP] && adj[BOT]) {
            results[b][0] = +1.f;
            results[b][1] = -1.f;
        } 

        if (adj[TOP]) { new_val = TOP; } 
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

    return results;
}

__host__ TT observe(TT board, TT seats) {
    c10::cuda::CUDAGuard g(board.device());
    auto black = ((board == BLACK) | (board == TOP) | (board == BOT));
    auto white = ((board == WHITE) | (board == LEFT) | (board == RIGHT));
    auto black_obs = at::stack({black, white}, -1).toType(at::kFloat);

    auto white_obs = black_obs.transpose(-3, -2).flip(-1);

    //TODO: This is garbage, what's a better way?
    auto dims = std::vector<int64_t>({});
    for (int d=0; d<seats.ndimension(); d++) { dims.push_back(seats.size(d)); }
    for (int d=0; d<3; d++) { dims.push_back(1); }

    return black_obs.where(seats.reshape(dims) == 0, white_obs);
}