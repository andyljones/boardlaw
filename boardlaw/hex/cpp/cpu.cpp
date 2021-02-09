#include "common.h"

namespace hexcpu {

enum {
    EMPTY,
    BLACK,
    WHITE,
    TOP,
    BOT, 
    LEFT,
    RIGHT
};

void flood(C3D::TA board, int row, int col, uint8_t new_val, uint b) {
    const uint S = board.size(1);

    const int neighbours[6][2] = {{-1, 0}, {-1, +1}, {0, -1}, {0, +1}, {+1, -1}, {+1, 0}};

    // If we don't need to flood the value, break
    if (new_val < TOP) { return; }

    uint8_t old_val = board[b][row][col];

    // Set up a queue to keep track of which cells need exploring
    int start = 0;
    int end = 1;
    uint8_t *queue = (uint8_t*)malloc(S*S*2*sizeof(uint8_t));
    queue[0] = row;
    queue[1] = col;

    // Set up a mask to keep track of which cells we've already seen
    uint8_t *seen = (uint8_t*)malloc(S*S*sizeof(uint8_t));
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

void step_kernel(
    C3D::TA board, I1D::TA seats, I1D::TA actions, F2D::TA results, uint b) {

    const uint B = board.size(0);
    const uint S = board.size(1);

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

    flood(board, row, col, new_val, b);
}

TT step(TT board, TT seats, TT actions) {
    const uint B = board.size(0);
    const uint S = board.size(1);

    TT results = board.new_zeros({B, 2}, at::kFloat);

    for (uint b=0; b<B; b++) {
        step_kernel(C3D(board).ta(), I1D(seats).ta(), I1D(actions).ta(), F2D(results).ta(), b);
    }

    return results;
}

void observe_kernel(C3D::TA board, I1D::TA seats, F4D::TA obs, uint b) {
    const uint B = board.size(0);
    const uint S = board.size(1);

    if (b >= B) return;

    uint8_t* colors = (uint8_t*)malloc(S*S*sizeof(uint8_t));

    // Copy the color into shared memory
    for (int i=0; i<S; i++) {
        for (int j=0; j<S; j++) {
            auto c = board[b][i][j];
            uint8_t color = 2;
            if ((c == BLACK) | (c == TOP) | (c == BOT)) {
                color = 0;
            } else if ((c == WHITE) | (c == LEFT) | (c == RIGHT)) {
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

TT observe(TT board, TT seats) {
    auto flatboard = board.clone().view({-1, board.size(-1), board.size(-1)});
    auto flatseats = seats.clone().view({-1}).to(at::kInt);

    const uint B = flatboard.size(0);
    const uint S = flatboard.size(1);

    auto obs = flatboard.new_zeros({B, S, S, 2}, at::kFloat);

    for (uint b=0; b<B; b++) {
        observe_kernel(C3D(flatboard).ta(), I1D(flatseats).ta(), F4D(obs).ta(), b);
    }

    auto sizes = board.sizes().vec();
    sizes.push_back(2);
    return obs.view(sizes);

}

}