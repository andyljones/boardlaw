#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"

const uint BLOCK = 8;

enum {
    EMPTY,
    BLACK,
    BLACK_WIN,
    TOP,
    BOT, 
    WHITE,
    WHITE_WIN,
    LEFT,
    RIGHT
};


__global__ void step_kernel(
    I3D::PTA board, I1D::PTA seats, I1D::PTA actions, F2D::PTA results) {

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
    __shared__ bool adj[9];
    for (int a=0; a<9; a++) {
        adj[a] = false;
    }

    // Populate the adjacency indicator
    for (int i=-1; i<=1; i++) {
        for (int j=-1; j<=1; j++) {
            int r = row + i;
            int c = col + j;
            if      (r <  0) { adj[TOP] = true; }
            else if (r >= S) { adj[BOT] = true; }
            else if (c <  0) { adj[LEFT] = true; }
            else if (c >= S) { adj[RIGHT] = true; }
            else { adj[board[b][r][c]] = true; }
        }
    }

    // Use the adjacency to decide what the new cell should be
    char new_cell;
    if (seat) {
        if (adj[LEFT] && adj[RIGHT]) { new_cell = WHITE_WIN; results[b][1] = 1.f; } 
        else if (adj[LEFT]) { new_cell = LEFT; } 
        else if (adj[RIGHT]) { new_cell = RIGHT; } 
        else { new_cell = WHITE; }
    } else {
        if (adj[TOP] && adj[BOT]) { new_cell = BLACK_WIN; results[b][0] = 1.f; } 
        else if (adj[TOP]) { new_cell = TOP; } 
        else if (adj[BOT]) { new_cell = BOT; } 
        else { new_cell = BLACK; }
    }

    board[b][row][col] = new_cell;
}

__host__ TT step(TT board, TT seats, TT actions) {
    const uint B = board.size(0);

    TT results = board.new_zeros({B, 2}, at::kFloat);

    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    step_kernel<<<{n_blocks}, {BLOCK}>>>(
        I3D(board).pta(), I1D(seats).pta(), I1D(actions).pta(), F2D(results).pta());

    return results;
}