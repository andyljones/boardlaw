#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"

const uint BLOCK = 8;

__global__ void step_kernel(
    I3D::PTA board, I1D::PTA seats, I1D::PTA actions, I1D::PTA results) {

    const uint B = board.size(0);
    const uint S = board.size(1);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    if (b >= B) return;

    int row = actions[b] / S;
    int col = actions[b] % S;

    board[b][row][col] = 1;

    results[b] = 1;

}

__host__ TT step(TT board, TT seats, TT actions) {
    const uint B = board.size(0);

    TT results = at::zeros_like(actions);

    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    step_kernel<<<{n_blocks}, {BLOCK}>>>(
        I3D(board).pta(), I1D(seats).pta(), I1D(actions).pta(), I1D(results).pta());

    return results;
}