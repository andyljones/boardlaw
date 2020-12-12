#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"

const uint BLOCK = 8;

__global__ void flood_kernel(I3D::PTA board, I2D::PTA actions) {
    const uint B = board.size(0);
    const uint S = board.size(1);
    const int b = blockIdx.x*blockDim.x + threadIdx.x;

    if (b >= B) return;

    board[b][actions[b][0]][actions[b][1]] = 1;
}

__host__ void flood(TT board, TT actions) {
    const uint B = board.size(0);

    const uint n_blocks = (B + BLOCK - 1)/BLOCK;
    flood_kernel<<<{n_blocks}, {BLOCK}>>>(I3D(board).pta(), I2D(actions).pta());
}