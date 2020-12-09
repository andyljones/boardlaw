#include <math_constants.h>
#include <ATen/ATen.h>
#include "common.h"
#include <ATen/cuda/CUDAContext.h>

at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}

__host__ Solution solve_policy(const TT pi, const TT q, const TT lambda_n) {
    const uint B = pi.size(0);
    const uint A = pi.size(1);

    Solution soln(B);

    return soln;
}