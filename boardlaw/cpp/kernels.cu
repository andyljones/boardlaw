#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}