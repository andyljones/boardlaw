#include <math.h>
#include <math_constants.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}