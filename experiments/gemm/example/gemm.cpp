#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/ATen.h>

#include "../../../boardlaw/cpp/common.h"

using namespace std;

const char* _cublasGetErrorEnum(cublasStatus_t error) {
  if (error == CUBLAS_STATUS_SUCCESS) {
    return "CUBLAS_STATUS_SUCCESS";
  }
  if (error == CUBLAS_STATUS_NOT_INITIALIZED) {
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  }
  if (error == CUBLAS_STATUS_ALLOC_FAILED) {
    return "CUBLAS_STATUS_ALLOC_FAILED";
  }
  if (error == CUBLAS_STATUS_INVALID_VALUE) {
    return "CUBLAS_STATUS_INVALID_VALUE";
  }
  if (error == CUBLAS_STATUS_ARCH_MISMATCH) {
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  }
  if (error == CUBLAS_STATUS_MAPPING_ERROR) {
    return "CUBLAS_STATUS_MAPPING_ERROR";
  }
  if (error == CUBLAS_STATUS_EXECUTION_FAILED) {
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  }
  if (error == CUBLAS_STATUS_INTERNAL_ERROR) {
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  if (error == CUBLAS_STATUS_NOT_SUPPORTED) {
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  }
}

// Copied in from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/Exceptions.h 
// since the underlying _cublasGetErrorEnum doesn't seem to be exported
#define TORCH_CUDABLAS_CHECK2(EXPR)                             \
  do {                                                          \
    cublasStatus_t __err = EXPR;                                \
    TORCH_CHECK(__err == CUBLAS_STATUS_SUCCESS,                 \
                "CUDA error: ",                                 \
                _cublasGetErrorEnum(__err),                     \
                " when calling `" #EXPR "`");                   \
  } while (0)

TT test(TT W, TT x, TT b, TT idxs) {
    int size = W.size(1);
    int num = W.size(0);

    float *devMatrices = W.data_ptr<float>();
    float *devVectors = x.data_ptr<float>();

    auto handle = at::cuda::getCurrentCUDABlasHandle();

    auto y = b.index(idxs);
    float *devResult = y.data_ptr<float>();

    auto range = at::arange(num, idxs.options());
    auto devAList = (long)devMatrices + W.stride(0)*size*range;
    auto devBList = (long)devVectors + x.stride(0)*range;
    auto devCList = (long)devResult + y.stride(0)*range;

    int lda = W.stride(0);
    int ldb = x.stride(0);
    int ldc = y.stride(0);
    const float alpha = 1.0f, beta = 1.0f;

    TORCH_CUDABLAS_CHECK2(cublasSgemmBatched(handle,
                CUBLAS_OP_T,
                CUBLAS_OP_T,
                size,
                1,
                size,
                &alpha,
                (const float**)devAList.data_ptr(),
                lda,
                (const float**)devBList.data_ptr(),
                ldb,
                &beta,
                (float**)devCList.data_ptr(),
                ldc,
                num));

    cout << y << endl;

    return y;
}

int main() {
    for (int i=0; i<100; i++) {
        auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
        // [1 2] [7] + [5] = [28]
        // [3 4] [8]   [6]   [59]
        auto W = at::tensor({1, 2, 3, 4}, options).view({1, 2, 2});
        auto x = at::tensor({7, 8}, options).view({1, 2});
        auto b = at::tensor({5, 6}, options).view({1, 2});

        auto idxs = at::tensor({0}).toType(at::kLong).cuda();

        test(W, x, b, idxs);
    }
    return 0;
}