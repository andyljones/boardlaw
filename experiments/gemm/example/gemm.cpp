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
    int l = W.size(0);
    int h = W.size(1);
    int w = W.size(2);

    float *w_ptr = W.data_ptr<float>();
    float *x_ptr = x.data_ptr<float>();

    auto y = b.index(idxs);
    float *y_ptr = y.data_ptr<float>();

    //TODO: Think these are wrong
    auto range = (long)sizeof(float)*at::arange(l, idxs.options());
    auto A = (long)w_ptr + w*h*range;
    auto B = (long)x_ptr + w*range;
    auto C = (long)y_ptr + h*range;

    int m = h;
    int n = 1;
    int k = w;

    int lda = h;
    int ldb = w;
    int ldc = h;
    const float alpha = 1.0f, beta = 1.0f;

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK2(cublasSgemmBatched(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                (const float**)A.data_ptr(),
                lda,
                (const float**)B.data_ptr(),
                ldb,
                &beta,
                (float**)C.data_ptr(),
                ldc,
                l));

    return y;
}

int test_one_batch() {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    // [1 2 3] [7] + [10] = [60]
    // [4 5 6] [8]   [11]   [133]
    //         [9]
    auto W = at::tensor({1, 2, 3, 4, 5, 6}, options).view({1, 2, 3});
    auto x = at::tensor({7, 8, 9}, options).view({1, 3});
    auto b = at::tensor({10, 11}, options).view({1, 2});

    W = W.transpose(1, 2).contiguous().transpose(1, 2);

    auto idxs = at::tensor({0}).toType(at::kLong).cuda();
    auto expected = at::bmm(W, x.unsqueeze(-1)).squeeze(-1) + b;

    auto y = test(W, x, b, idxs);

    cout << y << endl << expected << endl;

    assert(at::allclose(expected, y));
}

int test_linear_batches() {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    auto W = at::tensor({1, 2, 3}, options).view({3, 1, 1});
    auto x = at::tensor({4, 5, 6}, options).view({3, 1});
    auto b = at::tensor({0, 0, 0}, options).view({3, 1});

    W = W.transpose(1, 2).contiguous().transpose(1, 2);

    auto idxs = at::arange(3).cuda();
    auto expected = at::bmm(W, x.unsqueeze(-1)).squeeze(-1) + b;

    auto y = test(W, x, b, idxs);

    cout << y << endl << expected << endl;

    assert(at::allclose(expected, y));
}

int main() {
    test_one_batch();
    test_linear_batches();
    return 0;
}