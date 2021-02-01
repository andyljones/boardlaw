#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include "../../../boardlaw/cpp/common.h"
#include <ATen/cuda/CUDABlas.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;

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
#ifdef CUBLAS_STATUS_LICENSE_ERROR
  if (error == CUBLAS_STATUS_LICENSE_ERROR) {
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
#endif
  return "<unknown>";
}

// Copied in from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/Exceptions.h 
// since the underlying _cublasGetErrorEnum doesn't seem to be exported
#define TORCH_CUDABLAS_CHECK(EXPR)                              \
  do {                                                          \
    cublasStatus_t __err = EXPR;                                \
    TORCH_CHECK(__err == CUBLAS_STATUS_SUCCESS,                 \
                "CUDA error: ",                                 \
                _cublasGetErrorEnum(__err),                     \
                " when calling `" #EXPR "`");                   \
  } while (0)

TT linear(TT& W, const TT& x, const TT& b, const TT& idxs) {
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/LinearAlgebra.cu#L155
    CHECK_INPUT(W)
    CHECK_INPUT(x)
    CHECK_INPUT(b)
    CHECK_INPUT(idxs)

    //TODO: Write a macro for these
    TORCH_CHECK_TYPE(W.scalar_type() == at::kFloat, "W expected ", toString(at::kFloat), " got ", toString(W.scalar_type()));
    TORCH_CHECK_TYPE(x.scalar_type() == at::kFloat, "x expected ", toString(at::kFloat), " got ", toString(x.scalar_type()));
    TORCH_CHECK_TYPE(b.scalar_type() == at::kFloat, "b expected ", toString(at::kFloat), " got ", toString(b.scalar_type()));
    TORCH_CHECK_TYPE(idxs.scalar_type() == at::kLong, "idxs expected ", toString(at::kLong), " got ", toString(idxs.scalar_type()));

    TORCH_CHECK(W.dim() == 3, "W must be a 3D tensor");
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(b.dim() == 2, "b must be a 2D tensor");
    TORCH_CHECK(idxs.dim() == 1, "idxs must be a 1D tensor");

    TORCH_CHECK(W.size(0) == b.size(0), "W dim 0 must match b dim 0");
    TORCH_CHECK(W.size(1) == b.size(1), "W dim 1 must match b dim 1");
    TORCH_CHECK(W.size(2) == x.size(1), "W dim 2 must match x dim 1");
    TORCH_CHECK(idxs.size(0) == x.size(0), "idxs dim 0 must match x dim 0");

    auto y = b.index(idxs);

    // handle pathological cases that blas may not like
    if (y.numel() == 0) {
        return y;
    } 

    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDABlas.cpp#L264
    // globalContext().alertCuBLASConfigNotDeterministic();
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    auto l = W.size(0);
    auto m = W.size(1);
    auto n = 1;
    auto k = W.size(2);

    auto lda = m;
    auto ldb = k;
    auto ldc = m;

    float alpha = 1.f;
    float beta = 1.f;

    int size = sizeof(float);
    auto range = at::arange(l, idxs.options());
    auto A = (long)W.data_ptr<float>() + idxs*(m*k)*size;
    auto B = (long)x.data_ptr<float>() + range*k*size;
    auto C = (long)y.data_ptr<float>() + range*m*size;

    printf("l %d; m %d; n %d; k %d; lda %d; ldb %d; ldc %d\n", l, m, n, k, lda, ldb, ldc);

    // float tmp;
    // cudaMemcpy(&tmp, (void*)(C[0].item<long>()), sizeof(float), cudaMemcpyDeviceToHost);
    // C10_CUDA_CHECK(cudaGetLastError());

    // printf("C[0][0]: %f\n", tmp);
    printf("A: %p\n", A.data_ptr());
    printf("B: %p\n", B.data_ptr());
    printf("C: %p\n", C.data_ptr());
    printf("W: %p\n", W.data_ptr());
    printf("x: %p\n", x.data_ptr());
    printf("y: %p\n", y.data_ptr());

    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
    TORCH_CUDABLAS_CHECK(cublasSgemmBatched(
        handle, 
        CUBLAS_OP_T, 
        CUBLAS_OP_T, 
        m, n, k,
        &alpha, 
        (float**)A.data_ptr(), lda, 
        (float**)B.data_ptr(), ldb, 
        &beta,
        (float**)C.data_ptr(), ldc, 
        l));
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("linear", &linear, "W"_a, "x"_a, "b"_a, "idxs"_a, py::call_guard<py::gil_scoped_release>());
}