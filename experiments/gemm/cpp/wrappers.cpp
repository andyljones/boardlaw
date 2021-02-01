#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include "../../../boardlaw/cpp/common.h"
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDACachingAllocator.h>

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

    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDABlas.cpp#L364
    // globalContext().alertCuBLASConfigNotDeterministic();
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    assert(!cublasCreate(&handle));

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
    auto A = (long)x.data_ptr() + idxs*k*size;
    auto B = (long)W.data_ptr() + range*(m*k)*size;
    auto C = (long)y.data_ptr() + range*m*size;

    printf("l %d; m %d; n %d; k %d; lda %d; ldb %d; ldc %d\n", l, m, n, k, lda, ldb, ldc);

    int memsize = l*sizeof(float*);
    auto Ar = (float**)c10::cuda::CUDACachingAllocator::raw_alloc(memsize);
    auto Br = (float**)c10::cuda::CUDACachingAllocator::raw_alloc(memsize);
    auto Cr = (float**)c10::cuda::CUDACachingAllocator::raw_alloc(memsize);

    C10_CUDA_CHECK(cudaMemcpy(Ar, A.data_ptr(), memsize, cudaMemcpyDeviceToDevice));
    C10_CUDA_CHECK(cudaMemcpy(Br, B.data_ptr(), memsize, cudaMemcpyDeviceToDevice));
    C10_CUDA_CHECK(cudaMemcpy(Cr, C.data_ptr(), memsize, cudaMemcpyDeviceToDevice));

    float tmp;
    float* tmpptr;
    C10_CUDA_CHECK(cudaMemcpy(&tmpptr, Ar, sizeof(float*), cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaMemcpy(&tmp, tmpptr, sizeof(float), cudaMemcpyDeviceToHost));
    printf("tmp: %f\n", tmp);

    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
    TORCH_CUDABLAS_CHECK(cublasSgemmBatched(
        handle, 
        CUBLAS_OP_N, 
        CUBLAS_OP_N, 
        m, n, k,
        &alpha, 
        Ar, lda, 
        Br, ldb, 
        &beta,
        Cr, ldc, 
        l));
    
    return y;
}

float exemplar() {

    int size = 2;
    int num = 2;
    
    float *matrices = (float*)malloc(size * size * num * sizeof(float));
    float *vectors = (float*)malloc(size * num * sizeof(float));

    assert(matrices);
    assert(vectors);

    for(int i = 0; i < num * size * size; i++)
        matrices[i] = 3.f; 

    for(int i = 0; i < num * size; i++)
        vectors[i] = 2.f;

    at::globalContext().alertCuBLASConfigNotDeterministic();
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    // assert(!cublasCreate(&handle));

    // allocate input space on device
    float *devMatrices;
    size_t devMatricesPitch;
    C10_CUDA_CHECK(cudaMallocPitch((void**)&devMatrices, &devMatricesPitch, size * sizeof(float), num * size));

    float *devVectors = 0;
    size_t devVectorsPitch;
    C10_CUDA_CHECK(cudaMallocPitch((void**)&devVectors, &devVectorsPitch, size * sizeof(float), num));

    // allocate result space on device
    float *devResult = 0;
    size_t devResultPitch;
    C10_CUDA_CHECK(cudaMallocPitch((void**)&devResult, &devResultPitch, size * sizeof(float), num));

    // copy data to device
    C10_CUDA_CHECK(cudaMemcpy2D(devMatrices, devMatricesPitch, matrices, size * sizeof(float), size * sizeof(float), size * num, cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy2D(devVectors, devVectorsPitch, vectors, size * sizeof(float), size * sizeof(float), num, cudaMemcpyHostToDevice));

    // create lists of device pointers to inputs and outputs
    float **AList = 0, **BList = 0, **CList = 0;
    AList = (float**)malloc(num * sizeof(float*));
    BList = (float**)malloc(num * sizeof(float*));
    CList = (float**)malloc(num * sizeof(float*));

    for(int i = 0; i < num; i++){
        AList[i] = devMatrices + devMatricesPitch/sizeof(float) * size * i;
        BList[i] = devVectors + devVectorsPitch/sizeof(float) * i;
        CList[i] = devResult + devResultPitch/sizeof(float) * i;
    }

    // copy pointer lists to device
    float **devAList = 0, **devBList = 0, **devCList = 0;
    C10_CUDA_CHECK(cudaMalloc((void**)&devAList, num * sizeof(float*)));
    C10_CUDA_CHECK(cudaMalloc((void**)&devBList, num * sizeof(float*)));
    C10_CUDA_CHECK(cudaMalloc((void**)&devCList, num * sizeof(float*)));
    C10_CUDA_CHECK(cudaMemcpy(devAList, AList, num * sizeof(float*), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMemcpy(devBList, BList, num * sizeof(float*), cudaMemcpyHostToDevice)); 
    C10_CUDA_CHECK(cudaMemcpy(devCList, CList, num * sizeof(float*), cudaMemcpyHostToDevice));

    int lda = devMatricesPitch / sizeof(float);
    int ldb = devVectorsPitch / sizeof(float);
    int ldc = devResultPitch / sizeof(float);
    const float alpha = 1.0f, beta = 0.0f;

    double sum = 0.0;
    TORCH_CUDABLAS_CHECK(cublasSgemmBatched(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                size,
                1,
                size,
                &alpha,
                (const float**)devAList,
                lda,
                (const float**)devBList,
                ldb,
                &beta,
                devCList,
                ldc,
                num));

    // copy data to host
    float *result = (float*)malloc(devResultPitch);
    C10_CUDA_CHECK(cudaMemcpy2D(result, sizeof(float), devResult, devResultPitch, sizeof(float), num, cudaMemcpyDeviceToHost));

    float ret = result[0];

    free(matrices);
    free(vectors);
    free(result);

    free(AList);
    free(BList);
    free(CList);

    cudaFree(devVectors);
    cudaFree(devMatrices);
    cudaFree(devResult);
        
    return ret;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("linear", &linear, "W"_a, "x"_a, "b"_a, "idxs"_a, py::call_guard<py::gil_scoped_release>());
    m.def("exemplar", &exemplar, py::call_guard<py::gil_scoped_release>());
}