#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/ATen.h>

#include "../../../boardlaw/cpp/common.h"

using namespace std;

int main() {
    auto x = at::zeros(1).cuda();

    int size = 1;
    int num = 2;
    
    float *matrices = (float*)malloc(size * size * num * sizeof(float));
    float *vectors = (float*)malloc(size * num * sizeof(float));

    assert(matrices);
    assert(vectors);

    for(int i = 0; i < num * size * size; i++)
        matrices[i] = 3.f; 

    for(int i = 0; i < num * size; i++)
        vectors[i] = 2.f;

    // cublasStatus_t stat;
    // cublasHandle_t handle;
    // assert(!cublasCreate(&handle));
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    // allocate input space on device
    float *devMatrices;
    size_t devMatricesPitch;
    assert(!cudaMallocPitch((void**)&devMatrices, &devMatricesPitch, size * sizeof(float), num * size));

    float *devVectors = 0;
    size_t devVectorsPitch;
    assert(!cudaMallocPitch((void**)&devVectors, &devVectorsPitch, size * sizeof(float), num));

    // allocate result space on device
    float *devResult = 0;
    size_t devResultPitch;
    assert(!cudaMallocPitch((void**)&devResult, &devResultPitch, size * sizeof(float), num));

    // copy data to device
    assert(!cudaMemcpy2D(devMatrices, devMatricesPitch, matrices, size * sizeof(float), size * sizeof(float), size * num, cudaMemcpyHostToDevice));
    assert(!cudaMemcpy2D(devVectors, devVectorsPitch, vectors, size * sizeof(float), size * sizeof(float), num, cudaMemcpyHostToDevice));

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
    assert(!cudaMalloc((void**)&devAList, num * sizeof(float*)));
    assert(!cudaMalloc((void**)&devBList, num * sizeof(float*)));
    assert(!cudaMalloc((void**)&devCList, num * sizeof(float*)));
    assert(!cudaMemcpy(devAList, AList, num * sizeof(float*), cudaMemcpyHostToDevice));
    assert(!cudaMemcpy(devBList, BList, num * sizeof(float*), cudaMemcpyHostToDevice)); 
    assert(!cudaMemcpy(devCList, CList, num * sizeof(float*), cudaMemcpyHostToDevice));

    int lda = devMatricesPitch / sizeof(float);
    int ldb = devVectorsPitch / sizeof(float);
    int ldc = devResultPitch / sizeof(float);
    const float alpha = 1.0f, beta = 0.0f;

    assert(!cublasSgemmBatched(handle,
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
    assert(!cudaMemcpy2D(result, sizeof(float), devResult, devResultPitch, sizeof(float), num, cudaMemcpyDeviceToHost));

    for(int i = 0; i < num * size; i++)
        cout << result[i] << endl;

    free(matrices);
    free(vectors);
    free(result);

    free(AList);
    free(BList);
    free(CList);

    cudaFree(devVectors);
    cudaFree(devMatrices);
    cudaFree(devResult);
        
  return 0;
}
