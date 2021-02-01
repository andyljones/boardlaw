#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

int main(int argc, char ** argv){

  int status;
  int lower = 2;
  int upper = 100;
  int num = 25000;
  int reps = 5;
  int verbose = 0;
  
  while((status = getopt(argc, argv, "l:u:n:r:v")) != -1){
    switch(status){
    case 'l':
      lower = strtoul(optarg, 0, 0);
      break;
    case 'u':
      upper = strtoul(optarg, 0, 0);
      break;
    case 'n':
      num = strtoul(optarg, 0, 0);
      break;
    case 'r':
      reps = strtoul(optarg, 0, 0);
      break;
    case 'v':
      verbose = 1;
      break;
    default:
      cerr << "invalid argument: " << status << endl;
      exit(1);
    }
  }
  if(verbose) 
    cout << "running with" 
	 << " lower: " << lower
	 << " upper: " << upper
	 << " num: " << num
	 << " reps: " << reps
	 << endl;

  if(verbose) cout << "initializing inputs" << endl;
  float *matrices = (float*)malloc(upper * upper * num * sizeof(float));
  float *vectors = (float*)malloc(upper * num * sizeof(float));

  assert(matrices);
  assert(vectors);

  for(int i = 0; i < num * upper * upper; i++)
    matrices[i] = drand48();

  for(int i = 0; i < num * upper; i++)
    vectors[i] = drand48();

  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;

  stat = cublasCreate(&handle);
  if(stat != CUBLAS_STATUS_SUCCESS){
    cerr << "cublas init failed" << endl;
    exit(1);
  }

  if(verbose) cout << "allocating device variables" << endl;

  // allocate input space on device
  float *devMatrices;
  size_t devMatricesPitch;
  cudaStat = 
    cudaMallocPitch(&devMatrices,
		    &devMatricesPitch,
		    upper * sizeof(float),
		    num * upper);

  assert(!cudaStat);

  float *devVectors = 0;
  size_t devVectorsPitch;
  cudaStat = 
    cudaMallocPitch(&devVectors,
		    &devVectorsPitch,
		    upper * sizeof(float),
		    num);

  assert(!cudaStat);

  // allocate result space on device
  float *devResult = 0;
  size_t devResultPitch;
  cudaStat = 
    cudaMallocPitch(&devResult,
		    &devResultPitch,
		    upper * sizeof(float),
		    num);

  assert(!cudaStat);

  if(verbose) cout << "copying data to device" << endl;
  // copy data to device
  cudaStat = 
    cudaMemcpy2D(devMatrices,
		 devMatricesPitch,
		 matrices,
		 upper * sizeof(float),
		 upper * sizeof(float),
		 upper * num,
		 cudaMemcpyHostToDevice);

  assert(!cudaStat);
  
  cudaStat = 
    cudaMemcpy2D(devVectors,
		 devVectorsPitch,
		 vectors,
		 upper * sizeof(float),
		 upper * sizeof(float),
		 num,
		 cudaMemcpyHostToDevice);

  assert(!cudaStat);

  // create lists of device pointers to inputs and outputs
  float **AList = 0, **BList = 0, **CList = 0;

  AList = (float**)malloc(num * sizeof(float*));
  BList = (float**)malloc(num * sizeof(float*));
  CList = (float**)malloc(num * sizeof(float*));

  for(int i = 0; i < num; i++){
    AList[i] = devMatrices + devMatricesPitch/sizeof(float) * upper * i;
    BList[i] = devVectors + devVectorsPitch/sizeof(float) * i;
    CList[i] = devResult + devResultPitch/sizeof(float) * i;
  }

  // copy pointer lists to device
  float **devAList = 0, **devBList = 0, **devCList = 0;
  cudaStat = cudaMalloc(&devAList, num * sizeof(float*));
  assert(!cudaStat);

  cudaStat = cudaMalloc(&devBList, num * sizeof(float*));
  assert(!cudaStat);

  cudaStat = cudaMalloc(&devCList, num * sizeof(float*));
  assert(!cudaStat);

  cudaStat = cudaMemcpy(devAList,
			AList,
			num * sizeof(float*),
			cudaMemcpyHostToDevice);
  assert(!cudaStat);
  
  cudaStat = cudaMemcpy(devBList,
			BList,
			num * sizeof(float*),
			cudaMemcpyHostToDevice);
  assert(!cudaStat);

  cudaStat = cudaMemcpy(devCList,
			CList,
			num * sizeof(float*),
			cudaMemcpyHostToDevice);
  assert(!cudaStat);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  int 
    lda = devMatricesPitch / sizeof(float),
    ldb = devVectorsPitch / sizeof(float),
    ldc = devResultPitch / sizeof(float);
  const float alpha = 1.0f, beta = 0.0f;

  /* perform <num> <size x size> x <size x 1> multiplications 
     with distinct matrices
   */
  for(int size = lower; size <= upper; size++){
    if(verbose) cout << "running with size " << size << endl;
    double sum = 0.0;
    for(int rep = 0; rep < reps; rep++){
      cudaEventRecord(start, 0);
      stat = cublasSgemmBatched(handle,
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
				num);
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      if(stat != CUBLAS_STATUS_SUCCESS){
	cerr << "cublasSgemmBatched failed" << endl;
	exit(1);
      }
      assert(!cudaGetLastError());
      
      float elapsed;
      cudaEventElapsedTime(&elapsed, start, stop);
      elapsed /= 1000.0f;
      sum += elapsed;
      
      if(verbose)
	cout << "distinct; size " << size << ": " << elapsed << " s; " 
	     << elapsed / num << " s per operation" << endl;
    }
    cout << "distinct; size " << size << " average: " << sum/reps << " s; "
	 << sum / reps / num << " s per operation" << endl;
  }

  /* Perform <num> <size x size> x <size x 1> multiplications 
     with a single matrix.
     Is it possible to use constant memory cublas?
  */

  for(int i = 0; i < num; i++)
    AList[i] = devMatrices;

  for(int size = lower; size <= upper; size++){
    if(verbose) cout << "running with size " << size << endl;
    double sum = 0.0;
    for(int rep = 0; rep < reps; rep++){
      cudaEventRecord(start, 0);
      stat = cublasSgemmBatched(handle,
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
				num);
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      if(stat != CUBLAS_STATUS_SUCCESS){
	cerr << "cublasSgemmBatched failed" << endl;
	exit(1);
      }
      assert(!cudaGetLastError());
      
      float elapsed;
      cudaEventElapsedTime(&elapsed, start, stop);
      elapsed /= 1000.0f;
      sum += elapsed;
      
      if(verbose)
	cout << "single; size " << size << ": " << elapsed << " s; " 
	     << elapsed / num << " s per operation" << endl;
    }
    cout << "single; size " << size << " average: " << sum/reps << " s; "
	 << sum / reps / num << " s per operation" << endl;
  }

  free(matrices);
  free(vectors);
      
  return 0;
}
