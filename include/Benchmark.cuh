#pragma once
#define CUDA_ASSERT(error) ( assert((error) == cudaSuccess) )  

typedef struct
{
	size_t arrSize; //in bytes
	size_t numAccess;
	unsigned int blockSize;
	unsigned int gridSize;

} param_t;

inline void checkError()
{
	cudaError_t error =  cudaGetLastError();
	if(error != cudaSuccess)
	{
		const char *str = cudaGetErrorString(error); 
		printf("Cuda Error: %s\n", str);	
		exit(1);	
	}

}

template<typename T>
__global__ void readBW(T* __restrict__ d_ptr, T* __restrict__ dummy, const size_t N, const size_t numAccess)
{

	const size_t tid = threadIdx.x + (blockIdx.x * blockDim.x);
	T reg = 0;

	if(tid < N)
	{
		
	for(size_t idx = tid; idx < numAccess * (blockDim.x * gridDim.x); idx += blockDim.x * gridDim.x)
	{
		reg += d_ptr[idx];	
	}

	if(reg == 1)
	{
		printf("Exit Condition Met...");
		*dummy = reg;
		
	}
	}

}

template<typename T>
void readBWTest(param_t* __restrict__ pType)
{
	T* d_ptr, * dummy;
	size_t N = pType->arrSize / sizeof(T);

	CUDA_ASSERT(cudaMalloc((void **)&d_ptr, pType->arrSize));
	CUDA_ASSERT(cudaMalloc((void **)&dummy, sizeof(T) * 1));
	CUDA_ASSERT(cudaMemset(d_ptr, 0xFF, pType->arrSize));
	
	std::cout << "Block Size " << pType->blockSize 
	<< "Grid Size " << pType->gridSize 
	<< "Array Size " << pType->arrSize
	<< "Num Access " << pType->numAccess << std::endl;	
	
	
	readBW <T> <<<pType->gridSize, pType->blockSize>>>(d_ptr, dummy, N, pType->numAccess);
	checkError();	
	cudaDeviceSynchronize();
	checkError();
	CUDA_ASSERT(cudaFree(d_ptr));
	CUDA_ASSERT(cudaFree(dummy));
}

