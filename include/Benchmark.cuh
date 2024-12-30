#pragma once
#define CUDA_ASSERT(error) ( assert((error) == cudaSuccess) )  
#define NUM_ITER 10

typedef struct
{
	double meas;
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

	size_t totalReads = blockDim.x * gridDim.x * numAccess;
	size_t stride = blockDim.x * gridDim.x;

	for(size_t iter = 0; iter < 10; iter++)
	{
		for(size_t idx = tid; idx < totalReads; idx += stride)
		{
			reg += d_ptr[idx];	
		}

		if(reg == 0xFE)
		{
			assert(0); //throw nasty error if exit condition met
			*dummy = reg;
		}		
	}
}

template<typename T>
void readBWTest(param_t* pType)
{
	T* d_ptr;
	T* dummy;
	size_t N = pType->arrSize / sizeof(T);

	assert((pType->arrSize >= pType->blockSize * pType->gridSize * pType->numAccess) && pType->arrSize % (pType->blockSize * pType->gridSize) == 0); //array size must be multiple of total num threads to avoid seg fault

	CUDA_ASSERT(cudaMalloc((void **)&d_ptr, pType->arrSize));
	CUDA_ASSERT(cudaMalloc((void **)&dummy, sizeof(T) * 1));
	CUDA_ASSERT(cudaMemset(d_ptr, 0xFF, pType->arrSize));
	
	std::cout << "Block Size " << pType->blockSize 
	<< " Grid Size " << pType->gridSize 
	<< " Array Size " << pType->arrSize
	<< " Num Access " << pType->numAccess << std::endl;	

	auto start = std::chrono::high_resolution_clock::now();
	readBW <T> <<<pType->gridSize, pType->blockSize>>>(d_ptr, dummy, N, pType->numAccess);
	// checkError();
	CUDA_ASSERT(cudaDeviceSynchronize());
	checkError();
	auto end = std::chrono::high_resolution_clock::now();
	double duration = std::chrono::duration<double>(end - start).count(); //seconds

	double bw = (pType->gridSize * pType->blockSize / (1024.0 * 1024.0 * 1024.0)) * NUM_ITER * sizeof(T) * pType->numAccess;
	bw /= duration;

	std::cout << "Read Size: " << (pType->gridSize * pType->blockSize * pType->numAccess) << " Duration (s): " << duration << " BW (Gb/s)" << bw << std::endl;
	pType->meas = bw;
		
	CUDA_ASSERT(cudaFree(d_ptr));
	CUDA_ASSERT(cudaFree(dummy));
}

