#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

#include "../include/Benchmark.cuh"

void printDeviceProperties(int deviceIndex) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceIndex);
    
    checkError();
    std::cout << "Device " << deviceIndex << ": " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max thread dimensions: (" 
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max grid size: (" 
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
    std::cout << "  Multi-processor count: " << prop.multiProcessorCount << std::endl;
    std::cout << "  L2 cache size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "  Max register count per block: " << prop.regsPerBlock << std::endl;

    std::cout << "  Number of streaming multiprocessors (SMs): " << prop.multiProcessorCount << std::endl;
}

int main(int argc, char* argv[])
{

	if(argc < 3)
	{
		std::cout << "Expected Input <.exe> <i> <num_iter>" << std::endl;
		return 1;
	}
	
	printDeviceProperties(0);
	unsigned int i = (unsigned int)atoi(argv[1]);
	std::cout << "i " << i << std::endl;	
	size_t num_b = (1UL << 32); //defualt do 4 GB 
	param_t pType;
	pType.arrSize = (1UL << 10) << i; 
	pType.numAccess = num_b / pType.arrSize; 
	pType.blockSize = 512;
        pType.gridSize = 1024 * 4;
	std::cout << "Block Size " << pType.blockSize 
	<< "Grid Size " << pType.gridSize 
	<< "Array Size " << pType.arrSize
	<< "Num Access " << pType.numAccess << std::endl;	


	readBWTest<unsigned int>(&pType);	
	//std::cout << "Hello World!" << std::endl;
	return 0;
}
