#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

#include <iostream>
#include <numeric>
// includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
using namespace std;

__global__ void sum(long long* input)
{
	/*const int tid = threadIdx.x;
	auto step_size = 1;
	int number_of_threads = blockDim.x;
	while (number_of_threads > 0)
	{
		if (tid < number_of_threads) 
		{
			const auto fst = tid * step_size * 2;
			const auto snd = fst + step_size;
			input[fst] += input[snd];
		}
		step_size <<= 1; 
		number_of_threads >>= 1;
	}
	  __syncthreads();
	*/
	
	  if(threadIdx.x < 2048) input[threadIdx.x] += input[threadIdx.x+2048];
	  __syncthreads();
	  if(threadIdx.x < 1024) input[threadIdx.x] += input[threadIdx.x+1024];
	  __syncthreads();
	  if(threadIdx.x < 512) input[threadIdx.x] += input[threadIdx.x+512];
	  __syncthreads();	
	  if(threadIdx.x < 256) input[threadIdx.x] += input[threadIdx.x+256];
	  __syncthreads();
	  if(threadIdx.x < 128) input[threadIdx.x] += input[threadIdx.x+128];
	  __syncthreads();
		  if(threadIdx.x < 64) input[threadIdx.x] += input[threadIdx.x+64];
	  __syncthreads();
		  if(threadIdx.x < 32) input[threadIdx.x] += input[threadIdx.x+32];
	  __syncthreads();
		  if(threadIdx.x < 16) input[threadIdx.x] += input[threadIdx.x+16];
	  __syncthreads();
		  if(threadIdx.x < 8) input[threadIdx.x] += input[threadIdx.x+8];
	  __syncthreads();
		  if(threadIdx.x < 4) input[threadIdx.x] += input[threadIdx.x+4];
	  __syncthreads();
		  if(threadIdx.x < 2) input[threadIdx.x] += input[threadIdx.x+2];
	  __syncthreads();
		  if(threadIdx.x == 0) input[threadIdx.x] += input[threadIdx.x+1];
	  __syncthreads();
}

int main()
{

    tryAgain: // это лейбл
	
    srand(time(NULL));          //зерно рандома
    int i,n;                    //для цикла
	long long *h;
	printf("Input array size: ");
    scanf("%d",&n);             //задаем размер
    //int h[n];
	h = (long long*)malloc(n * sizeof(long long));
    
    for(i=0;i<n;i++)            //запоняем рандомом
	{
		h[i]=rand()%1699999+1699995;
		// cout << " " << h[i] << endl;
	}
        
	const auto count = n;
	const long long size = count * sizeof(long long);

	long long* d;

    auto elapsedTimeInMsGPU = 0.0f;
	float elapsedTimeInMsCPU = 0.0f;
	StopWatchInterface *timer = NULL;

	//GPU restart
	cudaDeviceReset();

	//Entry point to mesure time
	cudaEvent_t start, stop;
	//GPU timer
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	//SDK timer
	sdkCreateTimer(&timer);

	//start timer
	checkCudaErrors(cudaEventRecord(start, 0));
	sdkStartTimer(&timer);	

	cudaMalloc(&d, size);
	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

	sum <<<1, count / 2 >>>(d);

	long long result;
	cudaMemcpy(&result, d, sizeof(long long), cudaMemcpyDeviceToHost);
	
	//Stop the timer
	checkCudaErrors(cudaEventRecord(stop, 0));
		sdkStopTimer(&timer);
		elapsedTimeInMsCPU = sdkGetTimerValue(&timer);
	

	// make sure GPU has finished copying
	checkCudaErrors(cudaDeviceSynchronize());

	//Finish point to mesure time
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMsGPU, start, stop));
	
	printf("Execution time in ms via GPU timer %f\n", elapsedTimeInMsGPU);

	cout << "Sum(GPU) is " << result << endl;

	
	result = 0;
	for (int i = 0; i < count; i++)
		result += h[i];

	printf("Execution time in ms via CPU timer %f\n", elapsedTimeInMsCPU);

	cout << "Sum(CPU) is " << result << endl;

	
	getchar();

	cudaFree(d);
	delete[] h;

	goto tryAgain; // а это оператор goto
	
	return 0;
}