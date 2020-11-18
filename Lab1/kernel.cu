#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#ifndef BLOC_SIZE
#define BLOCK_SIZE 512
#endif

#include <device_functions.h>

#include <iostream>
#include <numeric>
// includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
using namespace std;


__global__ void reduce7(int * input, int n)
{
	extern __shared__ int Data[];
	int tid = threadIdx.x;
	int i = blockIdx.x*(BLOCK_SIZE*2) + tid;
	int gridSize = BLOCK_SIZE*2*gridDim.x;
	Data[tid] = 0;
	do{
		Data[tid] += input[i] + input[i+BLOCK_SIZE];
		i += gridSize;
	  }
	while (i < n);
	__syncthreads();
	if (BLOCK_SIZE >= 512)
	{
		if (tid < 256) {Data[tid] += Data[tid + 256];}
		__syncthreads();
	}
	if (BLOCK_SIZE >= 256)
	{
		if (tid < 128) { Data[tid] += Data[tid + 128];}
	    __syncthreads();
	}
    if (BLOCK_SIZE >= 128)
	{
    	if (tid < 64) {Data[tid] += Data[tid + 64];}
	__syncthreads();
	}
	//}
   if (tid < 32){
	if (BLOCK_SIZE >= 64) Data[tid] += Data[tid + 32];
	if (BLOCK_SIZE >= 32) Data[tid] += Data[tid + 16];
	if (BLOCK_SIZE >= 16) Data[tid] += Data[tid + 8];
	if (BLOCK_SIZE >= 8) Data[tid] += Data[tid + 4];
	if (BLOCK_SIZE >= 4) Data[tid] += Data[tid + 2];
	if (BLOCK_SIZE >= 2) Data[tid] += Data[tid + 1];
	}
    if (tid == 0) input[blockIdx.x] = Data[0];
  }

__global__ void reduce5(int* inData, int* outData)
{
   __shared__ int data [1024];
	int tid = threadIdx.x;
	int i = 2*blockIdx.x*blockDim.x+threadIdx.x;
		// Записать сумму первых двух элементов в разделяемую память
	data[tid] = inData[i]+inData[i+blockDim.x];
	__syncthreads(); // дождаться загрузки данных
	for (int s = blockDim.x/2; s>32; s>>1)
	{
		if(tid<s)
		{
			data[tid]+=data[tid+s];
		}
		__syncthreads();
	}
	if(tid<32) // развернуть последние итерации
	{
		data[tid]+=data[tid+32];
		data[tid]+=data[tid+16];
		data[tid]+=data[tid+8];
		data[tid]+=data[tid+4];
		data[tid]+=data[tid+2];
		data[tid]+=data[tid+1];
	}

	if(tid==0)
		outData[blockIdx.x]=data[0];
}

int reduce(int* data, int n)
{
	int* sums =NULL;
	int numBlocks = n/512;
	int res = 0;

	// выделить память под массив сумм блоков
	cudaMalloc((void**)&sums, numBlocks*sizeof(int));
	// провести поблочную редукцию, записав
	// суммы для каждого блока в массив sums
	reduce5<<<dim3(numBlocks),dim3(512)>>>(data,sums);

	if(numBlocks>512)
		res += reduce(sums,numBlocks);
	else
	{
		// Если значений мало, то просуммируем явно
		int* sumsHost = new int[numBlocks];
		cudaMemcpy(sumsHost,sums,numBlocks*sizeof(int),cudaMemcpyDeviceToHost);
		for(int i = 0; i<numBlocks;i++)
			res+=sumsHost[i];
		delete[] sumsHost;
	}
	cudaFree(sums);
	return res;
}


__global__ void sum(int* input)
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
		  if(threadIdx.x < 16) input[threadIdx.x] += input[threadIdx.x+16];
		  if(threadIdx.x < 8) input[threadIdx.x] += input[threadIdx.x+8];
		  if(threadIdx.x < 4) input[threadIdx.x] += input[threadIdx.x+4];
		  if(threadIdx.x < 2) input[threadIdx.x] += input[threadIdx.x+2];
		  if(threadIdx.x == 0) input[threadIdx.x] += input[threadIdx.x+1];
}

int main()
{

    tryAgain: // это лейбл
	
    srand(time(NULL));          //зерно рандома
    int i;                    //для цикла

	int n;
	int *h;
	printf("Input array size: ");
    scanf("%d",&n);             //задаем размер

	h = (int*)malloc(n * sizeof(int));
    
    for(i=0;i<n;i++)            //запоняем рандомом
	{
		h[i]=rand()%1999+1899;
		// cout << " " << h[i] << endl;
	}
        
	const auto count = n;
	const int size = count * sizeof(int);

	int* d;

	auto elapsedTimeInMsGPU1 = 0.0f;
    auto elapsedTimeInMsGPU = 0.0f;
	float elapsedTimeInMsCPU = 0.0f;
	StopWatchInterface *timer = NULL;
		StopWatchInterface *timer1 = NULL;

	//GPU restart
	cudaDeviceReset();

	//Entry point to mesure time
	cudaEvent_t start, stop;
	//GPU timer
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));



	//start timer
	checkCudaErrors(cudaEventRecord(start, 0));


	cudaMalloc(&d, size);
	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

			//SDK timer
	sdkCreateTimer(&timer1);
	sdkStartTimer(&timer1);	
	
	reduce7 <<<1, count / 2 >>>(d,n);
	
	//elapsedTimeInMsGPU1 = reduce(h,n);

	    sdkStopTimer(&timer1);
	
	
	int result;
	cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);
	
	//Stop the timer
	checkCudaErrors(cudaEventRecord(stop, 0));

	

	// make sure GPU has finished copying
	checkCudaErrors(cudaDeviceSynchronize());

	//Finish point to mesure time
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMsGPU, start, stop));
	elapsedTimeInMsGPU = sdkGetTimerValue(&timer1);
	
	printf("Execution time in ms via GPU timer %f\n", elapsedTimeInMsGPU);

	cout << "Sum(GPU) is " << result << endl;

		//SDK timer
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);	
	
	result = 0;
	for (int i = 0; i < count; i++)
		result += h[i];

    sdkStopTimer(&timer);
	elapsedTimeInMsCPU = sdkGetTimerValue(&timer);
	
	printf("Execution time in ms via CPU timer %f\n", elapsedTimeInMsCPU);

	cout << "Sum(CPU) is " << result << endl;

	
	getchar();

	cudaFree(d);
	delete[] h;

	goto tryAgain; // а это оператор goto
	
	return 0;
}