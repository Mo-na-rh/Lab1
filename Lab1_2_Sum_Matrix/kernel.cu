
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BLOCK_DIM 2 //������ ����������
int M, K;

using namespace std;

__global__ void matrixAdd (int *A, int *B, int *C, int M, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
 
    int index = col * M + row;
 
    //�������� �� GPU
    if (col < M && row < K) 
	{ 
        C[index] = A[index] + B[index];
    }
}

int main()
{
    cout << "M: ";
    cin >> M;
    cout << "K: ";
    cin >> K;
 
    int *A = new int [M*K];
 
 
    int *B = new int [M*K];
 
 
    int *C = new int [M*K];
 
 
    //���������� ������
    for(int i=0; i<M; i++)
        for (int j=0; j<K; j++){
            A[i*K+j] = 3;
            B[i*K+j] = 1;
            C[i*K+j] = 0;
        }
 
    int *dev_a, *dev_b, *dev_c; //��������� �� ���������� ������
 
    int size = M * K * sizeof(int); //���������� ������
 
    cudaMalloc((void**)&dev_a, size); //��������� ������
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);
 
    cudaMemcpy(dev_a, A, size, cudaMemcpyHostToDevice); //����������� �� GPU
    cudaMemcpy(dev_b, B, size, cudaMemcpyHostToDevice);
 
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); //����� ���������� ������
    dim3 dimGrid((M+dimBlock.x-1)/dimBlock.x, (K+dimBlock.y-1)/dimBlock.y); //������ � ����������� �����
    printf("dimGrid.x = %d, dimGrid.y = %d\n", dimGrid.x, dimGrid.y); //��������� ������ �����
 
    matrixAdd<<<dimGrid,dimBlock>>>(dev_a, dev_b, dev_c, M, K); //����� ����
    cudaDeviceSynchronize(); //�����������
    
    cudaMemcpy(C, dev_c, size, cudaMemcpyDeviceToHost);
 
    //�����    ����������
    printf("Result Matrix C:\n");
    for(int i=0; i<M; i++){ 
        for (int j=0; j<K; j++){
            printf("%d\t", C[i] );
        }
        printf("\n");
    }
 
 
    cudaFree(dev_a); //������������ ������
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
