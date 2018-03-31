// Jin Pyo Jeon
// Lab 07
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#define T 1024 // Shared needs to be known at compile time??
#define N (1024 * 1024)

// Times for Reduced and non-reduced dot product
//	N			Reduced			Non-reduced		Thread Count
//  2^27		8.95			 8.91				1024
//  2^26		4.49			 4.46				1024
//  2^20		0.072			 0.072				1024

#define cudaCheckError() { \ 
	cudaError_t e = cudaGetLastError(); \
	if (e != cudaSuccess) { \
		printf("Cuda failed: %d: %s\n", __LINE__, cudaGetErrorString(e)); \
	} \ 
}

__global__ void calculateDot(int* a, int* b, unsigned long long int*c){
	__shared__ unsigned long long int partialSum[2 * T];
	
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;

	// printf("%d %d\n", start+t, start + blockDim.x + t);

	if (start + t <= N) { 
		partialSum[t] = a[start + t] * b[start + t];
		partialSum[blockDim.x+t] = a[start + blockDim.x+t] * 
								   b[start + blockDim.x+t];

		for (int stride = blockDim.x; stride > 0; stride /= 2) {
			__syncthreads(); 
			if (t < stride) {
				partialSum[t] += partialSum[t + stride];
			}
		}
		
		if (threadIdx.x == 0) atomicAdd(c, partialSum[0]);
	}
}

void random_ints(int * arr, size_t size){
	int i = 0;
	for (i = 0; i < size; i++) {
		arr[i] = rand() % 100;
	}
}

int main(int argc, char**argv) {
	srand(time(NULL));
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	int *a, *b;
    unsigned long long int *c, *d_c;
	int * d_a, *d_b;
	unsigned long long int size = N * sizeof(int);

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, sizeof(unsigned long long int));

	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (unsigned long long int *)malloc(sizeof(unsigned long long int));

	random_ints(a, N);
	random_ints(b, N);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	dim3 threadDims(T, 1, 1);
	dim3 blockDims(ceil(N / 2.0 / (float) T), 1, 1);
	calculateDot<<<blockDims, threadDims>>>(d_a, d_b, d_c);
	
	cudaCheckError()

	cudaMemcpy(c, d_c, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("The dot product is %llu with elapsed time of %f s\n", *c, elapsedTime / 1000.0);

	free(a); free(b); free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
