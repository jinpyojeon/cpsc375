// Jin Pyo Jeon
// Lab 02
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#define T 1024 // Shared needs to be known at compile time??
#define B 128
#define N (1024 * 1024)

#define cudaCheckError() { \ 
	cudaError_t e = cudaGetLastError(); \
	if (e != cudaSuccess) { \
		printf("Cuda failed: %d: %s\n", __LINE__, cudaGetErrorString(e)); \
	} \ 
}

__global__ void calculateDot(int* a, int* b, unsigned long long int*c){
	__shared__ unsigned long long int temp[T];
	
	temp[threadIdx.x] = 0; 

	unsigned long long int sum = 0;
	
	int i = threadIdx.x + (blockIdx.x * T);
	int stride = T * B; 
	
	for (; i < N; i+=stride) {
		temp[threadIdx.x] += a[i] * b[i];
	}

	if (threadIdx.x == 0) {
	   for (i = 0; i < T; i++) {	
			sum += temp[i];
	   }
	   atomicAdd(c, sum);
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
	dim3 blockDims(B, 1, 1);
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
