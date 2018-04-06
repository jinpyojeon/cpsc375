// Jin Pyo Jeon
// Assign 7
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define T 1024 // Shared needs to be known at compile time??
#define B 65536
#define TB (ceil(B / T))
#define N (134217728)

// Times for Reduced and non-reduced dot product
//	N			Reduced			Non-reduced		Thread Count
//  2^27		8.95			 8.91				1024
//  2^26		4.49			 4.46				1024
//  2^20		0.072			 0.072				1024

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t e, const char *file, int line) {
	if (e != cudaSuccess) {
		fprintf(stderr, "GPU Assert: %s: %s %d\n", cudaGetErrorString(e), __FILE__, __LINE__);
	}
}

__global__ void generate_partials(int* X, int* Y, int *S){
	__shared__ int XY[T];
	
	int i = blockIdx.x * blockDim.x. + threadIdx.x; 

	// printf("%d %d\n", start+t, start + blockDim.x + t);

	if (i < N) {
		XY[threadIdx.x] = X[i];
	}
	
	for (int stride = 1; stride <= threadIdx.x; stride *= 2) {
		__syncthreads();
		int in1 = XY[threadIdx.x - stride];
		__syncthreads();
		XY[threadIdx.x] += in1;
	}
	__syncthreads();
	if (i < N) { Y[i] = XY[threadIdx.x]; }

	if (threadIdx.x == blockDim.x - 1) { S[i] = XY[i]; }

}

__global__ void add_partials(int *S) {
	__shared__ int PS[TB];
	
	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	if (i < B) { PS[threadIdx.x] = S[i]; }
	
	for (int stride = 1; stride <= threadIdx.x; stride *= 2) {
		__syncthreads();
		int in1 = PS[threadIdx.x - stride];
		__syncthreads();
		PS[threadIdx.x] += in1;
	}

	__syncthreads();
	if (i < B) { S[i] = PS[threadIdx.x]; }
}

__global__ void apply_partials(int *S, int *Y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (i < N) { Y[i] += S[blockIdx.x];	}
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
	
	int *X, *Y, *S;
	int *d_X, *d_Y, *d_S;
	unsigned long long int size = N * sizeof(int);

	cudaMalloc((void**)&d_X, size);
	cudaMalloc((void**)&d_Y, size);
	cudaMalloc((void**)&d_S, B * sizeof(int));

	X = (int *)malloc(size);
	Y = (int *)malloc(size);

	random_ints(X, N);

	cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
	
	dim3 threadDims(T, 1, 1);
	dim3 blockDims(ceil(N / (float) T), 1, 1);
	gpuAssert( generate_partials<<<blockDims, threadDims>>>(d_X, d_Y, d_S) );
	
	gpuAssert( add_partials<<<TB, T>>>(d_S) );
	
	gpuAssert( apply_partials<<<blockDims, threadDims >>>(d_S, d_Y) );

	gpuAssert( cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost) );

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("The  elapsed time of %f s\n", elapsedTime / 1000.0);

	free(X); free(Y); free(Z);
	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_Z);

	return 0;
}
