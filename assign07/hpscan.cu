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
#define TB (B / T)
#define N (134217728)

// Times for Reduced and non-reduced dot product
//	N			Reduced			Non-reduced		Thread Count
//  2^27		8.95			 8.91				1024
//  2^26		4.49			 4.46				1024
//  2^20		0.072			 0.072				1024

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t e, const char *file, int line) {
	if (e != cudaSuccess) {
		fprintf(stderr, "GPU Assert: %s: %s %d\n", cudaGetErrorString(e), file, line);
	}
}

__global__ void generate_partials(int* X, int* Y, int *S) {
	__shared__ int XY[T];
	
	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	if (i < N) { XY[threadIdx.x] = X[i]; }
	
	for (int stride = 1; stride <= threadIdx.x; stride *= 2) {
		__syncthreads();
		int in1 = XY[threadIdx.x - stride];
		__syncthreads();
		XY[threadIdx.x] += in1;
	}
	__syncthreads();
	
	if (i < N) { Y[i] = XY[threadIdx.x]; }

	if (threadIdx.x == blockDim.x - 1) { S[blockIdx.x] = XY[threadIdx.x]; }

}

__global__ void add_partials(int *S) {
	__shared__ int PS[T];
	
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
		arr[i] = rand() % 2	;
	}
}

void printVec(int *vec, size_t width) {
	for (int i = 0; i < width; i++) {
		printf("%d ", vec[i]);
	}
	printf("\n");
}

void printVec(int *vec, long begin, long end) {
	printf("[START: %ld]  ", begin);
	for (long i = begin; i < end; i++) {
		printf("%d ", vec[i]);
	}
	printf("[END: %ld] \n", end);
}

void printVecElement(int *vec, long index) { 
	printf("%d ", vec[index]);
}

long addVec(int *vec, long begin, long end) {
	long sum = 0;
	for (long i = begin; i < end; i++) {
		sum += vec[i];
	}
	return sum;
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

	gpuErrCheck( cudaMalloc((void**)&d_X, size) );
	gpuErrCheck( cudaMalloc((void**)&d_Y, size) );
	cudaMalloc((void**)&d_S, B * sizeof(int));

	X = (int *)malloc(size);
	Y = (int *)malloc(size);
	S = (int *)malloc(sizeof(int) * B);

	random_ints(X, N);

	gpuErrCheck( cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice) );
	
	dim3 threadDims(T, 1, 1);
	dim3 blockDims(B, 1, 1);
	generate_partials<<<blockDims, threadDims>>>(d_X, d_Y, d_S);

	gpuErrCheck( cudaPeekAtLastError() );
	gpuErrCheck( cudaDeviceSynchronize() );
	
	// Test
	gpuErrCheck( cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost) );	
	gpuErrCheck( cudaMemcpy(S, d_S, sizeof(int) * B, cudaMemcpyDeviceToHost) );

	for (long i = 0; i < 10; i++) {
		printVecElement(S, i);
		printVecElement(Y, ((i + 1) * T - 1));
		printf("\n");
	}

	printVec(S, 20);
	// 

	add_partials<<<TB, T>>>(d_S);
	
	gpuErrCheck( cudaPeekAtLastError() );
	gpuErrCheck( cudaDeviceSynchronize() );

	// Test	
	gpuErrCheck( cudaMemcpy(S, d_S, sizeof(int) * B, cudaMemcpyDeviceToHost) );
	printVec(S, 20);


	for (long i = 0; i < 10; i++) {
		printVec(Y, i * T, i * T + 10);
	}
	// 

	apply_partials<<<blockDims, threadDims>>>(d_S, d_Y);


	gpuErrCheck( cudaPeekAtLastError() );
	gpuErrCheck( cudaDeviceSynchronize() );

	gpuErrCheck( cudaMemcpy(Y, d_Y, size, cudaMemcpyDeviceToHost) );

	// Test
	for (long i = 0; i < 10; i++) {
		printVecElement(S, i);
		printVec(Y, i * T, i * T + 10);
	}
	//	

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("The  elapsed time of %f s\n", elapsedTime / 1000.0);

	free(X); free(Y); free(S);
	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_S);

	return 0;
}
