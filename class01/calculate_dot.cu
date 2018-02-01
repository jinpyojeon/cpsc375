// Jin Pyo Jeon
// Lab 02
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#define T 1024 // Shared needs to be known at compile time??

__global__ void calculateDot(int N, int* a, int* b, unsigned long* result){
	__shared__ int temp[T];

	int lowRange = ceil(N / (T * 1.0)) * threadIdx.x;
	int highRange = ceil(N / (T * 1.0)) * (threadIdx.x + 1);
	unsigned long sum = 0;

	int i = lowRange;
	for (; i < highRange; i++) {
		sum += a[i] * b[i];
	}

	temp[threadIdx.x] = sum;

	__syncthreads();

	if (0 == threadIdx.x) {
		unsigned long sum = 0;
		for (int i = 0; i < T; i++) {
			sum += temp[i];
		}
		*result = sum;
	}

}

void random_ints(int * arr, size_t size){
	int i = 0;
	for (i = 0; i < size; i++) {
		arr[i] = rand() % 2;
	}
}

int main(int argc, char**argv) {
	unsigned long N;
	if (argc >= 2) {
		N = strtol(argv[1], NULL, 10);
	} else {
		return 0;
	}
	srand(time(NULL));
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	

	int *a, *b;
    unsigned long *c;
	int * d_a, *d_b;
    unsigned long	*d_c;
	int size = N * sizeof(int);

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, sizeof(unsigned long));

	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (unsigned long *)malloc(sizeof(unsigned long));


	random_ints(a, N);
	random_ints(b, N);


	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	calculateDot<<<1, T>>>(N, d_a, d_b, d_c);

	cudaMemcpy(c, d_c, sizeof(unsigned long), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("The dot product is %lu with elapsed time of %f s\n", *c, elapsedTime / 1000.0);


	free(a); free(b); free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
