// Jin Pyo Jeon
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

//	N				Stream		Non-Stream
// 3 * 2^15			0.11		0.11
// 3 * 2^10*700		0.15		0.15
// 3 * 2^20			0.22		0.22
// 3 * 2^24			3.45		3.46
// 3 * 2^25			6.89		6.90

#define N (3 * 1024 * 700)
#define T (N / 3 / 1024)

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
	}
}

__global__ void VecAddKernel(int* a, int* b, int* c, long width){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	
	c[x] = a[x] + b[x];
}

void random_ints(int * arr, size_t size){
	int i = 0;
	for (i = 0; i < size; i++) {
		arr[i] = rand() % 2;
	}
}

void printVec(int* arr, size_t size) {
	for (int i = 0; i < size; i++) {
		printf("%d ", arr[i]);
	}
	printf("\n");
}

int main(int argc, char**argv) {
	srand(time(NULL));
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int *h_A, *h_B, *h_C;
	long size = N * sizeof(int);

	cudaStream_t stream0, stream1, stream2;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	int *d_A0, *d_B0, *d_C0;
	int *d_A1, *d_B1, *d_C1;
	int *d_A2, *d_B2, *d_C2;

	int SegSize = N / 3;
	int SegSizeByte = sizeof(int) * SegSize;

	cudaMalloc((void**)&d_A0, SegSizeByte);
	cudaMalloc((void**)&d_B0, SegSizeByte);
	cudaMalloc((void**)&d_C0, SegSizeByte);
	cudaMalloc((void**)&d_A1, SegSizeByte);
	cudaMalloc((void**)&d_B1, SegSizeByte);
	cudaMalloc((void**)&d_C1, SegSizeByte);
	cudaMalloc((void**)&d_A2, SegSizeByte);
	cudaMalloc((void**)&d_B2, SegSizeByte);
	cudaMalloc((void**)&d_C2, SegSizeByte);
	
	//cudaMalloc((void**)&d_a, size);
	//cudaMalloc((void**)&d_b, size);
	//cudaMalloc((void**)&d_c, size);

	assert(SegSize % T == 0);
	assert(N % 3 == 0);

	h_A = (int *)malloc(size);
	h_B = (int *)malloc(size);
	h_C = (int *)malloc(size);

	random_ints(h_A, N);
	random_ints(h_B, N);

	for (int i = 0; i < N; i += SegSize * 3) {
		
		cudaMemcpyAsync(d_A0, h_A+i, SegSizeByte, cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B0, h_B+i, SegSizeByte, cudaMemcpyHostToDevice, stream0);

		if (i > 0) {  
			cudaMemcpyAsync(h_C+i-SegSize, d_C2, SegSizeByte, cudaMemcpyDeviceToHost, stream2);
		} 

		VecAddKernel<<<SegSize/T, T, 0, stream0>>>(d_A0, d_B0, d_C0, N);

		cudaMemcpyAsync(d_A1, h_A+i+SegSize, SegSizeByte,cudaMemcpyHostToDevice,stream1);
		cudaMemcpyAsync(d_B1, h_B+i+SegSize, SegSizeByte,cudaMemcpyHostToDevice,stream1);

		// 
		cudaMemcpyAsync(h_C+i, d_C0, SegSizeByte, cudaMemcpyDeviceToHost, stream0);

		VecAddKernel<<<SegSize/T, T, 0, stream1>>>(d_A1, d_B1, d_C1, N);

		cudaMemcpyAsync(d_A2, h_A+i+(2 * SegSize), SegSizeByte, cudaMemcpyHostToDevice,stream2);
		cudaMemcpyAsync(d_B2, h_B+i+(2 * SegSize), SegSizeByte, cudaMemcpyHostToDevice,stream2);
		//		

		cudaMemcpyAsync(h_C+i+SegSize, d_C1, SegSizeByte, cudaMemcpyDeviceToHost,stream1);	
		
		VecAddKernel<<<SegSize/T, T, 0, stream2>>>(d_A2, d_B2, d_C2, N);
	}
	cudaMemcpyAsync(h_C+(N-SegSize), d_C2, SegSizeByte, cudaMemcpyDeviceToHost, stream2);

	gpuErrCheck( cudaDeviceSynchronize() );
	
	//printVec(h_A, N);
	//printVec(h_B, N);	
	//printVec(h_C, N);	

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("The elapsed time is %fs\n", elapsedTime / 1000.0);

	free(h_A); free(h_B); free(h_C);

	return 0;
}
