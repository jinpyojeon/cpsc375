// Jin Pyo Jeon

// Times 
// N			Thread/Block	seconds	
// 1 << 24		512				0.60
// 1 << 24		480				0.61
// 1 << 24		272				0.61
// 1 << 22		128				0.15
// 1 << 20		32				0.05
// 1 << 20		64				0.047
// 1 << 20		128				0.048
// 1 << 18		32				0.02
// 1 << 17		32				0.013

#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define MASK_WIDTH 5

__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, long Width) {
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float pValue = 0;
	int nStartPoint = i - (MASK_WIDTH / 2);

	if (i < Width) {
		for (int j = 0; j < MASK_WIDTH; j++) {
			if (nStartPoint + j >= 0 && nStartPoint + j < Width) {
				pValue += N[nStartPoint + j] * M[j];
			}
		}
	
		P[i] = pValue;	
	}
}

void generateMat(float *m, size_t height, size_t width){
	int i, j;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++) {
			m[i*width+j] = rand() % 100;
		}		
	}
}

void printMat(float *m, size_t height, size_t width) {
	int i, j;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++) {
			printf("%f ", m[i*width+j]);
		}		
		printf("\n");
	}
	printf("\n");
}




int main(int argc, char**argv){
	long width = 1 << 24;
	int THREAD_COUNT = 17; // Due to seeming Grid Dim x limitation of 65536	
	
	srand(time(NULL));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	float * m, *n, *p;
	float * d_m, *d_p, *d_n;

	long mSize = MASK_WIDTH * sizeof(float);
	long nSize = width * sizeof(float);
	long pSize = width * sizeof(float);
	
	cudaMalloc((void**)&d_m, mSize);
	cudaMalloc((void**)&d_n, nSize);
	cudaMalloc((void**)&d_p, pSize);	

	m = (float *)malloc(mSize);
	n = (float *)malloc(nSize);
	p = (float *)malloc(pSize);

	for (int i = 0; i < MASK_WIDTH; i++) {
		m[i] = 1.0/MASK_WIDTH; // averaging mask
	}

	generateMat(n, 1, width);	
	
	cudaMemcpy(d_m, m, mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, n, nSize, cudaMemcpyHostToDevice);
	
	cudaError err = cudaGetLastError();
	if (err != cudaSuccess) { 
		printf("%d: Error %d %s\n", __LINE__, err, cudaGetErrorString(err)); 
		exit(-1);
	}

	long blocks = ceil(width / (float) THREAD_COUNT);
	while (blocks >= 65535) {
		THREAD_COUNT *= 2;
		blocks = ceil(width / (float) THREAD_COUNT);
	}
	assert(THREAD_COUNT <= 1024);

	dim3 DimBlock(THREAD_COUNT, 1, 1);
	dim3 DimGrid(blocks, 1, 1);
	convolution_1D_basic_kernel<<<DimGrid, DimBlock>>>(d_n, d_m, d_p, width);

	err = cudaGetLastError();
	if (err != cudaSuccess) { 
		printf("%d: Error %d %s\n", __LINE__, err, cudaGetErrorString(err)); 
		exit(-1);
	}	


	cudaMemcpy(p, d_p, pSize, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("The elapsed time is %f s with %d threads/block\n", elapsedTime / 1000.0, THREAD_COUNT);

	free(n); free(m); free(p);
	cudaFree(d_n);
	cudaFree(d_m);
	cudaFree(d_p);
}



