#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#define TILE_WIDTH 16

__global__ void TiledMatrixMultKernel(float* M, float* N, float* P, int Width)
{
	 __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	 __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	 int bx = blockIdx.x; int by = blockIdx.y;
	 int tx = threadIdx.x; int ty = threadIdx.y;
	 int Row = by * TILE_WIDTH + ty;
	 int Col = bx * TILE_WIDTH + tx;
	 float Pvalue = 0;
	// Loop over the M and N tiles required to compute the P element
	for (int ph = 0; ph < (Width - 1)/TILE_WIDTH + 1; ++ph) {
	 // Collaborative loading of M and N tiles into shared memory
		 if (Row < Width && ph * TILE_WIDTH + tx < Width) {
			 ds_M[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
		 } else {
			 ds_M[ty][tx] = 0.0;
		 }

		 if (Col < Width && ph * TILE_WIDTH + ty < Width) {
			ds_N[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
		 } else {
			ds_N[ty][tx] = 0.0;
		 }

		 __syncthreads();
		 
		 for (int i = 0; i < TILE_WIDTH; ++i) { 
			Pvalue += ds_M[ty][i] * ds_N[i][tx];
		 }
		 __syncthreads();	
	 }
	if (Row < Width && Col < Width) {
		P[Row*Width+Col] = Pvalue;
	}
}

void generateMat(float *m, size_t size){
	int i, j;
	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++) {
			m[i*size+j] = rand() % 100;
		}		
	}
}

void printMat(float *m, size_t size) {
	int i, j;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			printf("%f ", m[i*size+j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char**argv){
	int k;	
	if (argc >= 1) {
		k = strtol(argv[1], NULL, 10);
	} else {
		return 0;		
	}

	srand(time(NULL));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	float * m, *n, *p;
	float * d_m, *d_p, *d_n;

	long size = k * k * sizeof(float);
	cudaMalloc((void**)&d_m, size);
	cudaMalloc((void**)&d_n, size);
	cudaMalloc((void**)&d_p, size);	

	m = (float *)malloc(size);
	n = (float *)malloc(size);
	p = (float *)malloc(size);

	generateMat(m, k);
	generateMat(n, k);	

	cudaMemcpy(d_m, m, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, n, size, cudaMemcpyHostToDevice);

	dim3 threadDims(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 blockDims(ceil(k * 1.0/TILE_WIDTH), ceil(k * 1.0/TILE_WIDTH), 1);	
	TiledMatrixMultKernel<<<blockDims, threadDims>>>(d_m, d_n, d_p, k);
	
	cudaThreadSynchronize();
	if (cudaGetLastError() != cudaSuccess) {
		printf("CUDA Error %d\n", cudaGetLastError());
		exit(-1);
	}
	
	cudaMemcpy(p, d_p, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("It took elapsed time of %f s\n", elapsedTime / 1000.0);

	free(n); free(m); free(p);
	cudaFree(d_n);
	cudaFree(d_m);
	cudaFree(d_p);
}



