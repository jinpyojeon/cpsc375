#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#define TILE_WIDTH 16

__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int j, int k, int l)
{
	 __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	 __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	 int bx = blockIdx.x; int by = blockIdx.y;
	 int tx = threadIdx.x; int ty = threadIdx.y;
	 int Row = by * TILE_WIDTH + ty;
	 int Col = bx * TILE_WIDTH + tx;
	 float Pvalue = 0;
	// Loop over the M and N tiles required to compute the P element
	for (int ph = 0; ph < (k - 1)/TILE_WIDTH + 1; ++ph) {
	 // Collaborative loading of M and N tiles into shared memory
		 if (Row < j && ph * TILE_WIDTH + tx < k) { 
			ds_M[ty][tx] = M[Row*k + ph*TILE_WIDTH + tx];
		 } else {
			ds_M[ty][tx] = 0.0;
		 }

		 if (Col < l && ph * TILE_WIDTH + ty < k) {
			ds_N[ty][tx] = N[(ph*TILE_WIDTH + ty)*l + Col];
		 } else {
			ds_N[ty][tx] = 0.0;
		 }

		 __syncthreads();

		 for (int i = 0; i < TILE_WIDTH; ++i) {
			 Pvalue += ds_M[ty][i] * ds_N[i][tx];
		 }
		 __syncthreads();
	 }	
	 if (Row < j && Col < l) {
		P[Row * l + Col] = Pvalue;
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
	int j,k,l;	
	if (argc >= 3) {
		j = strtol(argv[1], NULL, 10);
		k = strtol(argv[2], NULL, 10);
		l = strtol(argv[3], NULL, 10);
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

	long mSize = j * k * sizeof(float);
	long nSize = k * l * sizeof(float);
	long pSize = j * l * sizeof(float);
	
	cudaMalloc((void**)&d_m, mSize);
	cudaMalloc((void**)&d_n, nSize);
	cudaMalloc((void**)&d_p, pSize);	

	m = (float *)malloc(mSize);
	n = (float *)malloc(nSize);
	p = (float *)malloc(pSize);

	generateMat(m, j, k);
	generateMat(n, k, l);	
	
	cudaMemcpy(d_m, m, mSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, n, nSize, cudaMemcpyHostToDevice);

	dim3 threadDims(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 blockDims(ceil(j * 1.0/ TILE_WIDTH), ceil(l * 1.0/TILE_WIDTH), 1);
	TiledMatrixMulKernel<<<blockDims, threadDims>>>(d_m, d_n, d_p, j, k, l);

	cudaThreadSynchronize();
	cudaMemcpy(p, d_p, pSize, cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess) { 
		printf("Error %d\n", cudaGetLastError()); 
		exit(-1);
	}  

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("The elapsed time is %f s\n", elapsedTime / 1000.0);

	free(n); free(m); free(p);
	cudaFree(d_n);
	cudaFree(d_m);
	cudaFree(d_p);
}



