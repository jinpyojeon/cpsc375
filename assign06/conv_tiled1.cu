#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

// N			Time
// 1 << 10		0.0009
// 1 << 15		0.0024
// 1 << 17		0.011
// 1 << 18		0.017
// 1 << 20		0.049
// 1 << 21		0.084

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	}
}

#define O_TILE_WIDTH	16
#define MASK_WIDTH		5
#define BLOCK_WIDTH		(O_TILE_WIDTH + (MASK_WIDTH - 1))

__constant__ float M[MASK_WIDTH]; 

__global__ void convolution_2D_basic_kernel(float *N, float *P, long Width) {
	__shared__ float N_ds[BLOCK_WIDTH];

	int index_i = blockIdx.x * blockDim.x + threadIdx.x;

	int j;
	if ((index_i >= 0) && (index_i < Width)) {
		N_ds[threadIdx.x + (MASK_WIDTH / 2)] = N[index_i];
	
		// Repeating border for edge cases 	
		if (threadIdx.x < (MASK_WIDTH / 2)) {
			if (index_i - (MASK_WIDTH / 2) <= 0) N_ds[threadIdx.x] = N[index_i];
			else N_ds[threadIdx.x] = N[index_i - (MASK_WIDTH / 2)]; 
		}

		if (threadIdx.x > (MASK_WIDTH / 2)) {
			if (index_i + (MASK_WIDTH / 2) >= Width - 1) N_ds[threadIdx.x] = N[Width - 1];
			else N_ds[threadIdx.x] = N[index_i + (MASK_WIDTH / 2)];
		}
		
		// printf("Copying %d %d %f\n", index_i, threadIdx.x,  N_ds[threadIdx.x]);
	} else {
		// N_ds[threadIdx.x] = 0.0f;
	}
	
	float output = 0.0f;
	if (threadIdx.x < O_TILE_WIDTH) {
		for (j = 0; j < MASK_WIDTH; j++) {
			output += M[j] * N_ds[j + threadIdx.x];
		}
		// printf("%d %f\n", blockIdx.x * O_TILE_WIDTH + threadIdx.x, output);
		P[blockIdx.x * O_TILE_WIDTH + threadIdx.x] = output;
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
	long width = 1<<18;
	
	srand(time(NULL));

	float mask[MASK_WIDTH];
	for (int i = 0; i < MASK_WIDTH; i++) {
		mask[i] = 1.0/MASK_WIDTH;
	}

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

	generateMat(n, 1, width);
	// printMat(n, 1, width);

	gpuErrCheck( cudaMemcpy(d_n, n, nSize, cudaMemcpyHostToDevice) )

	gpuErrCheck( cudaMemcpyToSymbol(M, &mask, mSize) );

	dim3 blockDims(O_TILE_WIDTH,1,1);

	int blockNum = ((width-1)/(O_TILE_WIDTH))+ 1;
	dim3 gridDims(blockNum, 1, 1);

	convolution_2D_basic_kernel<<<gridDims, blockDims>>>(d_n, d_p, width);

	gpuErrCheck( cudaPeekAtLastError() );	
	gpuErrCheck( cudaDeviceSynchronize() );
	
	gpuErrCheck( cudaMemcpy(p, d_p, pSize, cudaMemcpyDeviceToHost) );
	
	// printMat(p, 1, width);

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



