#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void testCollatz(long n, long blockSize, long* counterEx) {

	long lowRange = ceil(n * 1.0 / blockSize) * blockIdx.x;
	long highRange = ceil(n * 1.0 / blockSize) * (blockIdx.x + 1);

	long i;
	for (i = lowRange; i < highRange && i <= n; i++) {
		long temp = i;
		int iteration = 0;
		if (temp == 0) continue;
		while (temp != 1) {
			iteration++;
			if (iteration >= 1000) { 
				*counterEx = i; 
				break;
			}
			if (temp % 2 == 0) temp /= 2;
			else temp = (3 * temp) + 1;
		}
	}
}

int main(int argc, char**argv){

	long N, B;
	long* h_counterEx, *d_counterEx;
	if (argc >= 2) {
		N = strtol(argv[1], NULL, 10);
		B = strtol(argv[2], NULL, 10);
	} else {
		return -1;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	h_counterEx = (long*)malloc(sizeof(long));
	*h_counterEx = -1;
	cudaMalloc((void**) &d_counterEx, sizeof(long));
	cudaMemcpy(d_counterEx, h_counterEx, sizeof(long), cudaMemcpyHostToDevice);
	
	testCollatz<<<B,1>>>(N, B, d_counterEx); 

	cudaMemcpy(h_counterEx, d_counterEx, sizeof(long), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime = -1;
	cudaEventElapsedTime(&elapsedTime,start, stop);
	if (*h_counterEx == -1) {
		printf("Verifying %ld took %f s\n", N, elapsedTime / 1000.0);
	} else {
		printf("Found a counterexample: %ld\n", *h_counterEx);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
