// Name: Jin Pyo Jeon
// CPSC375 
// Times:
// B		T		Time
// 100		16		13.84
// 100		32		7.43
// 100		40		6.27
// 100		64		3.86
// 100		128		3.1
// 100		256		2.92
// 100		1024	2.83
// 1000		32		5.99
// 1000		256		2.76
// 1000		512		2.73
// 1000		1024	2.83
// 1024		1024	2.75
// 10000	32		5.78
// 10000	128		2.74
// 10000	200		2.94
// 10000	256		2.56
// 10000	512		2.73
// 10000	1024	2.88
// 32768	126		2.66
// 32768	256		2.65
// 32768	512		2.68
// 65535	32		5.59
// 65535	128		2.64
// 65535	256		2.63
// 65535	400		2.92
// 65535	512		2.69
// 65535	768		3.0
// 65535	1024	3.8
// Discussion: From these experimental value, it seems that the optimal value for block size and 
// thread size is 10000 blocks and 256 threads per block. Beyond the most optimal, one thing that
// is evident is the fact that the optimal number of threads must be divisible by the warp size;
// in every instance where the thread number is not divisible by 32, the time suffered compared 
// to the times adjacent to it. Furthermore, it seems that the size of the number range that 
// each thread is assigned to does not correlate linearly for the most part .
// For example, the B/T pair (65535, 512) and (10000, 128) have similar times 
// despite the threads of first pair checking only 3 numbers and the latter around 78. 
// Furthermore, runs with small thread sizes suffered much more significant delay than others, 
// probably due to the fact that with small thread sizes ( t < 128 ), 8 blocks (per SM) 
// did not fill out the maximum number of threads possible (2048) and thus failed to fully 
// use the GPU.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N	100000000

__global__ void testCollatz(long n, long blockNum, long* counterEx) {

	long numPerBlock = ceil(n * 1.0 / blockNum);
	long numPerThread = ceil(n * 1.0 / blockNum / blockDim.x);

	long lowRange = (numPerBlock * blockIdx.x) + (threadIdx.x * numPerThread);
	long highRange = (numPerBlock * blockIdx.x) + ((threadIdx.x + 1) * numPerThread);

	long i;
	for (i = lowRange; i < highRange && i < N; i++) {
		long temp = i;
		int iteration = 0;
		if (temp == 0) continue;
		while (temp != 1) {
			iteration++;
			if (iteration >= 1000) { 
				*counterEx = i; 
				break;
			}
			if (temp % 2 == 0) temp = temp / 2;
			else temp = (3 * temp) + 1;
		}
	}
}

int main(int argc, char**argv){
	long B, T;
	long* h_counterEx, *d_counterEx;
	if (argc >= 2) {
		B = strtol(argv[1], NULL, 10);
		T = strtol(argv[2], NULL, 10);
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
	
	testCollatz<<<B,T>>>(N, B, d_counterEx); 

	cudaMemcpy(h_counterEx, d_counterEx, sizeof(long), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime = -1;
	cudaEventElapsedTime(&elapsedTime,start, stop);
	if (*h_counterEx == -1) {
		printf("Verifying %ld took %f s\n", (long) N, elapsedTime / 1000.0);
	} else {
		printf("Found a counterexample: %ld\n", *h_counterEx);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
