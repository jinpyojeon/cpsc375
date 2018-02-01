// Jin Pyo Jeon
// Lab 02 
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

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
	} else return 0;
	clock_t t;
	t = clock();

	int *a, *b, *temp;
	unsigned long size = N * sizeof(int);

	a = (int *)malloc(size);
	b = (int *)malloc(size);
	temp = (int *)malloc(size);

	srand(time(NULL));

	random_ints(a, N);
	random_ints(b, N);
	int i = 0;
	for (i = 0; i < N; i++) {
		temp[i] = a[i] * b[i];
	}

	unsigned long sum = 0;
	for (i = 0; i < N; i++) {
		sum += temp[i];
	}

	t = clock() - t;
	double time_taken = ((double)t)/CLOCKS_PER_SEC;

	printf("The dot product is %lu with elapsed time of %f s\n", sum, time_taken);

	free(a); free(b); 

	return 0;
}
