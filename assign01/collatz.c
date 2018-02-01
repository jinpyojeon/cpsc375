#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

bool startCollatz(long n) {
	int iterator = 0;
	while (n != 1) {
		iterator++;
		if (iterator == 1000) { return false; }
		if (n % 2 == 0) n /= 2;
		else n = (3 * n) + 1;
	}
}

void testCollatz(long n) {
	clock_t start, end;
	bool allSuccess = true;
	int counterExample = -1;

	start = clock();
	long i = 0;
	for (i = n; i > 0; i--) {
		// fprintf(stderr, "%ld\n", i);
		allSuccess = allSuccess && startCollatz(n);
		if (allSuccess == false) {
			counterExample = i;		
		}
	}

	end = clock();
	
	if (allSuccess) {
		double time = ((double) (end - start)) / CLOCKS_PER_SEC;
		fprintf(stderr, "%ld took %f time\n", n, time);
	} else {
		fprintf(stderr, "Found a counterexample: %d\n", counterExample);
	}
}


int main(int argc, char**argv){
	long N;
	if (argc >= 1) {
		N = strtol(argv[1], NULL, 10);
	} else {
		return -1;
	}

	testCollatz(N);

	return 0;
}
