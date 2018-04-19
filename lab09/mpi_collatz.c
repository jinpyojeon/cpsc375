#include <mpi.h>
#include <math.h>
#include <stdio.h>

typedef int bool;
#define true 1
#define false 0

int main(int argc, char *argv[]){
	int my_rank, comm_sz, N, p; 
	bool allSuccess;

	if (argc >= 1) {
		N = strtol(argv[1], NULL, 10);
	} else {
		return -1;
	}

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	p = comm_sz; 

	int lowRange = ((N / p) + 1) * my_rank;
	int highRange = ((N/ p) + 1) * (my_rank+ 1);
	int counterEx = -1;
	
	// Presuming rank 0 process participates as well
	for (int i = lowRange; i <= highRange || i <= N; i++) {
		int temp = i;
		int iteration = 0;
		if (temp == 0) continue;
		while (temp != 1) {
			iteration++;
			if (iteration >= 1000) {
				counterEx = i;
				break;
			}
			if (temp % 2 == 0) temp = temp / 2;
			else temp = (3 * temp) + 1;
		}
		if (counterEx != -1) { break; }
	}
	
	if (my_rank != 0) {
		// printf("Verified from %d to %d\n", lowRange, highRange);
		MPI_Send(&counterEx, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); 
	} else {
		allSuccess = true;
		for (int i = 1; i < comm_sz; i++) {
			MPI_Recv(&counterEx, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (counterEx != -1) {
				allSuccess = false;
			}
		}
	}

	
	if (my_rank == 0) {
		if (allSuccess) {
			printf("Verified Collatz on %d\n", N);	
		} else {
			printf("Found counterexample: %d\n", counterEx); 
		}
	}

	MPI_Finalize();


	return 0;
}
