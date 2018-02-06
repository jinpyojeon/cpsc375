# Makefile for CPSC 375
#
NVCC = nvcc
OPTION = -arch=sm_35
SRC = collatz2.cu
OBJ = collatz2.o 
GRD = ${SRC} Makefile 

collatz2: $(OBJ)
	$(NVCC) $(OPTION) -o collatz2 $(OBJ) 

collatz2.o: collatz2.cu
	$(NVCC) $(OPTION) -c collatz2.cu

clean:
	rm -f collatz2 $(OBJ)

grade:
	tar cvf - ${GRD} > assn02.tar
