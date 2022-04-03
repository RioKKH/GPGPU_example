CC= gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error
NVCC_OPTIONS = -lcublas

TARGET = cg_cpu cublas_gemv

all : $(TARGET)

cg_cpu : cg_cpu.c
	$(CC) $(GCC_OPTIONS) $< -o $@

cublas_gemv.o : cublas_gemv.cu
	$(NVCC) $(NVCC_OPTIONS) -c $< -o $@

cublas_gemv : cublas_gemv.o
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

clean:
	/bin/rm -f *.o *.~ $(TARGET)
