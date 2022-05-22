CC = gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error -D_DEBUG
NVCC_OPTIONS = -arch=sm_52 -D_DEBUG
#NVCC_OPTIONS = -arch=sm_86 -D_DEBUG

GNUOPENMP = -fopenmp
INTELOPEMP = -openmp

APPS = 05_vectoradd 05_vectoradd_constant 05_vectoradd_broadcast

TARGET = ${APPS}

all: ${APPS}
build: ${APPS}

05_vectoradd : 05_vectoradd.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

05_vectoradd_constant : 05_vectoradd_constant.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

05_vectoradd_broadcast : 05_vectoradd_broadcast.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@


