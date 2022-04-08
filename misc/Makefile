CC= gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error
NVCC_OPTIONS = -arch=sm_86
#NVCC_OPTIONS = -lcublas 

GNUOPENMP = -fopenmp
INTELOPENMP = -openmp

TARGET = cuda_add cuda_reduction_cpu\
	 cuda_reduction_gpu_ng cuda_reduction_gpu_ok
	

all : $(TARGET)

cuda_add : cuda_add.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_reduction_cpu : cuda_reduction_cpu.cpp
	$(CPP) $(GPP_OPTIONS) $^ -o $@

cuda_reduction_gpu_ng : cuda_reduction_gpu_ng.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_reduction_gpu_ok : cuda_reduction_gpu_ok.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

clean:
	/bin/rm -f *.o *.~ $(TARGET)
