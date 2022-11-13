CC= gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error
NVCC_OPTIONS = -arch=sm_75
#NVCC_OPTIONS = -arch=sm_52
#NVCC_OPTIONS = -arch=sm_50 --compiler-bindir "/usr/bin/gcc-9"
#NVCC_OPTIONS = -arch=sm_86
#NVCC_OPTIONS = -lcublas 

GNUOPENMP = -fopenmp
INTELOPENMP = -openmp

TARGET = cuda_add cuda_reduction_cpu\
	 cuda_reduction_gpu_ng cuda_reduction_gpu_ok\
	 cuda_index cuda_hellothreads cuda_init cuda_memcpy\
	 cuda_swap_memory cuda_swap_memory_malloc_in_kernel\
	 cuda_swap_memory_threads_in_kernel

	

all : $(TARGET)

cuda_init : cuda_init.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_add : cuda_add.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_index : cuda_index.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_reduction_cpu : cuda_reduction_cpu.cpp
	$(CPP) $(GPP_OPTIONS) $^ -o $@

cuda_reduction_gpu_ng : cuda_reduction_gpu_ng.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_reduction_gpu_ok : cuda_reduction_gpu_ok.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_hellothreads : cuda_hellothreads.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_memcpy : cuda_memcpy.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_swap_memory : cuda_swap_memory.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_swap_memory_malloc_in_kernel : cuda_swap_memory_malloc_in_kernel.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

cuda_swap_memory_threads_in_kernel : cuda_swap_memory_threads_in_kernel.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@


clean:
	/bin/rm -f *.o *.~ $(TARGET)
