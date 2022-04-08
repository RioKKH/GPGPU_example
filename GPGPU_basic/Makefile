CC= gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error
NVCC_OPTIONS = -arch=sm_86
#NVCC_OPTIONS = -lcublas 

GNUOPENMP = -fopenmp
INTELOPENMP = -openmp

TARGET = 08_vectoradd 08_vectoradd_malloc 08_vector_add_thread\
	 10_hellokernel 10_hellothread 10_hellothread_2 10_hellothreads\
	 11_vectoradd_cpu 11_vectoradd_malloc_cpu 11_vectoradd_1thread\
	 12_imageproc1d_cpu 12_imageproc1d_gpu\
	 12_vectoradd2d_cpu 12_vectoradd2d_1darray_cpu\
	 12_hellothreads2d_ng 12_hellothreads2d_ok
	 

all : $(TARGET)

08_vectoradd : 08_vectoradd.c
	$(CC) $(GCC_OPTIONS) $(GNUOPENMP) $^ -o $@

08_vectoradd_malloc : 08_vectoradd_malloc.c
	$(CC) $(GCC_OPTIONS) $(GNUOPENMP) $^ -o $@

08_vector_add_thread : 08_vector_add_thread.cpp
	$(CC) $(GCC_OPTIONS) $(GNUOPENMP) $^ -o $@

10_hellokernel : 10_hellokernel.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

10_hellothread : 10_hellothread.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

10_hellothread_2 : 10_hellothread_2.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

10_hellothreads : 10_hellothreads.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

11_vectoradd_cpu : 11_vectoradd_cpu.c
	$(CC) $(GCC_OPTIONS) $^ -o $@

11_vectoradd_malloc_cpu : 11_vectoradd_malloc_cpu.c
	$(CC) $(GCC_OPTIONS) $^ -o $@

11_vectoradd_1thread : 11_vectoradd_1thread.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

12_imageproc1d_cpu : 12_imageproc1d.c
	$(CC) $(GCC_OPTIONS) $^ -o $@

12_imageproc1d_gpu : 12_imageproc1d.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

12_vectoradd2d_cpu : 12_vectoradd2d.c
	$(CC) $(GCC_OPTIONS) $^ -o $@

12_vectoradd2d_1darray_cpu : 12_vectoradd2d_1darray.c
	$(CC) $(GCC_OPTIONS) $^ -o $@

12_hellothreads2d_ng : 12_hellothreads2d_ng.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

12_hellothreads2d_ok : 12_hellothreads2d_ok.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

clean:
	/bin/rm -f *.o *.~ $(TARGET)
