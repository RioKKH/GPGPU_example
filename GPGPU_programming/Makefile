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

TARGET = 06_vectoradd 06_vectoradd_simple_constant_memory\
	 06_vectoradd_constant_memory_broadcast\
	 06_differentiate\
	 07_reduction0\
	 07_reduction1 07_reduction1.1 07_reduction1.2\
	 07_reduction2\
	 07_reduction3 07_reduction3.1 07_reduction3.2\
	 08_reduction4
	
all : $(TARGET)

06_vectoradd : 06_vectoradd.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

06_vectoradd_simple_constant_memory : 06_vectoradd_simple_constant_memory.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

06_vectoradd_constant_memory_broadcast : 06_vectoradd_constant_memory_broadcast.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

06_differentiate : 06_differentiate.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@ -lm

07_reduction0 : 07_reduction0.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

07_reduction1 : 07_reduction1.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

07_reduction1.1 : 07_reduction1.1.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

07_reduction1.2 : 07_reduction1.2.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

07_reduction2 : 07_reduction2.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

07_reduction3 : 07_reduction3.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

07_reduction3.1 : 07_reduction3.1.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

07_reduction3.2 : 07_reduction3.2.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

08_reduction4 : 08_reduction4.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@


clean:
	/bin/rm -f *.o *.~ $(TARGET)
