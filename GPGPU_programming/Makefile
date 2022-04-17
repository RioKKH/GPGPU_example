CC= gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error
NVCC_OPTIONS = -arch=sm_86 -Xptxas -v
#NVCC_OPTIONS = -lcublas 

GNUOPENMP = -fopenmp
INTELOPENMP = -openmp

TARGET = 05_differentiate 05_differentiate_shared_memory\
	 05_differentiate_constant_memory\
	 06_vectoradd 06_vectoradd_simple_constant_memory\
	 06_vectoradd_constant_memory_broadcast\
	 07_reduction0\
	 07_reduction1 07_reduction1.1 07_reduction1.2\
	 07_reduction2\
	 07_reduction3 07_reduction3.1 07_reduction3.2\
	 08_reduction4 08_reduction4_2 08_reduction5 08_reduction6\
	 08_reduction7 08_reduction8 08_reduction9 08_reduction9switch\
	 09_matmul_cpu 09_matmul2 09_matmul_gpu\
	 #mm1 mm2 mm3 mm4
	
all : $(TARGET)

05_differentiate : 05_differentiate.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@ -lm

05_differentiate_shared_memory : 05_differentiate_shared_memory.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@ -lm

05_differentiate_constant_memory : 05_differentiate_constant_memory.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@ -lm 06_vectoradd : 06_vectoradd.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

06_vectoradd_simple_constant_memory : 06_vectoradd_simple_constant_memory.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

06_vectoradd_constant_memory_broadcast : 06_vectoradd_constant_memory_broadcast.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

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

08_reduction4_2 : 08_reduction4_2.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

08_reduction5 : 08_reduction5.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

08_reduction6 : 08_reduction6.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

08_reduction7 : 08_reduction7.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

08_reduction8 : 08_reduction8.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

08_reduction9 : 08_reduction9.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

08_reduction9switch : 08_reduction9switch.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

09_matmul_cpu : 09_matmul.c 
	$(CC) $(CC_OPTIONS) $^ -o $@

09_matmul2 : 09_matmul2.c 
	$(CC) $(CC_OPTIONS) $^ -o $@

09_matmul_gpu : 09_matmul.cu 
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@


clean:
	/bin/rm -f *.o *.~ $(TARGET)
