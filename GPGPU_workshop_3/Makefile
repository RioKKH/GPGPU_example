CC= gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error
NVCC_OPTIONS = -lcublas 

TARGET = thrust_example1 iterator

all : $(TARGET)

thrust_example1.o : thrust_example1.cu
	$(NVCC) $(NVCC_OPTIONS) -c $< -o $@

thrust_example1 : thrust_example1.o
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

iterator : iterator.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

clean:
	/bin/rm -f *.o *.~ $(TARGET)
