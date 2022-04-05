CC= gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error
NVCC_OPTIONS =
#NVCC_OPTIONS = -lcublas 

TARGET = thrust_example1 iterator thrust_vector_and_array\
	 thrust_vector_and_array_pointer\
	 thrust_vector_and_array_pointer2\
	 thrust_sort thrust_sort_by_key\
	 thrust_stable_sort

all : $(TARGET)

thrust_example1.o : thrust_example1.cu
	$(NVCC) $(NVCC_OPTIONS) -c $< -o $@

thrust_example1 : thrust_example1.o
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

iterator : iterator.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

thrust_vector_and_array : thrust_vector_and_array.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

thrust_vector_and_array_pointer : thrust_vector_and_array_pointer.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

thrust_vector_and_array_pointer2 : thrust_vector_and_array_pointer2.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

thrust_sort : thrust_sort.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

thrust_sort_by_key : thrust_sort_by_key.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

thrust_stable_sort : thrust_stable_sort.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

clean:
	/bin/rm -f *.o *.~ $(TARGET)
