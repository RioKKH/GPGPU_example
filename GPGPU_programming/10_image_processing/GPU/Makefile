CC= gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error
NVCC_OPTIONS = -arch=sm_86 -Xptxas -v
NVCC_OPTIONS_CHAR4 = -arch=sm_86 -Xptxas -v -D_UCHAR4
NVCC_OPTIONS_BGR = -arch=sm_86 -Xptxas -v -D_BGR
#NVCC_OPTIONS = -lcublas 

GNUOPENMP = -fopenmp
INTELOPENMP = -openmp

TARGET =  imageprocessing
	
all : $(TARGET)

imageprocessing : main.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@ -lm
	#$(NVCC) $(NVCC_OPTIONS_CHAR4) $^ -o $@ -lm

clean:
	/bin/rm -f *.o *.~ $(TARGET)
