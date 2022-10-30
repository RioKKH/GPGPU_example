CC= gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++14 -Wall --pedantic-error
NVCC_OPTIONS = --generate-code arch=compute_75,code=sm_75 \
							 --generate-code arch=compute_80,code=sm_80 \
							 --generate-code arch=compute_86,code=sm_86 \

#NVCC_OPTIONS_CHAR4 = -arch=sm_86 -Xptxas -v -D_UCHAR4
#NVCC_OPTIONS_BGR = -arch=sm_86 -Xptxas -v -D_BGR
#NVCC_OPTIONS = -lcublas 

GNUOPENMP = -fopenmp
INTELOPENMP = -openmp

TARGET =  imageprocessing\
					imageprocessing_pinned\
					imageprocessing_mapped
	
all : $(TARGET)

imageprocessing : main.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@ -lm
	#$(NVCC) $(NVCC_OPTIONS_CHAR4) $^ -o $@ -lm

imageprocessing_pinned : main_pinned.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@ -lm

imageprocessing_mapped : main_mapped.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@ -lm

clean:
	/bin/rm -f *.o *.~ $(TARGET)
