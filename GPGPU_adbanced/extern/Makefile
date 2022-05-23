NVCC = nvcc
# NVCC_OPTIONS = -arch=sm_86
CUDA_ARCH = --generate-code arch=compute_75,code=sm_75 \
            --generate-code arch=compute_50,code=sm_50 \
            --generate-code arch=compute_52,code=sm_52 \
            --generate-code arch=compute_53,code=sm_53 \
            --generate-code arch=compute_60,code=sm_60 \
            --generate-code arch=compute_61,code=sm_61 \
            --generate-code arch=compute_62,code=sm_62 \
            --generate-code arch=compute_70,code=sm_70 \
            --generate-code arch=compute_72,code=sm_72 \
            --generate-code arch=compute_75,code=sm_75 \
            --generate-code arch=compute_80,code=sm_80 \
            --generate-code arch=compute_86,code=sm_86

NVCC_OPTIONS=--device-c ${CUDA_ARCH} 

TARGET=test
all: ${TARGET}

%.o : %.cu
	$(NVCC) $(NVCC_OPTIONS) -o $@ -c $<

test: main.cu lib.o
	$(NVCC) $^ -o $@

clean:
	rm -f ${TARGET}
	rm -f *.o
