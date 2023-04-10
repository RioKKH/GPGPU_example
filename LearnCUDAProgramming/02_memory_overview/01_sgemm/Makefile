CUDA_PATH=/app/CUDA/11.8
#CUDA_PATH=/app/CUDA/11.6u2
CUDA_SAMPLE_PATH=/app/CUDA/10.2/cuda-samples
#CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET = sgemm 
	 
	 
	 

INCLUDES= -I${CUDA_SAMPLE_PATH}/common/inc
NVCC_FLAGS=-m64 -lineinfo

IS_CUDA_11:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode arguments
SMS = 35 37 50 52 60 61 70 75
ifeq "$(IS_CUDA_11)" "1"
	SMS = 52 60 61 70 75 80 86
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all : ${TARGET}

sgemm: sgemm.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@ $<

.PHONY: clean

clean:
	rm -f ${TARGET}
	rm -f *.o




