CC   = gcc
CPP  = g++
NVCC = nvcc

#NVCC_OPTIONS = --generate-code arch=compute_50,code=sm_50 \
#							 --generate-code arch=compute_52,code=sm_52 \
#							 --generate-code arch=compute_53,code=sm_53 \
#							 --generate-code arch=compute_60,code=sm_60 \
#							 --generate-code arch=compute_61,code=sm_61 \
#							 --generate-code arch=compute_62,code=sm_62 \
#							 --generate-code arch=compute_70,code=sm_70 \
#							 --generate-code arch=compute_72,code=sm_72 \
#							 --generate-code arch=compute_75,code=sm_75 \
#							 --generate-code arch=compute_80,code=sm_80 \
#               --generate-code arch=compute_86,code=sm_86
NVCC_OPTIONS = --generate-code arch=compute_75,code=sm_75

CPPFLAGS=-O3
#CPPFLAGS=-O3 -Xptxas=-v
CPPINCLUDE=-I./Common
#CPPINCLUDE=-I./Common  -I/usr/include/c++/9/
CFLAGS=
debug=
#debug= -g -D_DEBUG

APPS = hello_world hello_world2 vector_addition

TARGET = ${APPS}
LDFLAGS =

all: ${APPS}
build: ${APPS}

hello_world: p17_helloworld.cu
	${NVCC} ${NVCC_OPTIONS} ${debug} ${CPPINCLUDE} ${CPPFLAGS} -o $@ $^

hello_world2: p17_helloworld2.cu
	${NVCC} ${NVCC_OPTIONS} ${debug} ${CPPINCLUDE} ${CPPFLAGS} -o $@ $^
 
vector_addition: 01_vector_addtion.cu
	${NVCC} ${NVCC_OPTIONS} ${debug} ${CPPINCLUDE} ${CPPFLAGS} -o $@ $^
#
#%.o : $.cu
#	${NVCC} ${CPPINCLUDE} ${debug} ${CPPINCLUDE} ${NVCC_OPTIONS} -c $<

clean:
	rm -f ${APPS}
	rm -f *.o


