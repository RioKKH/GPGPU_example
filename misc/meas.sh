#!/bin/bash

LIST="32 64 128 256 512 1024"

for l in ${LIST}; do
	./cuda_thrust_sort t ${l} ${l} 8 > ${l}_${l}_8_thrust.csv
	./cuda_thrust_sort p ${l} ${l} 8 > ${l}_${l}_8_pseudo.csv
done

#nsys profile \
#	   -t cuda,osrt,nvtx,cudnn,cublas \
#		 --stats=true \
#		 -f true \
#		 -o ${1} \
#		 true ./cuda_thrust_sort 1024 1024 8
#		 #true ./gpuonemax

# -t : APIs to be tracked
# --stats : Outputs profiling information similar to nvprof
# -f : Overview the outputs
# -o : Output filename
