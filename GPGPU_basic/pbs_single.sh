#!/bin/bash

#PBS -l select=1:ncpus=16
#PBS -j oe
#PBS -N vector_add_thread-job
#PBS -q SINGLE

cd ${PBS_O_WORKDIR}

export OMP_NUM_THREADS=16

#./08_vectoradd
#./08_vectoradd_malloc
./08_vector_add_thread
