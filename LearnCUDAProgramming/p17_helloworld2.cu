#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

__global__ void print_from_gpu(void)
{
    printf("Hello world! from thread [%d, %d] From device\n",
            threadIdx.x, blockIdx.x);
}

int main(void)
{
    printf("Hello world from host!\n");
    cudaDeviceSynchronize();
    print_from_gpu<<<1, 1>>>();
    return 0;
}
