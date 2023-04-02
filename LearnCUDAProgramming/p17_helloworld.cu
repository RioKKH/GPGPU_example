#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

__global__ void print_from_gpu(void)
{
    printf("Hello world! from thread [%d, %d] From device\n",
            threadIdx.x, blockIdx.x);
}

int main(int argc, char** argv)
{
    printf("Hello world from host!\n");
    if (argc == 3)
    {
        print_from_gpu<<<atoi(argv[1]), atoi(argv[2])>>>();
        cudaDeviceSynchronize();
    }
    else
    {
        printf("Exit\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}
