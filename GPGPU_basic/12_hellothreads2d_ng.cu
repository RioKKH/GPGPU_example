#include <stdio.h>

__global__ void hello()
{
    printf("gridDim.x=%d, blockIdx.x=%d, blockDim.x=%d, threadIdx.x=%d\n",
            gridDim.x, blockIdx.x, blockDim.x, threadIdx.x);
    printf("gridDim.y=%d, blockIdx.y=%d, blockDim.y=%d, threadIdx.y=%d\n",
            gridDim.y, blockIdx.y, blockDim.y, threadIdx.y);
}

int main(void)
{
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}

