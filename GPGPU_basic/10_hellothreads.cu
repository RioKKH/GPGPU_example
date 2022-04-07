#include <stdio.h>

__global__ void hello()
{
    printf("gridDim.x = %d, blockIdx.x=%d, blockdim.x=%d, threadIdx.x=%d\n",
            gridDim.x, blockIdx.x, blockDim.x, threadIdx.x);
    // gridDim グリッド内にあるブロックの数
    // blockIdx ブロック内に割り当てられた番号
    // blockDim ブロック内にあるスレッドの数
    // threadIdx スレッドに割り当てられた番号
}

int main(void)
{
    hello<<<1, 8>>>();
    cudaDeviceSynchronize();

    printf("\n");

    hello<<<8, 1>>>();
    cudaDeviceSynchronize();

    printf("\n");

    hello<<<4, 2>>>();
    cudaDeviceSynchronize();

    return 0;
}
