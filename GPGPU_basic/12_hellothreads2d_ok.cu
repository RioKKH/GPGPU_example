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
    // dim3型変数block, threadを利用
    dim3 block(2, 4, 1);
    dim3 thread(4, 2, 1);
    hello<<<block, thread>>>();
    // 直接指定することも出来る
    // hello<<<dim3(2, 4, 1), dim3(4, 2, 1)>>>();
    cudaDeviceSynchronize();

    return 0;
}

