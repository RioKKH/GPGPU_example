#include <stdio.h>
#include <iostream>

__global__ void hello()
{
    printf("Hello Thread\n");
}

int main(void)
{
    hello<<<8, 1>>>();
    cudaDeviceSynchronize();

    std::cout << std::endl;

    hello<<<1, 8>>>();
    cudaDeviceSynchronize();

    std::cout << std::endl;

    hello<<<4, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}

