#include <iostream>
#include <stdlib.h>
#include <cassert>
#include <cuda_runtime.h>

int main(int argc, char* argv[])
{
    float *a_h, *b_h, *c_h; // host data
    float *a_d, *b_d, *c_d; // defice data
    int N = 10, nBytes, i;

    nBytes = N * sizeof(float);
    a_h = (float *)malloc(nBytes);
    b_h = (float *)malloc(nBytes);
    c_h = (float *)malloc(nBytes);

    cudaMalloc((void **)&a_d, nBytes);
    cudaMalloc((void **)&b_d, nBytes);
    cudaMalloc((void **)&c_d, nBytes);

    for (i = 0; i < N; ++i)
    {
        a_h[i] = 100.f + i;
        b_h[i] = 200.f + i;
    }

    cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, a_d, nBytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(c_d + 2, b_d + 3 , 4 * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(c_h, c_d, nBytes, cudaMemcpyDeviceToHost);

    for (i = 0; i < N; ++i)
    {
        std::cout << c_h[i] << std::endl;
        // assert(a_h[i] == b_h[i]);
    }

    free(a_h);
    free(b_h);
    free(c_h);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}

