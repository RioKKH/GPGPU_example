#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define N 512


__global__ void device_add(int *a, int *b, int *c)
{
    // c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void host_add(int *a, int *b, int *c)
{
    for (int idx = 0; idx < N; idx++)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// basically just fills the array with index
void fill_array(int *data)
{
    for (int idx = 0; idx < N; idx++)
    {
        data[idx] = idx;
    }
}

void print_output(int *a, int *b, int *c)
{
    for (int idx = 0; idx < N; idx++)
    {
        printf("%d + %d = %d\n", a[idx], b[idx], c[idx]);
    }
}

int main(void)
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); fill_array(a);
    b = (int *)malloc(size); fill_array(b);
    c = (int *)malloc(size);

    // Alloc space for device copies of vector (a, b, c)
    cudaMalloc((void **)d_a, N * sizeof(int));
    cudaMalloc((void **)d_b, N * sizeof(int));
    cudaMalloc((void **)d_c, N * sizeof(int));

    // Copy from host to device
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // device_add<<<1, 1>>>(d_a, d_b, d_c);
    device_add<<<1, N>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    // host_add(a, b, c);

    print_output(a, b, c);
    free(a); free(b); free(c);

    // Free GPU memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
