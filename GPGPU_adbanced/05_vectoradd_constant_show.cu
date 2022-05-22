#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define N (8 * 1024) // 64kBに収める
#define Nbytes (N * sizeof(float))
#define NT (256)
#define NB (N / NT)

__constant__ float a[N], b[N];

__global__ void init(float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = 0.0f;
}

__global__ void add(float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
    printf("%f\n", c[i]);
}

__global__ void show(void)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d %f %f\n", i, a[i], b[i]); 
}


int main(void)
{
    float *c;
    float *host_a, *host_b, *host_c;
    int i;

    host_a = (float *)malloc(Nbytes);
    host_b = (float *)malloc(Nbytes);
    host_c = (float *)malloc(Nbytes);

    cudaMalloc((void **)&c, Nbytes);

    for (i = 0; i < N; i++)
    {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
    }

    cudaMemcpyToSymbol(a, host_b, Nbytes);
    cudaMemcpyToSymbol(b, host_b, Nbytes);

    puts("Init");
    init<<<NB, NT>>>(c);
    puts("Add");
    add<<<NB, NT>>>(c);
    puts("Show");
    show<<<NB, NT>>>();

    /*
    cudaMemcpy(host_c, c, Nbytes, cudaMemcpyDeviceToHost);  
    for (int i = 0; i < N; i++)
    {
        printf("%d %f\n", i, host_c[i]);
    }
    */

    return 0;
}

