#include <cuda_runtime.h>
#include <stdio.h>

#define Nx (1024)
#define Ny (1024)
#define Nbytes (Nx * Ny * sizeof(float))
#define NTx (16)
#define NTy (16)
#define NBx (Nx/NTx)
#define NBy (Ny/NTy)

__global__ void init(float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ij = i + Nx * j;      // faster
    // int ij = i * Ny + j;   // slower

    a[ij] = 1.0;
    a[ij] = 2.0;
    a[ij] = 0.0;
}

__global__ void add(float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ij = i + Nx * j;      // faster
    // int ij = i * Ny + j;   // slower

    c[ij] = a[ij] + b[ij];
}

int main(void)
{
    float *a, *b, *c;
    dim3 thread(NTx, NTy, 1);
    dim3  block(NBx, NBy, 1);

    cudaMalloc((void **)&a, Nbytes);
    cudaMalloc((void **)&b, Nbytes);
    cudaMalloc((void **)&c, Nbytes);

    init<<<block, thread>>>(a, b, c);
     add<<<block, thread>>>(a, b, c);

     cudaFree(a);
     cudaFree(b);
     cudaFree(c);

     return 0;
}

