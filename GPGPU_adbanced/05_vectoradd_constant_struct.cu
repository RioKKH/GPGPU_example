#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define N (1 * 1024) // 64kBに収める
#define Nbytes (N * sizeof(float))
#define NT (256)
#define NB (N / NT)

struct EvoPrms
{
    int pop;
    int chromosome;
};

__constant__ float a[N], b[N];
__constant__ EvoPrms gpuevoprms;

__global__ void init(float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = 0.0f;
}

__global__ void add(float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void show()
{
    printf("%d\n", gpuevoprms.pop);
    printf("%d\n", gpuevoprms.chromosome);
}

void set(EvoPrms* prms)
{
    prms->pop = 1.0;
    prms->chromosome = 2.0;
}

int main(void)
{
    float *c;
    float *host_a, *host_b;
    int i;
    EvoPrms *prms;

    host_a = (float *)malloc(Nbytes);
    host_b = (float *)malloc(Nbytes);
    prms = (EvoPrms *)malloc(sizeof(EvoPrms));

    set(prms);

    cudaMalloc((void **)&c, Nbytes);

    for (i = 0; i < N; i++)
    {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
    }

    cudaMemcpyToSymbol(a, host_b, Nbytes);
    cudaMemcpyToSymbol(b, host_b, Nbytes);
    cudaMemcpyToSymbol(&gpuevoprms, prms, sizeof(EvoPrms));

    init<<<NB, NT>>>(c);
    cudaDeviceSynchronize();
    add<<<NB, NT>>>(c);
    cudaDeviceSynchronize();
    show<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}

