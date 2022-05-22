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
    int i;
    EvoPrms *prms;
    prms = (EvoPrms *)malloc(sizeof(EvoPrms));
    set(prms);

    printf("host %d\n", prms->pop);
    printf("host %d\n", prms->chromosome);

    cudaMemcpyToSymbol(gpuevoprms, prms, sizeof(EvoPrms));

    show<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}

