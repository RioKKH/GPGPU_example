#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#include "lib.h"

__constant__ int a;

void set(int host_a)
{
	cudaMemcpyToSymbol(a, &host_a, sizeof(host_a));
}

// __global__ void show(int b)
__global__ void show(void)
{
	printf("defice %d\n", a);
}
