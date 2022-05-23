#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include "lib.h"

// extern __constant__ float a;

int main(void)
{
	float host_a = 5.0f;
	set(host_a);
	// cudaMemcpyToSymbol(a, &host_a, sizeof(float));
	cudaDeviceSynchronize();

	show<<<1, 1>>>();
	cudaDeviceSynchronize();

	return 0;
}
