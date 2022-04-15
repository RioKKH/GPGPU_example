#include <iostream>

#define N (8 * 1024)
#define Nbytes (N * sizeof(float))
#define NT (256)
#define NB (N / NT)

__constant__ float a, b;

__global__ void init(float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = 0.0f;
}

__global__ void add(float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// broadcast!!
	c[i] = a + b;
}

int main(void)
{
	float *c;
	float host_a, host_b;

	host_a = 1.0f;
	host_b = 2.0f;

	cudaMalloc((void **)&c, Nbytes);

	// host_a, host_bが配列ではないのでアドレスを取り出すために&をつける
	cudaMemcpyToSymbol(a, &host_a, sizeof(float));
	cudaMemcpyToSymbol(b, &host_b, sizeof(float));

	init<<<NB, NT>>>(c);
	add<<<NB, NT>>>(c);

	return 0;
}
