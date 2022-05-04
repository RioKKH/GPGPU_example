#include <stdio.h>

#define N 3

__device__ int forloop()
{
	int i = 0;
	for (i = 0; i < N; ++i)
	{
		printf("%d\n", i);
	}
	return i;
}


__global__ void hello()
{
	int i = forloop();
	// printf("%d %d %d %d\n", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x);
	printf("i:%d\n", i);
}

int main()
{
	hello<<<1, 5>>>();
	cudaDeviceSynchronize();

	return 0;
}



