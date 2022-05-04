// cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// thrust
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>

// c++
#include <iostream>
#include <stdio.h>

const int TOURNAMENTSELECTION = 3;

template <class T>
__global__ void kernel(T* g_x)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	printf("%d,%d,%d,%d\n", blockDim.x, blockIdx.x, threadIdx.x, i);
	// g_x[i] = i;
}

__device__ int devfunc(int idx)
{
	int ii = 0;
	for (int i = 0; i < TOURNAMENTSELECTION; ++i)
	{
		printf("idx: %d\n", i);
		ii = i;
	}
	return ii;
}

__global__ void kernel2()
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("%d\n", idx);
	// int ii = devfunc(idx);
	// printf("%d %d %d %d %d\n", blockDim.x, blockIdx.x, threadIdx.x, idx, ii);
}

int main(void)
{
    const int blocks = 8;
    const int threads = 3;
    const int N = blocks * threads;

    // thrust::device_vector<int> d_o(blocks * threads);
    // kernel<int><<<blocks, threads>>>(thrust::raw_pointer_cast(d_o.data()));
    // thrust::copy(thrust::host, d_o.cbegin(), d_o.cend(), std::ostream_iterator<int>(std::cout, " "));

	kernel2<<<blocks, threads>>>();
    return 0;
}
