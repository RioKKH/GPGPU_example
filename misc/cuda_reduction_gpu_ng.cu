// cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// thrust
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>

// C++
#include <iostream>

template <typename T>
__global__ void reduction1_kernel(const T* g_x, T* g_o)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    g_o[0] += g_x[i];
}

int main(void)
{
    const int blocks = 1;
    const int threads = 5;
    const int N = blocks * threads;

    thrust::counting_iterator<int> ci(1);
    thrust::device_vector<int> d_x(ci, ci + N);
    thrust::device_vector<int> d_o(1);

    reduction1_kernel<int> <<<blocks, threads>>> (
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_o.data())
            );

    thrust::copy(thrust::host, d_o.cbegin(), d_o.cend(), std::ostream_iterator<int>(std::cout, " "));
    return 0;
}


