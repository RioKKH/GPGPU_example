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

template <typename T, int N>
__global__ void reduction1_kernel(const T* g_x, T* g_o)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // ブロック内のスレッド数分の共有メモリ領域を確保
    // 共有メモリはブロック内のスレッドが共有出来るメモリ
    __shared__ T s_x[N];
    // グローバルメモリg_xから共有メモリs_xへ値を転送する
    s_x[tid] = (i < N) ? g_x[i] : T{};
    __syncthreads(); // 同一ブロク内のスレッドに対するバリア

    // 共有メモリの0番目に合計された値が格納されている
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if ((tid % (2 * s)) == 0)
        {
            s_x[tid] += s_x[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_o[blockIdx.x] = s_x[0];
    }
}

int main(void)
{
    const int blocks = 1;
    const int threads = 5;
    const int N = blocks * threads;

    thrust::counting_iterator<int> ci(1);
    thrust::device_vector<int> d_x(ci, ci + N);
    thrust::device_vector<int> d_o(1);

    reduction1_kernel<int, N> <<<blocks, threads>>> (
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_o.data())
            );

    thrust::copy(thrust::host, d_o.cbegin(), d_o.cend(), std::ostream_iterator<int>(std::cout, " "));
    return 0;
}


