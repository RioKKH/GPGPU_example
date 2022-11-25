#include <iostream>
#include <cstdio>
#include <numeric>

// cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// thrust
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>


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


__global__ void sort_using_cascading(const int& dev)
{
    // TODO
}

// show fit and id data on CPU
void show_host(thrust::host_vector<int> host_id,
               thrust::host_vector<int> host_fit)
{
    if (host_id.size() == host_fit.size())
    {
        for (int i = 0; i < host_id.size(); ++i)
        {
            printf("%d,%d\n", host_id[i], host_fit[i]);
        }
    }
}

// CPU
void make_initial_fitness(int *host, int POPSIZE, int CHROMOSOME)
{
    for (int i = 0; i < POPSIZE; ++i)
    {
        host[i] = rand() % CHROMOSOME;
    }
}


int main(int argc, char **argv)
{
    // const int blocks = 1;
    // const int threads = 5;
    // const int N = blocks * threads;

    int POPSIZE = 100;
    int CHROMOSOME = 128;

    if (argc == 2)
    {
        POPSIZE = std::atoi(argv[0]);
        CHROMOSOME = std::atoi(argv[1]);
    }

    thrust::host_vector  <int> host_fit(POPSIZE);
    thrust::device_vector<int> dev_fit(POPSIZE);
    thrust::host_vector  <int> host_id(POPSIZE);
    thrust::device_vector<int> dev_id(POPSIZE);

    // イニシャライズ
    make_initial_fitness(thrust::raw_pointer_cast(&host_fit[0]), POPSIZE, CHROMOSOME);
    thrust::sequence(host_id.begin(), host_id.end());
    printf("### PRE ###\n");
    show_host(host_id, host_fit);

    // コピー CPU --> GPU
    dev_fit = host_fit;
    dev_id  = host_id;

    // sort
    thrust::sort_by_key(dev_fit.begin(), dev_fit.end(), dev_id.begin());

    // コピー GPU --> CPU
    host_fit = dev_fit;
    host_id  = dev_id;
    printf("### POST ###\n");
    show_host(host_id, host_fit);

    // thrust::counting_iterator<int> ci(1);
    // thrust::device_vector<int> d_x(ci, ci + N);
    // thrust::device_vector<int> d_o(1);

    /*
    int *pdev = thrust::raw_pointer_cast(&dev[0]);

    reduction1_kernel<int, N> <<<blocks, threads>>> (
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_o.data())
            );

    thrust::copy(thrust::host, d_o.cbegin(), d_o.cend(), std::ostream_iterator<int>(std::cout, " "));
    */
    return 0;
}


