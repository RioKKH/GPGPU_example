#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdio>
#include <numeric>
#include <thread>

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


__global__ void pseudo_elisism(const int* dev, int* eliteIdx)
{
    int numOfEliteIdx    = blockIdx.x;
    int localFitnessIdx  = threadIdx.x;
    int globalFitnessIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const int OFFSET     = blockDim.x;

    extern __shared__ volatile int s_fitness[];

    s_fitness[localFitnessIdx] = dev[globalFitnessIdx];
    s_fitness[localFitnessIdx + OFFSET] = globalFitnessIdx;
    __syncthreads();

    for (int stride = OFFSET/2; stride >= 1; stride >>= 1)
    {
        if (localFitnessIdx < stride)
        {
            unsigned int index = (s_fitness[localFitnessIdx] >= s_fitness[localFitnessIdx + stride])
                ? localFitnessIdx : localFitnessIdx + stride;
            s_fitness[localFitnessIdx]          = s_fitness[index];
            s_fitness[localFitnessIdx + OFFSET] = s_fitness[index + OFFSET];
        }
        __syncthreads();
    }
    if (localFitnessIdx == 0 && blockIdx.x < gridDim.x)
    {
        eliteIdx[numOfEliteIdx] = s_fitness[localFitnessIdx + OFFSET];
    }
}

__host__ void thrust_sort(thrust::host_vector<int> dev_fit,
                          thrust::host_vector<int> dev_id)
{
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    thrust::sort_by_key(dev_fit.begin(), dev_fit.end(), dev_id.begin());
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

    int NUMOFTEST = 100;
    float elapsed_time = 0.0f;
    cudaEvent_t start, end;

    // parameters for genetic operator
    int POPSIZE    = 100;
    int CHROMOSOME = 128;
    int ELITESIZE  = 8;

    // CUDA threads settings
    dim3 blocks;
    dim3 threads;

    // argument parser
    if (argc == 4)
    {
        // std::cout << argv[1] << "," << argv[2] << std::endl;
        POPSIZE = std::atoi(argv[1]);
        CHROMOSOME = std::atoi(argv[2]);
        ELITESIZE = std::atoi(argv[3]);
    }

    std::cout << POPSIZE << "," << CHROMOSOME << std::endl;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // host
    thrust::host_vector  <int> host_fit(POPSIZE);
    thrust::host_vector  <int> host_id(POPSIZE);

    thrust::host_vector  <int> result_fit(POPSIZE);
    thrust::host_vector  <int> result_id(POPSIZE);

    // device
    thrust::device_vector<int> dev_fit(POPSIZE);
    thrust::device_vector<int> dev_id(POPSIZE);

    thrust::device_vector<int> dev_eliteId(ELITESIZE);

    printf("### PRE THRUST###\n");
    for (int i = 0; i < NUMOFTEST; ++i)
    {
        // イニシャライズ
        make_initial_fitness(thrust::raw_pointer_cast(&host_fit[0]), POPSIZE, CHROMOSOME);
        thrust::sequence(host_id.begin(), host_id.end());
        // show_host(host_id, host_fit);

        // コピー CPU --> GPU
        dev_fit = host_fit;
        dev_id  = host_id;

        // sort
        cudaEventRecord(start, 0);
        // thrust::sort_by_key(dev_fit.begin(), dev_fit.end(), dev_id.begin());
        thrust_sort(dev_fit, dev_id);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, start, end);
        printf("Elapsed Time(thrust sort),%d,%f\n", i, elapsed_time);

        // コピー GPU --> CPU
        result_fit = dev_fit;
        result_id  = dev_id;
        // show_host(host_id, host_fit);
    }
    printf("### POST THRUST###\n");

    blocks.x = ELITESIZE;
    blocks.y = 1;
    blocks.z = 1;

    threads.x = POPSIZE / ELITESIZE;
    threads.y = 1;
    threads.z = 1;

    printf("### PRE PSEUDO ELITISM###\n");
    for (int i = 0; i < NUMOFTEST; ++i)
    {
        make_initial_fitness(thrust::raw_pointer_cast(&host_fit[0]), POPSIZE, CHROMOSOME);
        // コピー CPU --> GPU
        dev_fit = host_fit;
        // dev_id  = host_id; pseudo_elisismではhost_idは使わない

        cudaEventRecord(start, 0);
        pseudo_elisism<<<blocks, threads, threads.x * 2 * sizeof(int)>>>
            (thrust::raw_pointer_cast(&dev_fit[0]), 
            thrust::raw_pointer_cast(&dev_eliteId[0]));
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, start, end);
        printf("Elapsed Time(pseudo_elisism) %f\n", elapsed_time);

        // コピー GPU --> CPU
        result_fit = dev_fit;
        result_id  = dev_id;
    }
    printf("### POST THRUST###\n");
        // show_host(host_id, host_fit);

    // thrust::counting_iterator<int> ci(1);
    // thrust::device_vector<int> d_x(ci, ci + N);
    // thrust::device_vector<int> d_o(1);

    /*
    int *pdev = thrust::raw_pointer_cast(&dev[0]);

    
    printf("### POST PSEUDO ELITISM###\n");

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


