#include <iostream>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

void showPopulation(int *population, int N)
{
    for (int i = 0; i < N; ++i)
    {
        printf("%d", population[i]);
    }
    printf("\n");
}

__global__
void swapPopulation(int *parent, int *offspring)
{
    std::uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx == 1)
    {
        for (int i = 0; i < 256000; ++i)
        {
            parent[i] = offspring[i];
        }
    }
}


int main()
{
    int *parent_host, *offspring_host;
    int *parent_dev, *offspring_dev, *temp_dev;

    float elapsed_time = 0.0f;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int N = 256000;
    int nBytes = N * sizeof(float);
    
    parent_host = (int *)malloc(nBytes);
    offspring_host = (int *)malloc(nBytes);

    cudaMalloc((void **)&parent_dev, nBytes);
    cudaMalloc((void **)&offspring_dev, nBytes);
    // cudaMalloc((void **)&temp_dev, nBytes);

    // initialize host data
    for (int i = 0; i < N; ++i)
    {
        parent_host[i] = i;
        offspring_host[i] = i * 10;
    }

    printf("\npost-parent\n");
    // showPopulation(parent_host, N);
    printf("\npost-offspring\n");
    // showPopulation(offspring_host, N);

    cudaEventRecord(start, 0);
    // Host to Device
    cudaMemcpy(parent_dev,    parent_host,    nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(offspring_dev, offspring_host, nBytes, cudaMemcpyHostToDevice);

    dim3 blocks;
    dim3 threads;
    const int WARP = 32;
    blocks.x = 1;
    // blocks.x = N / 32;
    blocks.y = 1;
    blocks.z = 1;
    threads.x = 1;
    threads.y = 1;
    threads.z = 1;
    // swap memories between parents and offsprings
    // temp_dev = parent_dev;
    // parent_dev = offspring_dev;
    // offspring_dev = temp_dev;
    swapPopulation<<<blocks, threads>>>(parent_dev, offspring_dev);
    cudaDeviceSynchronize();

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    // Device to Host
    cudaMemcpy(parent_host,    parent_dev,    nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(offspring_host, offspring_dev, nBytes, cudaMemcpyDeviceToHost);

    printf("ElapsedTime: %f\n", elapsed_time);
    printf("\npost-parent\n");
    // showPopulation(parent_host, N);
    printf("\npost-offspring\n");
    // showPopulation(offspring_host, N);

    // free memories
    free(parent_host);
    free(offspring_host);
    cudaFree(parent_dev);
    cudaFree(offspring_dev);
}

