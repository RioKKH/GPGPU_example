#include <stdio.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>

#define N (1024*1024)
#define Nbytes (N*sizeof(float))

// __global__ : GPUカーネルの目印
__global__ void init(float *a, float *b, float *c)
{
    int i;

    for (i=0; i<N; i++)
    {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }
}

// __global__ : GPUカーネルの目印
__global__ void add(float *a, float *b, float *c)
{
    int i;

    for (i=0; i<N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    float *a, *b, *c;

    // GPUのメモリを確保
    cudaMalloc((void **)&a, Nbytes);
    cudaMalloc((void **)&b, Nbytes);
    cudaMalloc((void **)&c, Nbytes);

    // 並列実行の度合を指定
    init<<<1, 1>>>(a, b, c);
    add<<<1, 1>>>(a, b, c);
    cudaDeviceSynchronize();

    thrust::device_ptr<float> dev_c(c);
    float sum = thrust::reduce(dev_c, dev_c+N);
    float ave = sum / N;
    printf("%f, %f\n", sum, ave);

    // 確保したGPU上のメモリを解放
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
