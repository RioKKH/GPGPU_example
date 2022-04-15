#include <iostream>

#define N (8 * 1024) // 64kBに収める
#define Nbytes (N * sizeof(float))
#define NT (256)
#define NB (N / NT)

// オフチップメモリのコンスタントメモリ。カーネル外で定義する
// グローバルメモリとなる
// 各スレッドがコンスタントメモリの異なるアドレスにアクセスすると、
// グローバルメモリよりも遅くなる
__constant__ float a[N], b[N];

__global__ void init(float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	c[i] = 0.0f;
}

__global__ void add(float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	c[i] = a[i] + b[i];
}

int main(void)
{
	float *c;
	float *host_a, *host_b;
        int i;

	host_a = (float *)malloc(Nbytes);
	host_b = (float *)malloc(Nbytes);

	cudaMalloc((void **)&c, Nbytes);

	for (i=0; i<N; i++)
	{
		host_a[i] = 1.0f;
		host_b[i] = 2.0f;
	}

	// ホストからコンスタントメモリにデータをコピーする
	cudaMemcpyToSymbol(a, host_a, Nbytes);
	cudaMemcpyToSymbol(b, host_b, Nbytes);

	init<<<NB, NT>>>(c);
	add<<<NB, NT>>>(c);

	return 0;
}

