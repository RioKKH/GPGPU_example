#include <iostream>

#define N (8 * 1024) // 64kBに収める
#define Nbytes (N * sizeof(float))
#define NT (256)
#define NB (N / NT)

__global__ void init(float *a, float *b, float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	a[i] = 1.0;
	b[i] = 2.0;
	c[i] = 0.0;
}

__global__ void add(float *a, float *b, float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

int main(void)
{
	float *a, *b, *c;
	// 経過時間保存用
	float elapsed_time_ms = 0.0f;
	// イベントを取り扱う変数
	cudaEvent_t start, end;

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaMalloc((void **)&a, Nbytes);
	cudaMalloc((void **)&b, Nbytes);
	cudaMalloc((void **)&c, Nbytes);

	init<<<NB, NT>>>(a, b, c);

	// イベント発生時間の記録
	cudaEventRecord(start, 0);
	add<<<NB, NT>>>(a, b, c);
	// イベント発生時間の記録
	cudaEventRecord(end, 0);

	// startとstopの時間を正しく記録し終わっている事を保証
	cudaEventSynchronize(end);

	// イベント間の時間差を計算
	cudaEventElapsedTime(&elapsed_time_ms, start, end);
	std::cout << "Elapsed time: " << elapsed_time_ms << std::endl;

	// イベントの破棄
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	
	return 0;
}
