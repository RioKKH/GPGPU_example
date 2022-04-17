#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include "mm5.cu" // ここを変更して行列-行列積のKernel切り替えする

#define SIZE 4096
#define Bytes (SIZE * SIZE * sizeof(float))
#define MILLISEC_PER_SEC 1000
#define FLOPS_TO_GFLOPS 1e-9


int main(void)
{
	float *dA, *dB, *dC;
	float *GPUresult; // GPUでの計算結果を受け取る用

	float *hA, *hB, *hC;
	int i, j, k;

	clock_t start_c, stop_c;
	float time_s, time_ms;
	float gflops;


	hA = (float *)malloc(Bytes);
	hB = (float *)malloc(Bytes);
	hC = (float *)malloc(Bytes);

	for (i = 0; i < SIZE; i++)
	{
		for (k = 0; k < SIZE; k++)
		{
			hA[i * SIZE + k] = (float)(i + 1) * 0.1f;
		}
	}
	for (k = 0; k < SIZE; k++)
	{
		for (j = 0; j < SIZE; j++)
		{
			hB[k * SIZE + j] = (float)(j + 1) * 0.1f;
		}
	}
	for (i = 0; i < SIZE; i++)
	{
		for (j = 0; j < SIZE; j++)
		{
			hC[i * SIZE + j] = 0.0f;
		}
	}

	cudaMalloc((void **)&dA, Bytes);
	cudaMalloc((void **)&dB, Bytes);
	cudaMalloc((void **)&dC, Bytes);
	GPUresult = (float *)malloc(Bytes);

	cudaMemcpy(dA, hA, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dC, hC, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);

	dim3 Thread = dim3(THREADX, THREADY, 1); // 実行用パラメータの宣言と設定
	dim3 Block  = dim3(BLOCKX, BLOCKY, 1);   // 実行用パラメータの宣言と設定

	start_c = clock();
	matmulGPU<<<Block, Thread>>>(dA, dB, dC); // 行列-行列積の実行
	stop_c = clock();

	time_s = (stop_c - start_c) / (float)CLOCKS_PER_SEC;
	time_ms = time_s * MILLISEC_PER_SEC;
	gflops = (2.0 * SIZE * SIZE * SIZE / time_s) * FLOPS_TO_GFLOPS;

	printf("%f ms\n", time_ms);
	printf("%f GFLOPS\n", gflops);

	cudaMemcpy(GPUresult, dC, SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	// ここ以降で結果が正しいかチェックする
	return 0;
}
