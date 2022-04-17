#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mm2.cu" // ここを変更して行列-行列積のKernel切り替えする

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

	cudaMalloc((void **)&dA, Bytes);
	cudaMalloc((void **)&dB, Bytes);
	cudaMalloc((void **)&dC, Bytes);
	GPUresult = (float *)malloc(Bytes);

	cudaMemcpy(dA, hA, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dC, hC, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);

	dim3 Thread = dim3(THREADX, THREADY, 1); // 実行用パラメータの宣言と設定
	dim3 Block  = dim3(BLOCKX, BLOCKY, 1);   // 実行用パラメータの宣言と設定
	matmulGPU<<<Block, Thread>>>(dA, dB, dC); // 行列-行列積の実行

	cudaMemcpy(GPUresult, dC, SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	// ここ以降で結果が正しいかチェックする
	return 0;
}
