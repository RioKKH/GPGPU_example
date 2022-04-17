#define SIZE 4096
#define CALC_COL 16
#define THREADX 256
#define THREADY 1
#define BLOCKX (SIZE / THREADX)
#define BLOCKY (SIZE / THREADY / CALC_COL) // block数が増えないように調整

// gpgpuprogramming10
// 1スレッドが複数列の点の計算を担当する
// これによって更に性能が改善する
__global__ void matmulGPU(float *A, float *B, float *C)
{
	int i, j, k, c;
	// レジスタを使う
	float sum[CALC_COL]; 

	// 共有メモリを使う
	__shared__ float sB[THREADX];

	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * CALC_COL   + threadIdx.y; // blockDikm.yは1なので省略

	for (c = 0; c < CALC_COL; c++)
	{
		sum[c] = 0.0f; // 結果を保持するレジスタの初期化
	}
	for (k = 0; k < SIZE; k++)
	{
		for (c = 0; c < CALC_COL; c++)
		{
			sum[c] += A[i + SIZE * k] * B[k + SIZE * (j + c)];
		}
	}
	for (c = 0; c < CALC_COL; c++)
	{
		C[i + SIZE*(j + c)] = sum[c]; // 結果の書き出し
	}
}

