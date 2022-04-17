#define SIZE 4096
#define CALC_COL 16
#define THREADX 256
#define THREADY 1
#define BLOCKX (SIZE / THREADX)
#define BLOCKY (SIZE / THREADY / CALC_COL) // block数が増えないように調整

// gpgpuprogramming10
// 1スレッドが複数列の点の計算を担当する
// さらに共有メモリによるデータの再利用版
__global__ void matmulGPU(float *A, float *B, float *C)
{
	int i, j, k, c, w;
	// レジスタを使う
	float sum[CALC_COL], A_cache; 
	int tx;

	// 共有メモリを使う
	__shared__ float sB[THREADX][CALC_COL];

	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * CALC_COL   + threadIdx.y; // blockDikm.yは1なので省略
	tx = threadIdx.x;

	for (c = 0; c < CALC_COL; c++)
	{
		sum[c] = 0.0f; // 結果を保持するレジスタの初期化
	}
	for (k = 0; k < SIZE; k+=THREADX) // k = 0
	{
		for (c = 0; c < CALC_COL; c++)
		{
			// 共有メモリへ代入
			sB[tx][c] = B[(k + tx) + SIZE*(j + c)];
		}
	}
	__syncthreads(); // 同期を取る

	// 1つのスレッドが複数の点を計算する
	for (w = 0; w < THREADX; w++)
	{
		A_cache = A[i + SIZE*(k+w)];
		for (c = 0; c < CALC_COL; c++)
		{
			sum[c] += A_cache * sB[w][c];
		}
	}
	__syncthreads();
	for (c = 0; c < CALC_COL; c++)
	{
		C[i + SIZE*(j + c)] = sum[c]; // 結果の書き出し
	}
}

