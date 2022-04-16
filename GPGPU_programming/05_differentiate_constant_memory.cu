#include <iostream>
#include <stdlib.h>
#include <math.h>

#define Lx (2.0 * M_PI)
#define Nx (1024 * 1024)
#define dx (Lx / (Nx - 1))
#define Nbytes (Nx * sizeof(double))
#define NT (256)
#define NB (Nx / NT)

__constant__ double idx2;

void init(double *u)
{
	int i;
	for (i=0; i<Nx; ++i)
	{
		u[i] = sin(i * dx);
	}
}

__global__ void differentiate(double *u, double *dudx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// 共有メモリの宣言と代入
	// 右と左の袖領域を追加して宣言
	__shared__ double su[1 + NT + 1];
	int tx = threadIdx.x + 1;

	su[tx] = u[i];
	__syncthreads();

	// メモリの話をしているのか、スレッドの話をしているのかがこんがらがる

	// 袖領域の処理
	// 左のブロックの一番右端のグローバルメモリから1つ右隣のブロックの左端
	// の共有メモリに値をコピーする
	if (blockIdx.x > 0 		&& threadIdx.x == 0)
	{
		su[tx - 1] = u[i - 1];
	}
	// 袖領域の処理
	// 1つ右のブロックの一番左端のグローバルメモリから左隣のブロクの
	// 右端の共有メモリに値をコピーする
	if (blockIdx.x < gridDim.x - 1	&& threadIdx.x == blockDim.x - 1)
	{
		su[tx + 1] = u[i + 1];
	}
	// 境界条件の処理
	// blockIdx.x == 0の一番左端の共有メモリにアサインする値を作成する
	if (blockIdx.x == 0		&& threadIdx.x == 0)
	{
		su[tx - 1] = 3.0*su[tx] - 3.0*su[tx + 1] + su[tx + 2];
	}
	// 境界条件の処理
	// blockIdx.x == girdDim.x -1の一番右端の共有メモリにアサインする値を作成する
	// つまり全体でも一番右端の共有メモリとなる
	if (blockIdx.x == gridDim.x - 1	&& threadIdx.x == blockIdx.x - 1)
	{
		su[tx + 1] = 3.0*su[tx] - 3.0*su[tx - 1] + su[tx - 2];
	}
	__syncthreads();
	// 共有メモリに値の読み込み完了

	// 全スレッドが中心差分を計算
	dudx[i] = (su[tx + 1] - su[tx - 1]) * idx2;
}

int main(void)
{
	double *host_u, *host_dudx;
	double *u, *dudx;
	float elapsed_time_ms = 0.0f; // 経過時間保存用

	// イベントを取り扱う変数
	cudaEvent_t start, end;
	// イベントのクリエイト
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// コンスタントメモリにデータを送るために
	// ホストで 1 / (2.0 * dx)を計算する
	double h_idx2 = 1.0 / (2.0 * dx);

	host_u		= (double *)malloc(Nbytes);
	host_dudx	= (double *)malloc(Nbytes);
	cudaMalloc((void **)&u, Nbytes);
	cudaMalloc((void **)&dudx, Nbytes);

	init(host_u);
	cudaMemcpy(u, host_u, Nbytes, cudaMemcpyHostToDevice);
	
	// コンスタントメモリへデータを送る
	cudaMemcpyToSymbol(idx2, &h_idx2, sizeof(double));

	cudaEventRecord(start, 0);
	differentiate<<<NB, NT>>>(u, dudx);
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&elapsed_time_ms, start, end);
	std::cout << "Elapsed Time: " << elapsed_time_ms << std::endl;

	cudaMemcpy(host_dudx, dudx, Nbytes, cudaMemcpyDeviceToHost);

	cudaEventDestroy(start);
	cudaEventDestroy(end);
	free(host_u);
	free(host_dudx);
	cudaFree(u);
	cudaFree(dudx);

	return 0;
}


