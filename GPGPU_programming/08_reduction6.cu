#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N (256)
#define Nbytes (N*sizeof(int))
#define NT (N / 2)		// スレッド数を半分に
#define NB (N / NT / 2)		// スレッド数を半分にするとブロック数が増えてしまうので、
				// ブロック数を増やさないように調整(2で割る)

//共有メモリによるキャッシュ利用
// 1ブロック内のスレッド数を1/2に削減
// 1すれっどがグローバルメモリから2点のデータをコピー
// コピーの歳に加算を行い、共有メモリ使用量も1/2に削減する
// ただし、バンクコンフリクトが発生するプログラムになっている

__global__ void reduction6(int *idata, int *odata)
{
    // スレッドと配列の要素の対応
    int i = blockIdx.x * blockDim.x*2 + threadIdx.x;
    // スレッド番号
    int tx = threadIdx.x;
    int stride; // "隣"の配列要素まで距離

    // コンパイラの最適化を抑制
    // 複数のスレッドからアクセスされる変数に対する最適化
    // コンパイラが不要と判断して処理を削除してしまうことが有り、
    // 複数スレッドが変数の値をプライベートな領域にコピーして
    // 書き戻さない等が発生してしまう-->なのでvolatileを指定する
    
    // NTは定義の時点で1/2倍されている
    __shared__ volatile int s_idata[N]; // 共有メモリの宣言

    // グローバルメモリから共有メモリへデータをコピー
    // 1スレッドが2点読み込んで加算
    // blockDim.xは1スレッドが読み込む1点目と2点目の間隔
    s_idata[i] = idata[i] + idata[i + blockDim.x]; 
    __syncthreads(); // 共有メモリのデータは全スレッドから参照されるので同期を取る

    // ストライドをblockDim.x / 2からループ毎に1/2
    for (stride = blockDim.x / 2; stride >= 1; stride >>= 1)
    {
	if (tx < stride) // ストライドよりスレッド番号が小さいスレッドのみ計算に参加
	{
		s_idata[tx] += s_idata[tx + stride];
	}
        __syncthreads(); // スレッド間の同期を取る
    }
    if (tx == 0) // スレッド0が総和を出力用変数odataに書き込んで終了
    {
	odata[blockIdx.x] = s_idata[tx];
    }
}

void init(int *idata)
{
    int i;
    for (i=0; i<N; ++i)
    {
        idata[i] = 1;
    }
}

int main()
{
    // GPU用変数 idata: 入力、odata: 出力(総和)
    int *idata, *odata;

    // CPU用変数 host_idata: 初期化用、sum: 総和
    int *host_idata, sum;

    // 実行時間計測用
    float elapsed_time_ms = 0.0f; // 経過時間保存用
    // イベントを取り扱う変数
    cudaEvent_t start, end;
    // イベントのクリエイト
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaMalloc((void **)&idata, Nbytes);
    cudaMalloc((void **)&odata, sizeof(int));

    // CPU側でデータを初期化してGPUへコピー
    host_idata = (int *)malloc(Nbytes);
    init(host_idata);
    cudaMemcpy(idata, host_idata, Nbytes, cudaMemcpyHostToDevice);
    free(host_idata);

    cudaEventRecord(start, 0);
    reduction6<<<NB, NT>>>(idata, odata);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&elapsed_time_ms, start, end);
    std::cout << "Elapsed Time: " << elapsed_time_ms << std::endl;

    // GPUから総和の結果を受け取って画面表示
    cudaMemcpy(&sum, odata, sizeof(int), cudaMemcpyDeviceToHost);

    printf("sum = %d\n", sum);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(idata);
    cudaFree(odata);
    return 0;
}
