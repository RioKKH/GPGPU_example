#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N (1024)
#define Nbytes (N*sizeof(int))
#define NPB (512)       // 1ブロックが部分和を計算する配列要素数 (Number of Points per Block)
#define NT (NPB / 2)    // スレッド数を半分に
#define NB (N / NPB) // スレッド数を半分にするとブロック数が増えてしまうので、
                        // ブロック数を増やさないように調整(2で割る)

// 共有メモリの要素数を決定
// NT >  32の時、1 * NT + 0 * (NT + NT / 2)
// NT <= 32の時、0 * NT + 1 * (NT + NT / 2)
// 関数形式マクロで共有メモリサイズを決定
#define smemSize(x) (((x) > 32)*x + ((x)<=32) * ((x) + (x)/2))



// blockArraySize(1ブロックが処理する配列要素数=スレッド数の2倍)をテンプレート
// パラメータとしてテンプレート関数を記述
template <unsigned int blockArraySize>
__global__ void reduction9(int *idata, int *odata)
{
    // スレッドと配列の要素の対応
    int i = blockIdx.x * blockDim.x*2 + threadIdx.x;
    // スレッド番号
    int tx = threadIdx.x;

    // 入力データの数に応じて必要な共有メモリサイズが変化
    __shared__ volatile int s_idata[smemSize(blockArraySize/2)]; // 共有メモリの宣言

    // 1ブロックあたりに処理する配列要素が1なら関数を終了
    if (blockArraySize == 1) return;

    // グローバルメモリから共有メモリへデータをコピー 1スレッドが2点読み込んで加算
    s_idata[i] = idata[i] + idata[i + blockDim.x]; 
    __syncthreads(); // 共有メモリのデータは全スレッドから参照されるので同期を取る

    // blockArraySizeはコンパイル時に確定するので、if 文は最適化される
    // このプログラムでは常に1024になっている
    if (blockArraySize >= 2048)
    {
        if (tx < 512) { s_idata[tx] += s_idata[tx + 512]; } __syncthreads();
    }
    // もしblockArraySizeが1024以上の場合
    if (blockArraySize >= 1024)
    {
        // txが256より小さいスレッドについてreductionを行いsyncthreadする
        if (tx < 256) { s_idata[tx] += s_idata[tx + 256]; } __syncthreads();
    }
    // blocksizeは1024なので、以下も実行される
    if (blockArraySize >= 512)
    {
        // txが128より小さいスレッドについてreductionを行いsyncthreadする
        if (tx < 128) { s_idata[tx] += s_idata[tx + 128]; } __syncthreads();
    }
    // blocksizeは1024なので、以下も実行される
    if (blockArraySize >= 256)
    {
        if (tx <  64) { s_idata[tx] += s_idata[tx +  64]; } __syncthreads();
    }
    // blocksizeは1024なので、以下も実行される
    if (tx < 32)
    {
        // 同一ワープ内では__syncthreads()が必要ない！
        if (blockArraySize >= 128) s_idata[tx] += s_idata[tx + 32];
        if (blockArraySize >=  64) s_idata[tx] += s_idata[tx + 16];
        if (blockArraySize >=  32) s_idata[tx] += s_idata[tx +  8];
        if (blockArraySize >=  16) s_idata[tx] += s_idata[tx +  4];
        if (blockArraySize >=   8) s_idata[tx] += s_idata[tx +  2];
        if (blockArraySize >=   4) s_idata[tx] += s_idata[tx +  1];
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
    cudaMalloc((void **)&odata, NB*sizeof(int));

    // CPU側でデータを初期化してGPUへコピー
    host_idata = (int *)malloc(Nbytes);
    init(host_idata);
    cudaMemcpy(idata, host_idata, Nbytes, cudaMemcpyHostToDevice);
    free(host_idata);

    cudaEventRecord(start, 0);

    // 各ブロックで部分和を計算する
    reduction9<NPB><<<NB, NT>>>(idata, odata);
    if (NB > 1)
    {
        // 部分和から総和を計算
        reduction9<NB ><<<1, NB/2>>>(odata, odata);
    }
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
