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


void reduction(int *, int *, int, int);

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
    // 共有メモリの宣言 externがつく
    // 共有メモリのサイズ指定が無くなっている
    extern __shared__ volatile int s_idata[];

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

    cudaEventRecord(start, 0);
    // 各ブロックで部分和を計算する
    reduction(idata, odata, NB, NT);        // 各ブロックで部分和を計算
    if (NB > 1)
    {
        reduction(odata, odata, 1, NB/2);   //部分和から総和を計算
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
    free(host_idata);
    return 0;
}

void reduction(int *idata, int *odata, int numBlock, int numThread)
{
    int smemSize = sizeof(int) * numThread;
    if (numThread <= 32)
    {
        smemSize += sizeof(int) * numThread / 2;
    }

    switch(numThread)
    {
        // numBlock, numThread, smemSizeはint型変数
        case 1024:
            reduction9<1024*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        case 512:
            reduction9< 512*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        case 256:
            reduction9< 256*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        case 128:
            reduction9< 128*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        case  64:
            reduction9<  64*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        case  32:
            reduction9<  32*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        case  16:
            reduction9<  16*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        case   8:
            reduction9<   8*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        case   4:
            reduction9<   4*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        case   2:
            reduction9<   2*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        case   1:
            reduction9<   1*2><<<numBlock, numThread, smemSize>>>(idata, odata);
            break;
        default:
            printf("configuration error\n");
    }
}


